from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import queue
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .contracts import ContractError, canonical_json_bytes, utc_now, validate_record


TELEMETRY_SCHEMA_VERSION = "aptus.experiment-telemetry-sample.v1"
NANOSECONDS_PER_SECOND = 1_000_000_000
SAMPLE_INTERVAL_SECONDS = 1
MINIMUM_QUALIFYING_COVERAGE = 0.99
MAXIMUM_QUALIFYING_GAP_SECONDS = 2.5
MIB = 1024**2
GIB = 1024**3

_BYTE_UNITS = {
    "B": 1,
    "kB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
    "TiB": 1024**4,
}

_PROBE_FIELDS = frozenset({"gpu", "host"})
_GPU_PROBE_FIELDS = frozenset(
    {
        "uuid",
        "memory_used",
        "memory_free",
        "memory_total",
        "utilization_percent",
        "temperature_c",
        "power_draw_w",
        "power_limit_w",
        "graphics_clock_mhz",
        "memory_clock_mhz",
        "performance_state",
        "throttle_reasons",
        "throttle_state",
        "xid_errors",
        "reset_detected",
        "device_lost",
        "hardware_error",
        "compute_processes",
    }
)
_GPU_SAMPLE_FIELDS = frozenset(
    (_GPU_PROBE_FIELDS - {"memory_used", "memory_free", "memory_total"}) | {"memory"}
)
_HOST_FIELDS = frozenset(
    {
        "mem_available_bytes",
        "swap_used_bytes",
        "swap_read_bytes",
        "swap_write_bytes",
        "load_1m",
        "filesystem_free_bytes",
        "managed_process_rss_bytes",
        "managed_process_cpu_seconds",
        "managed_process_read_bytes",
        "managed_process_write_bytes",
        "disk_growth_bytes",
        "aptus_lease_active",
        "cpu_temperature",
        "nvme_temperature",
    }
)
_MEMORY_SOURCE_FIELDS = frozenset({"value", "unit"})
_MEMORY_SAMPLE_FIELDS = frozenset({"source_value", "source_unit", "bytes"})
_PROCESS_PROBE_FIELDS = frozenset({"pid", "used_memory", "managed"})
_PROCESS_SAMPLE_FIELDS = frozenset({"pid", "used_memory_bytes", "managed"})
_COLLECTOR_FIELDS = frozenset({"healthy", "status_code", "probe_duration_ns"})
_WATCHDOG_FIELDS = frozenset({"healthy", "heartbeat_monotonic_ns", "ownership_certain"})
_OPTIONAL_SENSOR_FIELDS = frozenset({"status", "value", "reason_code"})
_THROTTLE_REASON_CODES = frozenset(
    {
        "GPU_IDLE",
        "APPLICATION_CLOCKS_SETTING",
        "SW_POWER_CAP",
        "HW_SLOWDOWN",
        "SYNC_BOOST",
        "SW_THERMAL_SLOWDOWN",
        "HW_THERMAL_SLOWDOWN",
        "HW_POWER_BRAKE_SLOWDOWN",
        "DISPLAY_CLOCK_SETTING",
        "CLOCK_EVENT_ACTIVE_OTHER",
    }
)
_THERMAL_STOP_REASONS = frozenset(
    {
        "HW_SLOWDOWN",
        "SW_THERMAL_SLOWDOWN",
        "HW_THERMAL_SLOWDOWN",
        "HW_POWER_BRAKE_SLOWDOWN",
        "CLOCK_EVENT_ACTIVE_OTHER",
    }
)
_SAMPLE_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "experiment_run_id",
        "scheduled_slot",
        "scheduled_monotonic_ns",
        "observed_monotonic_ns",
        "wall_time_utc",
        "sample_interval_seconds",
        "gpu",
        "host",
        "collector",
        "watchdog",
    }
)

IMMEDIATE_STOP_REASON_CODES = frozenset(
    {
        "CUDA_OOM",
        "CUDA_XID",
        "CUDA_DEVICE_RESET",
        "CUDA_DEVICE_LOST",
        "HARDWARE_ERROR",
        "NONFINITE_VALUE",
        "ARTIFACT_INTEGRITY_FAILURE",
        "CHECKPOINT_CONTINUATION_FAILURE",
        "EXPORT_VERIFICATION_FAILURE",
        "THERMAL_THROTTLE",
        "THERMAL_LIMIT_DISAPPEARED",
        "UNRELATED_GPU_ACTIVITY",
        "TELEMETRY_COLLECTOR_FAILURE",
        "WATCHDOG_HEARTBEAT_LOST",
        "OWNERSHIP_UNCERTAIN",
        "DISK_BUDGET_INSUFFICIENT",
        "EMERGENCY_DEADLINE_EXCEEDED",
    }
)

_IMMEDIATE_PRIORITY = (
    "CUDA_OOM",
    "CUDA_XID",
    "CUDA_DEVICE_RESET",
    "CUDA_DEVICE_LOST",
    "HARDWARE_ERROR",
    "NONFINITE_VALUE",
    "ARTIFACT_INTEGRITY_FAILURE",
    "CHECKPOINT_CONTINUATION_FAILURE",
    "EXPORT_VERIFICATION_FAILURE",
    "THERMAL_THROTTLE",
    "THERMAL_LIMIT_DISAPPEARED",
    "UNRELATED_GPU_ACTIVITY",
    "TELEMETRY_COLLECTOR_FAILURE",
    "WATCHDOG_HEARTBEAT_LOST",
    "OWNERSHIP_UNCERTAIN",
    "DISK_BUDGET_INSUFFICIENT",
    "EMERGENCY_DEADLINE_EXCEEDED",
)


class TelemetryValidationError(ValueError):
    """A telemetry value failed the frozen, private raw-record contract."""


class ProbeFailure(RuntimeError):
    """A bounded probe failed without exposing its byte-exact exception."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HardwareProbe(Protocol):
    def __call__(self) -> Mapping[str, Any]: ...


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append(f"{len(extra)} unexpected field(s)")
        raise TelemetryValidationError(
            f"{label} fields are invalid: {'; '.join(details)}"
        )


def _require_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TelemetryValidationError(f"{label} must be a nonnegative integer")
    return value


def _require_positive_integer(value: Any, label: str) -> int:
    result = _require_nonnegative_integer(value, label)
    if result == 0:
        raise TelemetryValidationError(f"{label} must be positive")
    return result


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise TelemetryValidationError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise TelemetryValidationError(f"{label} must be a finite number") from None
    if not math.isfinite(result):
        raise TelemetryValidationError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise TelemetryValidationError(f"{label} is below its minimum")
    return result


def _normalize_optional_sensor(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TelemetryValidationError(f"{label} must be an object")
    _require_exact_fields(value, _OPTIONAL_SENSOR_FIELDS, label)
    status = value["status"]
    reason_code = value["reason_code"]
    if status == "supported":
        if reason_code is not None:
            raise TelemetryValidationError(
                f"{label} supported state cannot have a reason"
            )
        return {
            "status": "supported",
            "value": _finite_number(value["value"], f"{label}.value"),
            "reason_code": None,
        }
    if status != "unsupported" or value["value"] is not None:
        raise TelemetryValidationError(f"{label} has an invalid support state")
    if reason_code not in {"NOT_CONFIGURED", "UNAVAILABLE_AT_FREEZE"}:
        raise TelemetryValidationError(f"{label} has an invalid reason code")
    return {
        "status": "unsupported",
        "value": None,
        "reason_code": reason_code,
    }


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise TelemetryValidationError(
            f"{label} must use an integer, decimal string, or Decimal"
        )
    if not isinstance(value, (int, str, Decimal)):
        raise TelemetryValidationError(
            f"{label} must use an integer, decimal string, or Decimal"
        )
    if isinstance(value, str) and (not value or value != value.strip()):
        raise TelemetryValidationError(f"{label} is not a canonical decimal")
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError):
        raise TelemetryValidationError(f"{label} is not a valid decimal") from None
    if not result.is_finite() or result < 0:
        raise TelemetryValidationError(f"{label} must be finite and nonnegative")
    return result


def exact_bytes(value: int | str | Decimal, unit: str) -> int:
    """Convert an explicit decimal source value to bytes without rounding."""

    if unit not in _BYTE_UNITS:
        raise TelemetryValidationError("memory source unit is unsupported")
    converted = _decimal(value, "memory source value") * _BYTE_UNITS[unit]
    integral = converted.to_integral_value()
    if converted != integral:
        raise TelemetryValidationError("memory source value does not convert exactly")
    return int(integral)


def _normalize_memory_source(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TelemetryValidationError(f"{label} must be an object")
    _require_exact_fields(value, _MEMORY_SOURCE_FIELDS, label)
    source_value = value["value"]
    unit = value["unit"]
    if not isinstance(unit, str):
        raise TelemetryValidationError(f"{label}.unit must be a string")
    byte_count = exact_bytes(source_value, unit)
    return {
        "source_value": str(source_value),
        "source_unit": unit,
        "bytes": byte_count,
    }


def _validate_normalized_memory(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TelemetryValidationError(f"{label} must be an object")
    _require_exact_fields(value, _MEMORY_SAMPLE_FIELDS, label)
    source_value = value["source_value"]
    source_unit = value["source_unit"]
    byte_count = _require_nonnegative_integer(value["bytes"], f"{label}.bytes")
    if not isinstance(source_value, str) or not isinstance(source_unit, str):
        raise TelemetryValidationError(f"{label} source metadata is invalid")
    if exact_bytes(source_value, source_unit) != byte_count:
        raise TelemetryValidationError(f"{label} normalized byte count is invalid")
    return dict(value)


def _normalize_processes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TelemetryValidationError("gpu.compute_processes must be a list")
    result: list[dict[str, Any]] = []
    seen_pids: set[int] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise TelemetryValidationError(
                "gpu.compute_processes entries must be objects"
            )
        _require_exact_fields(item, _PROCESS_PROBE_FIELDS, "gpu.compute_process")
        pid = _require_positive_integer(item["pid"], "gpu.compute_process.pid")
        if pid in seen_pids:
            raise TelemetryValidationError(
                "gpu.compute_processes contains a duplicate PID"
            )
        seen_pids.add(pid)
        if not isinstance(item["managed"], bool):
            raise TelemetryValidationError(
                "gpu.compute_process.managed must be boolean"
            )
        memory = _normalize_memory_source(
            item["used_memory"], "gpu.compute_process.used_memory"
        )
        result.append(
            {
                "pid": pid,
                "used_memory_bytes": memory["bytes"],
                "managed": item["managed"],
            }
        )
    return sorted(result, key=lambda item: item["pid"])


def _validate_processes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TelemetryValidationError("gpu.compute_processes must be a list")
    result: list[dict[str, Any]] = []
    seen_pids: set[int] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise TelemetryValidationError(
                "gpu.compute_processes entries must be objects"
            )
        _require_exact_fields(item, _PROCESS_SAMPLE_FIELDS, "gpu.compute_process")
        pid = _require_positive_integer(item["pid"], "gpu.compute_process.pid")
        if pid in seen_pids:
            raise TelemetryValidationError(
                "gpu.compute_processes contains a duplicate PID"
            )
        seen_pids.add(pid)
        memory = _require_nonnegative_integer(
            item["used_memory_bytes"], "gpu.compute_process.used_memory_bytes"
        )
        if not isinstance(item["managed"], bool):
            raise TelemetryValidationError(
                "gpu.compute_process.managed must be boolean"
            )
        result.append(
            {"pid": pid, "used_memory_bytes": memory, "managed": item["managed"]}
        )
    return result


def _normalize_gpu(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TelemetryValidationError("gpu probe output must be an object")
    _require_exact_fields(value, _GPU_PROBE_FIELDS, "gpu probe")
    uuid = value["uuid"]
    if not isinstance(uuid, str) or not uuid.strip() or uuid != uuid.strip():
        raise TelemetryValidationError("gpu.uuid must be a nonempty protected string")
    used = _normalize_memory_source(value["memory_used"], "gpu.memory_used")
    free = _normalize_memory_source(value["memory_free"], "gpu.memory_free")
    total = _normalize_memory_source(value["memory_total"], "gpu.memory_total")
    if total["bytes"] - used["bytes"] != free["bytes"]:
        raise TelemetryValidationError(
            "GPU memory used, free, and total do not reconcile"
        )
    utilization = _finite_number(
        value["utilization_percent"], "gpu.utilization_percent", minimum=0
    )
    if utilization > 100:
        raise TelemetryValidationError("gpu.utilization_percent exceeds 100")
    throttle_reasons = value["throttle_reasons"]
    if not isinstance(throttle_reasons, list) or any(
        item not in _THROTTLE_REASON_CODES for item in throttle_reasons
    ):
        raise TelemetryValidationError("gpu.throttle_reasons contains an invalid code")
    xid_errors = value["xid_errors"]
    if not isinstance(xid_errors, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in xid_errors
    ):
        raise TelemetryValidationError("gpu.xid_errors must be nonnegative integers")
    performance_state = value["performance_state"]
    if not isinstance(performance_state, str) or not performance_state:
        raise TelemetryValidationError(
            "gpu.performance_state must be a nonempty string"
        )
    throttle_state = value["throttle_state"]
    if not isinstance(throttle_state, str) or not re.fullmatch(
        r"(?:0x[0-9A-Fa-f]{1,16}|Active|Not Active)", throttle_state
    ):
        raise TelemetryValidationError("gpu.throttle_state is invalid")
    throttle_active = (
        int(throttle_state, 16) != 0
        if throttle_state.startswith("0x")
        else throttle_state == "Active"
    )
    if throttle_active != bool(throttle_reasons):
        raise TelemetryValidationError("gpu throttle state and reasons disagree")
    flags: dict[str, bool] = {}
    for field_name in ("reset_detected", "device_lost", "hardware_error"):
        flag = value[field_name]
        if not isinstance(flag, bool):
            raise TelemetryValidationError(f"gpu.{field_name} must be boolean")
        flags[field_name] = flag
    return {
        "uuid": uuid,
        "memory": {"used": used, "free": free, "total": total},
        "utilization_percent": utilization,
        "temperature_c": _finite_number(value["temperature_c"], "gpu.temperature_c"),
        "power_draw_w": _finite_number(
            value["power_draw_w"], "gpu.power_draw_w", minimum=0
        ),
        "power_limit_w": _finite_number(
            value["power_limit_w"], "gpu.power_limit_w", minimum=0
        ),
        "graphics_clock_mhz": _finite_number(
            value["graphics_clock_mhz"], "gpu.graphics_clock_mhz", minimum=0
        ),
        "memory_clock_mhz": _finite_number(
            value["memory_clock_mhz"], "gpu.memory_clock_mhz", minimum=0
        ),
        "performance_state": performance_state,
        "throttle_reasons": list(throttle_reasons),
        "throttle_state": throttle_state,
        "xid_errors": list(xid_errors),
        **flags,
        "compute_processes": _normalize_processes(value["compute_processes"]),
    }


def _validate_gpu(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TelemetryValidationError("gpu sample must be an object")
    _require_exact_fields(value, _GPU_SAMPLE_FIELDS, "gpu sample")
    uuid = value["uuid"]
    if not isinstance(uuid, str) or not uuid.strip() or uuid != uuid.strip():
        raise TelemetryValidationError("gpu.uuid must be a nonempty protected string")
    memory = value["memory"]
    if not isinstance(memory, Mapping):
        raise TelemetryValidationError("gpu.memory must be an object")
    _require_exact_fields(memory, frozenset({"used", "free", "total"}), "gpu.memory")
    used = _validate_normalized_memory(memory["used"], "gpu.memory.used")
    free = _validate_normalized_memory(memory["free"], "gpu.memory.free")
    total = _validate_normalized_memory(memory["total"], "gpu.memory.total")
    if total["bytes"] - used["bytes"] != free["bytes"]:
        raise TelemetryValidationError(
            "GPU memory used, free, and total do not reconcile"
        )
    utilization = _finite_number(
        value["utilization_percent"], "gpu.utilization_percent", minimum=0
    )
    if utilization > 100:
        raise TelemetryValidationError("gpu.utilization_percent exceeds 100")
    for field_name in (
        "temperature_c",
        "power_draw_w",
        "power_limit_w",
        "graphics_clock_mhz",
        "memory_clock_mhz",
    ):
        _finite_number(value[field_name], f"gpu.{field_name}", minimum=0)
    if (
        not isinstance(value["performance_state"], str)
        or not value["performance_state"]
    ):
        raise TelemetryValidationError(
            "gpu.performance_state must be a nonempty string"
        )
    if not isinstance(value["throttle_reasons"], list) or any(
        item not in _THROTTLE_REASON_CODES for item in value["throttle_reasons"]
    ):
        raise TelemetryValidationError("gpu.throttle_reasons contains an invalid code")
    if not isinstance(value["throttle_state"], str) or not re.fullmatch(
        r"(?:0x[0-9A-Fa-f]{1,16}|Active|Not Active)", value["throttle_state"]
    ):
        raise TelemetryValidationError("gpu.throttle_state is invalid")
    throttle_active = (
        int(value["throttle_state"], 16) != 0
        if value["throttle_state"].startswith("0x")
        else value["throttle_state"] == "Active"
    )
    if throttle_active != bool(value["throttle_reasons"]):
        raise TelemetryValidationError("gpu throttle state and reasons disagree")
    if not isinstance(value["xid_errors"], list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value["xid_errors"]
    ):
        raise TelemetryValidationError("gpu.xid_errors must be nonnegative integers")
    for field_name in ("reset_detected", "device_lost", "hardware_error"):
        if not isinstance(value[field_name], bool):
            raise TelemetryValidationError(f"gpu.{field_name} must be boolean")
    _validate_processes(value["compute_processes"])
    return dict(value)


def _normalize_host(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TelemetryValidationError("host probe output must be an object")
    _require_exact_fields(value, _HOST_FIELDS, "host probe")
    result: dict[str, Any] = {}
    for field_name in (
        "mem_available_bytes",
        "swap_used_bytes",
        "swap_read_bytes",
        "swap_write_bytes",
        "filesystem_free_bytes",
        "managed_process_rss_bytes",
        "managed_process_read_bytes",
        "managed_process_write_bytes",
        "disk_growth_bytes",
    ):
        result[field_name] = _require_nonnegative_integer(
            value[field_name], f"host.{field_name}"
        )
    result["load_1m"] = _finite_number(value["load_1m"], "host.load_1m", minimum=0)
    result["managed_process_cpu_seconds"] = _finite_number(
        value["managed_process_cpu_seconds"],
        "host.managed_process_cpu_seconds",
        minimum=0,
    )
    if not isinstance(value["aptus_lease_active"], bool):
        raise TelemetryValidationError("host.aptus_lease_active must be boolean")
    result["aptus_lease_active"] = value["aptus_lease_active"]
    result["cpu_temperature"] = _normalize_optional_sensor(
        value["cpu_temperature"], "host.cpu_temperature"
    )
    result["nvme_temperature"] = _normalize_optional_sensor(
        value["nvme_temperature"], "host.nvme_temperature"
    )
    return result


def _validate_collector(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TelemetryValidationError("collector must be an object")
    _require_exact_fields(value, _COLLECTOR_FIELDS, "collector")
    if not isinstance(value["healthy"], bool):
        raise TelemetryValidationError("collector.healthy must be boolean")
    status = value["status_code"]
    allowed_statuses = {
        "COLLECTOR_FAILED",
        "COLLECTOR_STOPPED",
        "PROBE_FAILED",
        "PROBE_INVALID",
        "PROBE_TIMEOUT",
        "TELEMETRY_SAMPLE_INVALID",
    }
    if status is not None and status not in allowed_statuses:
        raise TelemetryValidationError("collector.status_code is invalid")
    if value["healthy"] != (status is None):
        raise TelemetryValidationError("collector health and status code disagree")
    _require_nonnegative_integer(
        value["probe_duration_ns"], "collector.probe_duration_ns"
    )
    return dict(value)


def _validate_watchdog(value: Any, observed_monotonic_ns: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TelemetryValidationError("watchdog must be an object")
    _require_exact_fields(value, _WATCHDOG_FIELDS, "watchdog")
    if not isinstance(value["healthy"], bool):
        raise TelemetryValidationError("watchdog.healthy must be boolean")
    if not isinstance(value["ownership_certain"], bool):
        raise TelemetryValidationError("watchdog.ownership_certain must be boolean")
    heartbeat = _require_nonnegative_integer(
        value["heartbeat_monotonic_ns"], "watchdog.heartbeat_monotonic_ns"
    )
    if heartbeat > observed_monotonic_ns:
        raise TelemetryValidationError("watchdog heartbeat cannot be in the future")
    return dict(value)


def normalize_observation_facts(
    *,
    probe_reading: Mapping[str, Any],
    collector: Mapping[str, Any],
    watchdog: Mapping[str, Any],
    observed_monotonic_ns: int,
) -> dict[str, Any]:
    """Normalize the shared GPU/host/health facts for any strict envelope.

    Experiment telemetry and pre-slot admission use different identity
    envelopes.  This helper deliberately owns only their identical normalized
    observation facts, so admission never needs a synthetic experiment-run ID.
    """

    if not isinstance(probe_reading, Mapping):
        raise TelemetryValidationError("probe reading must be an object")
    _require_exact_fields(probe_reading, _PROBE_FIELDS, "probe reading")
    observed = _require_nonnegative_integer(
        observed_monotonic_ns, "observed_monotonic_ns"
    )
    return {
        "gpu": _normalize_gpu(probe_reading["gpu"]),
        "host": _normalize_host(probe_reading["host"]),
        "collector": _validate_collector(collector),
        "watchdog": _validate_watchdog(watchdog, observed),
    }


def validate_observation_facts(
    value: Mapping[str, Any], *, observed_monotonic_ns: int
) -> dict[str, Any]:
    """Validate and detach normalized facts shared by strict envelopes."""

    if not isinstance(value, Mapping):
        raise TelemetryValidationError("observation facts must be an object")
    _require_exact_fields(
        value,
        _PROBE_FIELDS | {"collector", "watchdog"},
        "observation facts",
    )
    observed = _require_nonnegative_integer(
        observed_monotonic_ns, "observed_monotonic_ns"
    )
    return {
        "gpu": _validate_gpu(value["gpu"]),
        "host": _normalize_host(value["host"]),
        "collector": _validate_collector(value["collector"]),
        "watchdog": _validate_watchdog(value["watchdog"], observed),
    }


def construct_telemetry_sample(
    *,
    sequence: int,
    experiment_run_id: str,
    scheduled_slot: int,
    scheduled_monotonic_ns: int,
    observed_monotonic_ns: int,
    wall_time_utc: str,
    probe_reading: Mapping[str, Any],
    collector: Mapping[str, Any],
    watchdog: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a strict raw telemetry sample; arbitrary probe fields never pass through."""

    _require_nonnegative_integer(sequence, "sequence")
    _require_nonnegative_integer(scheduled_slot, "scheduled_slot")
    _require_nonnegative_integer(scheduled_monotonic_ns, "scheduled_monotonic_ns")
    _require_nonnegative_integer(observed_monotonic_ns, "observed_monotonic_ns")
    if observed_monotonic_ns < scheduled_monotonic_ns:
        raise TelemetryValidationError("observed time precedes the scheduled time")
    if not isinstance(experiment_run_id, str) or not experiment_run_id:
        raise TelemetryValidationError("experiment_run_id must be a nonempty string")
    if not isinstance(wall_time_utc, str) or not wall_time_utc:
        raise TelemetryValidationError("wall_time_utc must be a nonempty string")
    facts = normalize_observation_facts(
        probe_reading=probe_reading,
        collector=collector,
        watchdog=watchdog,
        observed_monotonic_ns=observed_monotonic_ns,
    )
    sample = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "sequence": sequence,
        "experiment_run_id": experiment_run_id,
        "scheduled_slot": scheduled_slot,
        "scheduled_monotonic_ns": scheduled_monotonic_ns,
        "observed_monotonic_ns": observed_monotonic_ns,
        "wall_time_utc": wall_time_utc,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        **facts,
    }
    return validate_telemetry_sample(sample)


def validate_telemetry_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(sample, Mapping):
        raise TelemetryValidationError("telemetry sample must be an object")
    _require_exact_fields(sample, _SAMPLE_FIELDS, "telemetry sample")
    validated = dict(sample)
    if validated["schema_version"] != TELEMETRY_SCHEMA_VERSION:
        raise TelemetryValidationError("telemetry schema version is unsupported")
    if (
        not isinstance(validated["experiment_run_id"], str)
        or not validated["experiment_run_id"]
    ):
        raise TelemetryValidationError("experiment_run_id must be a nonempty string")
    if (
        not isinstance(validated["wall_time_utc"], str)
        or not validated["wall_time_utc"]
    ):
        raise TelemetryValidationError("wall_time_utc must be a nonempty string")
    sequence = _require_nonnegative_integer(validated["sequence"], "sequence")
    scheduled_slot = _require_nonnegative_integer(
        validated["scheduled_slot"], "scheduled_slot"
    )
    scheduled = _require_nonnegative_integer(
        validated["scheduled_monotonic_ns"], "scheduled_monotonic_ns"
    )
    observed = _require_nonnegative_integer(
        validated["observed_monotonic_ns"], "observed_monotonic_ns"
    )
    if observed < scheduled:
        raise TelemetryValidationError("observed time precedes the scheduled time")
    if validated["sample_interval_seconds"] != SAMPLE_INTERVAL_SECONDS:
        raise TelemetryValidationError("telemetry sample interval must be one second")
    validate_observation_facts(
        {name: validated[name] for name in ("gpu", "host", "collector", "watchdog")},
        observed_monotonic_ns=observed,
    )
    try:
        validate_record(validated, TELEMETRY_SCHEMA_VERSION)
    except ContractError as error:
        raise TelemetryValidationError(
            "telemetry sample violates the canonical campaign envelope"
        ) from error
    if sequence < 0 or scheduled_slot < 0:  # keeps the validated facts visibly used
        raise AssertionError("unreachable")
    return validated


def telemetry_sample_bytes(sample: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(validate_telemetry_sample(sample))


def _bounded_probe(probe: Callable[[], Any], timeout_seconds: float) -> Any:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("probe timeout must be positive and finite")
    results: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            results.put_nowait((True, probe()))
        except BaseException:
            try:
                results.put_nowait((False, None))
            except queue.Full:
                pass

    worker = threading.Thread(target=invoke, name="aptus-telemetry-probe", daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise ProbeFailure("PROBE_TIMEOUT")
    try:
        succeeded, value = results.get_nowait()
    except queue.Empty:
        raise ProbeFailure("PROBE_FAILED") from None
    if not succeeded:
        raise ProbeFailure("PROBE_FAILED")
    return value


@dataclass(frozen=True)
class ProbeCommandResult:
    returncode: int
    stdout: str


@dataclass(frozen=True)
class TrustedExecutable:
    """Resolved executable identity accepted by the qualifying host factory."""

    path: str
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    binding_sha256: str

    def verify(self) -> str:
        try:
            metadata = Path(self.path).stat()
        except OSError:
            raise ProbeFailure("NVIDIA_SMI_IDENTITY_CHANGED") from None
        observed = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        expected = (self.device, self.inode, self.mode, self.size, self.modified_ns)
        if observed != expected:
            raise ProbeFailure("NVIDIA_SMI_IDENTITY_CHANGED")
        return self.path


def resolve_trusted_nvidia_smi(value: str | None) -> TrustedExecutable:
    """Resolve and pin one trustworthy executable without retaining a public path."""

    candidate = value or shutil.which("nvidia-smi")
    if not isinstance(candidate, str) or not candidate:
        raise ProbeFailure("NVIDIA_SMI_UNAVAILABLE")
    try:
        path = Path(candidate).resolve(strict=True)
        metadata = path.stat()
    except OSError:
        raise ProbeFailure("NVIDIA_SMI_UNAVAILABLE") from None
    if (
        not path.is_absolute()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o022
        or not metadata.st_mode & 0o111
        or (sys.platform.startswith("linux") and metadata.st_uid != 0)
    ):
        raise ProbeFailure("NVIDIA_SMI_UNTRUSTED")
    identity = {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "modified_ns": metadata.st_mtime_ns,
        "path": str(path),
        "size": metadata.st_size,
    }
    binding = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return TrustedExecutable(
        path=str(path),
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        binding_sha256=binding,
    )


def resolve_trusted_journalctl(value: str | None = None) -> TrustedExecutable:
    """Resolve the root-owned journal reader used for qualifying boot evidence."""

    candidate = value or shutil.which("journalctl")
    if not isinstance(candidate, str) or not candidate:
        raise ProbeFailure("JOURNALCTL_UNAVAILABLE")
    try:
        path = Path(candidate).resolve(strict=True)
        metadata = path.stat()
    except OSError:
        raise ProbeFailure("JOURNALCTL_UNAVAILABLE") from None
    if (
        not path.is_absolute()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o022
        or not metadata.st_mode & 0o111
        or (sys.platform.startswith("linux") and metadata.st_uid != 0)
    ):
        raise ProbeFailure("JOURNALCTL_UNTRUSTED")
    identity = {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "modified_ns": metadata.st_mtime_ns,
        "path": str(path),
        "size": metadata.st_size,
    }
    binding = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return TrustedExecutable(
        path=str(path),
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        binding_sha256=binding,
    )


_KERNEL_EVENT_FIELDS = frozenset(
    {"xid_errors", "reset_detected", "device_lost", "hardware_error"}
)
_JOURNAL_MAX_BYTES = 256 * 1024
_JOURNAL_MAX_ENTRIES = 256
_JOURNAL_CURSOR = re.compile(r"^-- cursor: ([\x21-\x7e]{1,512})$")
_NVRM_XID = re.compile(
    r"\bNVRM:\s*Xid\s*\([^)]{1,128}\)\s*:\s*([0-9]{1,5})\b",
    re.IGNORECASE,
)
_RESET_XIDS = frozenset({45, 46, 79})
_DEVICE_LOST_XIDS = frozenset({79})
_HARDWARE_ERROR_XIDS = frozenset(
    {48, 56, 57, 58, 62, 63, 64, 74, 79, 92, 94, 95, 119, 120}
)

JOURNAL_BOOT_AUTHORITY_SCHEMA = "aptus.cuda-journal-boot-authority.v1"


@dataclass(frozen=True)
class JournalBootAuthority:
    """Protected current-boot and cursor authority for retained evidence."""

    boot_id_sha256: str
    journalctl_binding_sha256: str
    initial_cursor_sha256: str
    final_cursor_sha256: str
    initial_projection: Mapping[str, Any]
    final_projection: Mapping[str, Any]

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": JOURNAL_BOOT_AUTHORITY_SCHEMA,
            "boot_id_sha256": self.boot_id_sha256,
            "journalctl_binding_sha256": self.journalctl_binding_sha256,
            "initial_cursor_sha256": self.initial_cursor_sha256,
            "final_cursor_sha256": self.final_cursor_sha256,
            "initial_projection": dict(self.initial_projection),
            "final_projection": dict(self.final_projection),
        }


def _default_journal_runner(
    command: Sequence[str], timeout_seconds: float
) -> ProbeCommandResult:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ProbeFailure("JOURNALCTL_EXECUTION_FAILED") from None
    return ProbeCommandResult(completed.returncode, completed.stdout)


class LinuxNvidiaJournalEventProvider:
    """Current-boot, cursor-bounded NVIDIA kernel event projection."""

    def __init__(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        raise TypeError("Use LinuxNvidiaJournalEventProvider.production().")

    def _initialize(
        self,
        *,
        journalctl_path: str,
        boot_id: str,
        command_runner: Callable[
            [Sequence[str], float], ProbeCommandResult
        ] = _default_journal_runner,
        timeout_seconds: float = 0.25,
        trusted_journalctl: TrustedExecutable | None = None,
        verify_executable: bool = False,
    ) -> None:
        executable = Path(journalctl_path)
        if not executable.is_absolute() or not journalctl_path:
            raise ProbeFailure("JOURNALCTL_UNAVAILABLE")
        if type(trusted_journalctl) is not TrustedExecutable:
            raise ProbeFailure("JOURNALCTL_PROVENANCE_UNAVAILABLE")
        if trusted_journalctl.path != str(executable):
            raise ProbeFailure("JOURNALCTL_PROVENANCE_INVALID")
        normalized_boot_id = boot_id.replace("-", "").lower()
        if re.fullmatch(r"[0-9a-f]{32}", normalized_boot_id) is None:
            raise ProbeFailure("JOURNAL_BOOT_ID_INVALID")
        if (
            not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or timeout_seconds >= SAMPLE_INTERVAL_SECONDS
        ):
            raise ValueError("journal timeout must be positive and below one second")
        self._journalctl_path = str(executable)
        self._trusted_journalctl = trusted_journalctl
        self._verify_executable = bool(verify_executable)
        self._boot_id = normalized_boot_id
        self._command_runner = command_runner
        self._timeout_seconds = float(timeout_seconds)
        self._lock = threading.Lock()
        historical, self._cursor = self._query(
            lines=_JOURNAL_MAX_ENTRIES + 1,
            require_event=True,
            reject_full=True,
            filter_nvidia=True,
        )
        self._historical_projection = self._project(historical)
        self._initial_cursor = self._cursor
        self._initial_projection = dict(self._historical_projection)
        self._final_cursor = self._cursor
        self._final_projection = dict(self._historical_projection)

    @classmethod
    def production(cls) -> "LinuxNvidiaJournalEventProvider":
        """Use the real journalctl subprocess and the running kernel boot ID."""

        if cls is not LinuxNvidiaJournalEventProvider:
            raise TypeError("Production journal provider subclasses are forbidden.")
        return _create_production_journal_provider()

    @classmethod
    def _for_test(
        cls,
        *,
        journalctl_path: str,
        boot_id: str,
        command_runner: Callable[[Sequence[str], float], ProbeCommandResult],
        journalctl_binding_sha256: str | None = None,
    ) -> "LinuxNvidiaJournalEventProvider":
        binding = (
            journalctl_binding_sha256
            or hashlib.sha256(journalctl_path.encode("utf-8")).hexdigest()
        )
        trusted = TrustedExecutable(
            path=journalctl_path,
            device=0,
            inode=0,
            mode=0,
            size=0,
            modified_ns=0,
            binding_sha256=binding,
        )
        if cls is not LinuxNvidiaJournalEventProvider:
            raise TypeError("Test journal provider subclasses are forbidden.")
        return _create_nonproduction_journal_provider(
            journalctl_path=journalctl_path,
            boot_id=boot_id,
            command_runner=command_runner,
            trusted_journalctl=trusted,
        )

    @property
    def production_authorized(self) -> bool:
        """Whether this exact instance came from the concrete production factory."""

        return _production_journal_provider_is_authentic(self)

    def _verify_journalctl_identity(self) -> None:
        if not self._verify_executable:
            return
        trusted = self._trusted_journalctl
        try:
            metadata = Path(trusted.path).stat()
        except OSError:
            raise ProbeFailure("JOURNALCTL_IDENTITY_CHANGED") from None
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
        ) != (
            trusted.device,
            trusted.inode,
            trusted.mode,
            trusted.size,
            trusted.modified_ns,
        ):
            raise ProbeFailure("JOURNALCTL_IDENTITY_CHANGED")

    def _command(
        self,
        *,
        lines: int,
        cursor: str | None = None,
        filter_nvidia: bool = False,
    ) -> tuple[str, ...]:
        command = [
            self._journalctl_path,
            f"--boot={self._boot_id}",
            "--dmesg",
            "--no-pager",
            "--quiet",
            "--output=json",
            "--output-fields=MESSAGE,_BOOT_ID,_TRANSPORT",
            f"--lines={lines}",
            "--show-cursor",
        ]
        if filter_nvidia:
            command.append("--grep=(NVRM|nvidia|nouveau)")
        if cursor is not None:
            command.append(f"--after-cursor={cursor}")
        return tuple(command)

    def _query(
        self,
        *,
        lines: int,
        cursor: str | None = None,
        require_event: bool = False,
        reject_full: bool = False,
        filter_nvidia: bool = False,
    ) -> tuple[list[str], str]:
        self._verify_journalctl_identity()
        try:
            result = self._command_runner(
                self._command(
                    lines=lines,
                    cursor=cursor,
                    filter_nvidia=filter_nvidia,
                ),
                self._timeout_seconds,
            )
        except ProbeFailure:
            raise
        except Exception:
            raise ProbeFailure("JOURNALCTL_EXECUTION_FAILED") from None
        if (
            type(result) is not ProbeCommandResult
            or result.returncode != 0
            or not isinstance(result.stdout, str)
        ):
            raise ProbeFailure("JOURNALCTL_QUERY_FAILED")
        try:
            payload = result.stdout.encode("utf-8")
        except UnicodeEncodeError:
            raise ProbeFailure("JOURNAL_OUTPUT_INVALID") from None
        if len(payload) > _JOURNAL_MAX_BYTES:
            raise ProbeFailure("JOURNAL_OUTPUT_LIMIT_EXCEEDED")
        rows = [row for row in result.stdout.splitlines() if row]
        cursor_rows = [row for row in rows if row.startswith("-- cursor:")]
        if len(cursor_rows) != 1:
            raise ProbeFailure("JOURNAL_CURSOR_INVALID")
        match = _JOURNAL_CURSOR.fullmatch(cursor_rows[0])
        if match is None:
            raise ProbeFailure("JOURNAL_CURSOR_INVALID")
        event_rows = [row for row in rows if not row.startswith("-- cursor:")]
        if len(event_rows) > _JOURNAL_MAX_ENTRIES or (
            reject_full and lines and len(event_rows) >= lines
        ):
            raise ProbeFailure("JOURNAL_OUTPUT_LIMIT_EXCEEDED")
        if require_event and not event_rows:
            raise ProbeFailure("JOURNAL_KERNEL_ACCESS_UNPROVEN")
        messages: list[str] = []
        for row in event_rows:
            try:
                value = json.loads(row)
            except json.JSONDecodeError:
                raise ProbeFailure("JOURNAL_OUTPUT_INVALID") from None
            if (
                type(value) is not dict
                or value.get("_BOOT_ID", "").lower() != self._boot_id
                or value.get("_TRANSPORT") != "kernel"
                or not isinstance(value.get("MESSAGE"), str)
                or len(value["MESSAGE"].encode("utf-8")) > 16 * 1024
            ):
                raise ProbeFailure("JOURNAL_OUTPUT_INVALID")
            messages.append(value["MESSAGE"])
        return messages, match.group(1)

    @staticmethod
    def _project(messages: Sequence[str]) -> dict[str, Any]:
        xids = sorted(
            {
                int(match.group(1))
                for message in messages
                for match in _NVRM_XID.finditer(message)
            }
        )
        lowered = "\n".join(messages).lower()
        return {
            "xid_errors": xids,
            "reset_detected": bool(_RESET_XIDS.intersection(xids))
            or "resetting gpu" in lowered
            or "gpu reset required" in lowered,
            "device_lost": bool(_DEVICE_LOST_XIDS.intersection(xids))
            or "fallen off the bus" in lowered
            or "device lost" in lowered,
            "hardware_error": bool(_HARDWARE_ERROR_XIDS.intersection(xids))
            or "uncorrectable hardware error" in lowered,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return sticky current-boot NVIDIA history plus cursor advances."""

        with self._lock:
            messages, cursor = self._query(
                lines=_JOURNAL_MAX_ENTRIES + 1,
                cursor=self._cursor,
                reject_full=True,
                filter_nvidia=True,
            )
            self._cursor = cursor
            current = self._project(messages)
            historical = self._historical_projection
            combined = {
                "xid_errors": sorted(
                    set(historical["xid_errors"]) | set(current["xid_errors"])
                ),
                "reset_detected": historical["reset_detected"]
                or current["reset_detected"],
                "device_lost": historical["device_lost"] or current["device_lost"],
                "hardware_error": historical["hardware_error"]
                or current["hardware_error"],
            }
            self._historical_projection = combined
            self._final_cursor = cursor
            self._final_projection = dict(combined)
            return dict(combined)

    @property
    def boot_id_sha256(self) -> str:
        return hashlib.sha256(self._boot_id.encode("ascii")).hexdigest()

    @property
    def journalctl_binding_sha256(self) -> str:
        return self._trusted_journalctl.binding_sha256

    def support_bindings(self) -> dict[str, str]:
        suffix = (
            f"boot-sha256:{self.boot_id_sha256}:"
            f"journalctl-sha256:{self.journalctl_binding_sha256}"
        )
        return {
            "hardware_events": f"journalctl-current-boot-cursor-v1:{suffix}",
            "xid_projection": f"journalctl-nvrm-xid-v1:{suffix}",
        }

    def authority(self) -> JournalBootAuthority:
        """Return the cursor-bracketed authority after at least one final query."""

        return JournalBootAuthority(
            boot_id_sha256=self.boot_id_sha256,
            journalctl_binding_sha256=self.journalctl_binding_sha256,
            initial_cursor_sha256=hashlib.sha256(
                self._initial_cursor.encode("utf-8")
            ).hexdigest(),
            final_cursor_sha256=hashlib.sha256(
                self._final_cursor.encode("utf-8")
            ).hexdigest(),
            initial_projection=dict(self._initial_projection),
            final_projection=dict(self._final_projection),
        )


class StatvfsDiskGrowthProvider:
    """Measure campaign filesystem consumption from one production baseline."""

    def __init__(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        raise TypeError("Use StatvfsDiskGrowthProvider.production().")

    def _initialize(
        self,
        path: Path,
        *,
        statvfs: Callable[[os.PathLike[str] | str], os.statvfs_result],
    ) -> None:
        self._path = path.resolve(strict=True)
        self._statvfs = statvfs
        self._baseline = self._available_bytes()

    @classmethod
    def production(cls, path: Path) -> "StatvfsDiskGrowthProvider":
        if cls is not StatvfsDiskGrowthProvider:
            raise TypeError("Production disk provider subclasses are forbidden.")
        return _create_production_disk_growth_provider(path)

    @property
    def production_authorized(self) -> bool:
        """Whether this exact instance came from the concrete production factory."""

        return _production_disk_growth_provider_is_authentic(self)

    def _available_bytes(self) -> int:
        try:
            value = self._statvfs(self._path)
            available = int(value.f_bavail) * int(value.f_frsize)
        except (OSError, TypeError, ValueError):
            raise ProbeFailure("DISK_GROWTH_PROVIDER_FAILED") from None
        if available < 0:
            raise ProbeFailure("DISK_GROWTH_PROVIDER_FAILED")
        return available

    def __call__(self) -> int:
        return max(0, self._baseline - self._available_bytes())


def _install_production_provider_factories() -> tuple[
    Callable[[], LinuxNvidiaJournalEventProvider],
    Callable[..., LinuxNvidiaJournalEventProvider],
    Callable[[object], bool],
    Callable[[Path], StatvfsDiskGrowthProvider],
    Callable[[object], bool],
]:
    """Keep provider authenticity registries outside importable module state."""

    journals: weakref.WeakKeyDictionary[
        LinuxNvidiaJournalEventProvider, tuple[object, ...]
    ] = weakref.WeakKeyDictionary()
    disks: weakref.WeakKeyDictionary[StatvfsDiskGrowthProvider, tuple[object, ...]] = (
        weakref.WeakKeyDictionary()
    )

    def create_production_journal() -> LinuxNvidiaJournalEventProvider:
        trusted = resolve_trusted_journalctl()
        try:
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
                encoding="ascii"
            )
        except OSError:
            raise ProbeFailure("JOURNAL_PROVENANCE_UNAVAILABLE") from None
        provider = object.__new__(LinuxNvidiaJournalEventProvider)
        provider._initialize(
            journalctl_path=trusted.path,
            boot_id=boot_id.strip(),
            command_runner=_default_journal_runner,
            trusted_journalctl=trusted,
            verify_executable=True,
        )
        journals[provider] = (
            trusted,
            provider._journalctl_path,
            provider._boot_id,
            provider._timeout_seconds,
        )
        return provider

    def create_nonproduction_journal(
        *,
        journalctl_path: str,
        boot_id: str,
        command_runner: Callable[[Sequence[str], float], ProbeCommandResult],
        trusted_journalctl: TrustedExecutable,
    ) -> LinuxNvidiaJournalEventProvider:
        provider = object.__new__(LinuxNvidiaJournalEventProvider)
        provider._initialize(
            journalctl_path=journalctl_path,
            boot_id=boot_id,
            command_runner=command_runner,
            trusted_journalctl=trusted_journalctl,
            verify_executable=False,
        )
        return provider

    def journal_is_authentic(value: object) -> bool:
        if type(value) is not LinuxNvidiaJournalEventProvider:
            return False
        registered = journals.get(value)
        if registered is None:
            return False
        trusted, path, boot_id, timeout = registered
        return bool(
            value._trusted_journalctl is trusted
            and value._journalctl_path == path
            and value._boot_id == boot_id
            and value._timeout_seconds == timeout
            and value._command_runner is _default_journal_runner
            and value._verify_executable is True
        )

    def create_production_disk(path: Path) -> StatvfsDiskGrowthProvider:
        provider = object.__new__(StatvfsDiskGrowthProvider)
        provider._initialize(path, statvfs=os.statvfs)
        disks[provider] = (provider._path, provider._baseline)
        return provider

    def disk_is_authentic(value: object) -> bool:
        if type(value) is not StatvfsDiskGrowthProvider:
            return False
        registered = disks.get(value)
        return bool(
            registered is not None
            and value._path == registered[0]
            and value._baseline == registered[1]
            and value._statvfs is os.statvfs
        )

    return (
        create_production_journal,
        create_nonproduction_journal,
        journal_is_authentic,
        create_production_disk,
        disk_is_authentic,
    )


(
    _create_production_journal_provider,
    _create_nonproduction_journal_provider,
    _production_journal_provider_is_authentic,
    _create_production_disk_growth_provider,
    _production_disk_growth_provider_is_authentic,
) = _install_production_provider_factories()
del _install_production_provider_factories


_NVIDIA_GPU_QUERY_FIELDS = (
    "uuid",
    "memory.used",
    "memory.free",
    "memory.total",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.current.graphics",
    "clocks.current.memory",
    "pstate",
    "clocks_event_reasons.active",
)
_THROTTLE_REASON_BITS = (
    (0x1, "GPU_IDLE"),
    (0x2, "APPLICATION_CLOCKS_SETTING"),
    (0x4, "SW_POWER_CAP"),
    (0x8, "HW_SLOWDOWN"),
    (0x10, "SYNC_BOOST"),
    (0x20, "SW_THERMAL_SLOWDOWN"),
    (0x40, "HW_THERMAL_SLOWDOWN"),
    (0x80, "HW_POWER_BRAKE_SLOWDOWN"),
    (0x100, "DISPLAY_CLOCK_SETTING"),
)


def _default_command_runner(
    command: Sequence[str], timeout_seconds: float
) -> ProbeCommandResult:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ProbeFailure("NVIDIA_SMI_EXECUTION_FAILED") from None
    return ProbeCommandResult(completed.returncode, completed.stdout)


_NVIDIA_THERMAL_QUERY_FIELDS = (
    "temperature.gpu.max",
    "temperature.gpu.slowdown",
    "temperature.gpu.shutdown",
    "temperature.gpu.target",
)


class NvidiaSmiThermalLimitProvider:
    """Trusted, identity-pinned NVIDIA thermal-limit provider."""

    def __init__(
        self,
        trusted_executable: TrustedExecutable,
        *,
        gpu_index: int,
        command_runner: Callable[
            [Sequence[str], float], ProbeCommandResult
        ] = _default_command_runner,
    ) -> None:
        if type(trusted_executable) is not TrustedExecutable:
            raise TypeError("Thermal-limit provider requires a trusted executable.")
        if (
            isinstance(gpu_index, bool)
            or not isinstance(gpu_index, int)
            or gpu_index < 0
        ):
            raise ValueError("Thermal-limit GPU index is invalid.")
        self._trusted = trusted_executable
        self._gpu_index = gpu_index
        self._command_runner = command_runner

    def __call__(self) -> Mapping[str, float | None]:
        command = (
            self._trusted.verify(),
            f"--id={self._gpu_index}",
            "--query-gpu=" + ",".join(_NVIDIA_THERMAL_QUERY_FIELDS),
            "--format=csv,noheader,nounits",
        )
        try:
            result = self._command_runner(command, 0.25)
        except Exception:
            raise ProbeFailure("GPU_THERMAL_LIMIT_PROBE_FAILED") from None
        if (
            type(result) is not ProbeCommandResult
            or result.returncode != 0
            or not isinstance(result.stdout, str)
            or len(result.stdout.encode("utf-8")) > 4096
        ):
            raise ProbeFailure("GPU_THERMAL_LIMIT_PROBE_FAILED")
        rows = list(csv.reader(result.stdout.splitlines()))
        if len(rows) != 1 or len(rows[0]) != len(_NVIDIA_THERMAL_QUERY_FIELDS):
            raise ProbeFailure("GPU_THERMAL_LIMIT_INVALID")
        values = [
            _metric(
                item.strip(),
                allowed_units=frozenset({"", "C"}),
                failure_code="GPU_THERMAL_LIMIT_INVALID",
                unit_optional=True,
            )
            for item in rows[0]
        ]
        return {
            "maximum_operating_temperature_c": values[0],
            "slowdown_temperature_c": values[1],
            "shutdown_temperature_c": values[2],
            "target_temperature_c": values[3],
        }


@dataclass(frozen=True)
class NvidiaThermalLimitAuthority:
    status: str
    support_binding: str
    limits: Mapping[str, float | None] | None
    provider: NvidiaSmiThermalLimitProvider | None


def detect_nvidia_thermal_limit_authority(
    trusted_executable: TrustedExecutable,
    *,
    gpu_index: int,
    command_runner: Callable[
        [Sequence[str], float], ProbeCommandResult
    ] = _default_command_runner,
) -> NvidiaThermalLimitAuthority:
    """Detect supported limits or seal trusted help-query proof of absence."""

    try:
        result = command_runner((trusted_executable.verify(), "--help-query-gpu"), 0.5)
    except Exception:
        raise ProbeFailure("GPU_THERMAL_LIMIT_DISCOVERY_FAILED") from None
    if (
        type(result) is not ProbeCommandResult
        or result.returncode != 0
        or not isinstance(result.stdout, str)
    ):
        raise ProbeFailure("GPU_THERMAL_LIMIT_DISCOVERY_FAILED")
    help_bytes = result.stdout.encode("utf-8")
    if not help_bytes or len(help_bytes) > 256 * 1024:
        raise ProbeFailure("GPU_THERMAL_LIMIT_DISCOVERY_FAILED")
    present = {
        field for field in _NVIDIA_THERMAL_QUERY_FIELDS if field in result.stdout
    }
    help_digest = hashlib.sha256(help_bytes).hexdigest()
    if not present:
        return NvidiaThermalLimitAuthority(
            status="unsupported",
            support_binding=f"unsupported:trusted-nvidia-help-query:{help_digest}",
            limits=None,
            provider=None,
        )
    if present != set(_NVIDIA_THERMAL_QUERY_FIELDS):
        raise ProbeFailure("GPU_THERMAL_LIMIT_DISCOVERY_PARTIAL")
    provider = NvidiaSmiThermalLimitProvider(
        trusted_executable,
        gpu_index=gpu_index,
        command_runner=command_runner,
    )
    limits = dict(provider())
    binding = hashlib.sha256(
        json.dumps(
            {
                "fields": list(_NVIDIA_THERMAL_QUERY_FIELDS),
                "limits": limits,
                "nvidia_smi_binding_sha256": trusted_executable.binding_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return NvidiaThermalLimitAuthority(
        status="supported",
        support_binding=f"supported:trusted-nvidia-query:{binding}",
        limits=limits,
        provider=provider,
    )


def _quantity(
    raw: str,
    *,
    allowed_units: frozenset[str],
    failure_code: str,
    unit_optional: bool = False,
) -> tuple[str, str]:
    if not isinstance(raw, str) or len(raw) > 64:
        raise ProbeFailure(failure_code)
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?|\.\d+)\s*([A-Za-z%]+)?\s*",
        raw,
    )
    if match is None:
        raise ProbeFailure(failure_code)
    number, unit_value = match.groups()
    unit = unit_value or ""
    if unit not in allowed_units or (not unit and not unit_optional):
        raise ProbeFailure(failure_code)
    try:
        parsed = Decimal(number)
    except InvalidOperation:
        raise ProbeFailure(failure_code) from None
    if not parsed.is_finite() or parsed < 0:
        raise ProbeFailure(failure_code)
    return number, unit


def _metric(
    raw: str,
    *,
    allowed_units: frozenset[str],
    failure_code: str,
    unit_optional: bool = False,
) -> float:
    number, _unit = _quantity(
        raw,
        allowed_units=allowed_units,
        failure_code=failure_code,
        unit_optional=unit_optional,
    )
    value = float(Decimal(number))
    if not math.isfinite(value):
        raise ProbeFailure(failure_code)
    return value


@dataclass(frozen=True)
class ManagedProcessGroup:
    """Identity-bound ownership of one live Linux process group."""

    process_group_id: int
    leader_pid: int
    leader_identity: str

    def __post_init__(self) -> None:
        for name, value in (
            ("process_group_id", self.process_group_id),
            ("leader_pid", self.leader_pid),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if re.fullmatch(r"linux-start-ticks:\d+", self.leader_identity) is None:
            raise ValueError(
                "leader_identity must be an exact Linux start-time identity"
            )


class LinuxNvidiaHostProbe:
    """Bounded Linux/NVIDIA raw probe with injectable OS boundaries.

    The harness supplies exact managed PIDs and the journal-derived hardware
    event channels. Raw subprocess stderr and byte-exact provider exceptions
    are intentionally discarded in favor of stable ``ProbeFailure.code`` values.
    """

    def __init__(
        self,
        *,
        filesystem_path: Path,
        managed_pids: Callable[[], Iterable[int]],
        managed_process_groups: Callable[[], Iterable["ManagedProcessGroup"]]
        | None = None,
        xid_errors: Callable[[], Sequence[int]] | None = None,
        hardware_events: Callable[[], Mapping[str, bool]] | None = None,
        kernel_events: Callable[[], Mapping[str, Any]] | None = None,
        lease_active: Callable[[], bool],
        disk_growth_bytes: Callable[[], int],
        gpu_index: int = 0,
        nvidia_smi_path: str | None = None,
        trusted_nvidia_executable: TrustedExecutable | None = None,
        command_timeout_seconds: float = 0.25,
        command_runner: Callable[
            [Sequence[str], float], ProbeCommandResult
        ] = _default_command_runner,
        proc_root: Path = Path("/proc"),
        statvfs: Callable[[os.PathLike[str] | str], os.statvfs_result] = os.statvfs,
        cpu_temperature: Callable[[], float | None] | None = None,
        nvme_temperature: Callable[[], float | None] | None = None,
        gpu_thermal_limits: Callable[[], Mapping[str, float | None] | None]
        | None = None,
        page_size_bytes: int | None = None,
        clock_ticks_per_second: int | None = None,
    ) -> None:
        if (
            isinstance(gpu_index, bool)
            or not isinstance(gpu_index, int)
            or gpu_index < 0
        ):
            raise ValueError("gpu_index must be a nonnegative integer")
        if (
            not math.isfinite(command_timeout_seconds)
            or command_timeout_seconds <= 0
            or command_timeout_seconds >= SAMPLE_INTERVAL_SECONDS
        ):
            raise ValueError("command timeout must be positive and below one second")
        self.filesystem_path = filesystem_path
        self._managed_pids = managed_pids
        self._managed_process_groups = managed_process_groups
        self._xid_errors = xid_errors
        self._hardware_events = hardware_events
        self._kernel_events = kernel_events
        if kernel_events is None:
            if not callable(xid_errors) or not callable(hardware_events):
                raise TypeError("split kernel event providers must both be callable")
        elif (
            not callable(kernel_events)
            or xid_errors is not None
            or hardware_events is not None
        ):
            raise TypeError(
                "use one combined kernel event provider or both split providers"
            )
        self._lease_active = lease_active
        self._disk_growth_bytes = disk_growth_bytes
        self.gpu_index = gpu_index
        if (
            trusted_nvidia_executable is not None
            and type(trusted_nvidia_executable) is not TrustedExecutable
        ):
            raise TypeError("trusted NVIDIA executable identity is invalid")
        if (
            trusted_nvidia_executable is not None
            and nvidia_smi_path != trusted_nvidia_executable.path
        ):
            raise ValueError("trusted NVIDIA executable path does not match")
        self._nvidia_smi_path = nvidia_smi_path
        self._trusted_nvidia_executable = trusted_nvidia_executable
        self._command_timeout_seconds = command_timeout_seconds
        self._command_runner = command_runner
        self.proc_root = proc_root
        self._statvfs = statvfs
        try:
            self._page_size_bytes = page_size_bytes or int(os.sysconf("SC_PAGE_SIZE"))
            self._clock_ticks_per_second = clock_ticks_per_second or int(
                os.sysconf("SC_CLK_TCK")
            )
        except (OSError, TypeError, ValueError):
            raise ProbeFailure("HOST_CLOCK_FACTS_UNAVAILABLE") from None
        if self._page_size_bytes <= 0 or self._clock_ticks_per_second <= 0:
            raise ValueError("page size and clock ticks must be positive")
        self._optional_sensors = {
            "cpu": cpu_temperature,
            "nvme": nvme_temperature,
        }
        self._optional_sensor_states: dict[str, bool] = {}
        self._sensor_lock = threading.Lock()
        self._gpu_thermal_limits = gpu_thermal_limits
        self._gpu_thermal_limits_supported: bool | None = None
        self._gpu_thermal_limits_fingerprint: tuple[float, ...] | None = None

    def _executable(self) -> str:
        if self._trusted_nvidia_executable is not None:
            return self._trusted_nvidia_executable.verify()
        executable = self._nvidia_smi_path or shutil.which("nvidia-smi")
        if not executable:
            raise ProbeFailure("NVIDIA_SMI_UNAVAILABLE")
        return executable

    def _run_nvidia(self, command: Sequence[str]) -> str:
        if self._trusted_nvidia_executable is not None:
            executable = self._trusted_nvidia_executable.verify()
            if not command or command[0] != executable:
                raise ProbeFailure("NVIDIA_SMI_IDENTITY_CHANGED")
        try:
            result = self._command_runner(command, self._command_timeout_seconds)
        except ProbeFailure:
            raise
        except Exception:
            raise ProbeFailure("NVIDIA_SMI_EXECUTION_FAILED") from None
        if (
            not isinstance(result, ProbeCommandResult)
            or isinstance(result.returncode, bool)
            or not isinstance(result.returncode, int)
            or not isinstance(result.stdout, str)
        ):
            raise ProbeFailure("NVIDIA_SMI_RESULT_INVALID")
        if result.returncode != 0:
            raise ProbeFailure("NVIDIA_SMI_QUERY_FAILED")
        return result.stdout

    def _managed_pid_set(self) -> set[int]:
        try:
            values = set(self._managed_pids())
        except Exception:
            raise ProbeFailure("MANAGED_PID_PROVIDER_FAILED") from None
        if any(
            isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
            for pid in values
        ):
            raise ProbeFailure("MANAGED_PID_SET_INVALID")
        if self._managed_process_groups is None:
            return values
        try:
            groups = tuple(self._managed_process_groups())
        except Exception:
            raise ProbeFailure("MANAGED_PROCESS_GROUP_PROVIDER_FAILED") from None
        if any(type(group) is not ManagedProcessGroup for group in groups):
            raise ProbeFailure("MANAGED_PROCESS_GROUP_SET_INVALID")
        group_ids = [group.process_group_id for group in groups]
        if len(group_ids) != len(set(group_ids)):
            raise ProbeFailure("MANAGED_PROCESS_GROUP_SET_INVALID")
        for group in groups:
            values.update(self._live_process_group_members(group))
        return values

    def _process_group_identity(self, pid: int) -> tuple[int, str] | None:
        try:
            stat_text = (self.proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            raise ProbeFailure("PROC_PROCESS_IDENTITY_UNAVAILABLE") from None
        try:
            after_name = stat_text.rsplit(")", 1)[1].split()
            process_group_id = int(after_name[2])
            start_ticks = int(after_name[19])
        except (IndexError, ValueError):
            raise ProbeFailure("PROC_PROCESS_IDENTITY_INVALID") from None
        if process_group_id <= 0 or start_ticks < 0:
            raise ProbeFailure("PROC_PROCESS_IDENTITY_INVALID")
        return process_group_id, f"linux-start-ticks:{start_ticks}"

    def _require_live_process_group_leader(self, group: "ManagedProcessGroup") -> None:
        identity = self._process_group_identity(group.leader_pid)
        if identity != (group.process_group_id, group.leader_identity):
            raise ProbeFailure("MANAGED_PROCESS_GROUP_IDENTITY_LOST")

    def _live_process_group_members(self, group: "ManagedProcessGroup") -> set[int]:
        """Resolve an exact live group without ever signaling a discovered PID."""

        self._require_live_process_group_leader(group)
        members: set[int] = set()
        try:
            entries = tuple(self.proc_root.iterdir())
        except OSError:
            raise ProbeFailure("PROC_PROCESS_ENUMERATION_FAILED") from None
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid <= 0:
                continue
            identity = self._process_group_identity(pid)
            if identity is not None and identity[0] == group.process_group_id:
                members.add(pid)
        # Close the enumeration race: a dead/reused group leader invalidates
        # the snapshot instead of attributing unrelated recycled PIDs.
        self._require_live_process_group_leader(group)
        if group.leader_pid not in members:
            raise ProbeFailure("MANAGED_PROCESS_GROUP_IDENTITY_LOST")
        return members

    def _gpu_reading(self, executable: str, managed: set[int]) -> dict[str, Any]:
        output = self._run_nvidia(
            (
                executable,
                f"--id={self.gpu_index}",
                "--query-gpu=" + ",".join(_NVIDIA_GPU_QUERY_FIELDS),
                "--format=csv,noheader",
            )
        )
        rows = [row for row in csv.reader(output.splitlines()) if row]
        if len(rows) != 1 or len(rows[0]) != len(_NVIDIA_GPU_QUERY_FIELDS):
            raise ProbeFailure("NVIDIA_GPU_QUERY_INVALID")
        row = [item.strip() for item in rows[0]]
        uuid = row[0]
        if not re.fullmatch(r"GPU-[A-Za-z0-9-]{8,80}", uuid):
            raise ProbeFailure("NVIDIA_GPU_UUID_INVALID")
        memory: list[dict[str, str]] = []
        for raw in row[1:4]:
            number, unit = _quantity(
                raw,
                allowed_units=frozenset(_BYTE_UNITS),
                failure_code="NVIDIA_MEMORY_FIELD_INVALID",
            )
            memory.append({"value": number, "unit": unit})
        used_bytes = exact_bytes(memory[0]["value"], memory[0]["unit"])
        free_bytes = exact_bytes(memory[1]["value"], memory[1]["unit"])
        total_bytes = exact_bytes(memory[2]["value"], memory[2]["unit"])
        if total_bytes - used_bytes != free_bytes:
            raise ProbeFailure("NVIDIA_MEMORY_INTEGRITY_FAILED")
        performance_state = row[10]
        if re.fullmatch(r"P\d{1,2}", performance_state) is None:
            raise ProbeFailure("NVIDIA_PSTATE_INVALID")
        throttle_state = row[11]
        if re.fullmatch(r"0x[0-9A-Fa-f]{1,16}", throttle_state):
            active_mask = int(throttle_state, 16)
            active = active_mask != 0
            throttle_reasons = [
                reason for bit, reason in _THROTTLE_REASON_BITS if active_mask & bit
            ]
            known_mask = sum(bit for bit, _reason in _THROTTLE_REASON_BITS)
            if active_mask & ~known_mask:
                throttle_reasons.append("CLOCK_EVENT_ACTIVE_OTHER")
        elif throttle_state in {"Active", "Not Active"}:
            active = throttle_state == "Active"
            throttle_reasons = ["CLOCK_EVENT_ACTIVE_OTHER"] if active else []
        else:
            raise ProbeFailure("NVIDIA_THROTTLE_FIELD_INVALID")
        if active and not throttle_reasons:
            throttle_reasons.append("CLOCK_EVENT_ACTIVE_OTHER")
        try:
            if self._kernel_events is not None:
                combined = dict(self._kernel_events())
                if set(combined) != _KERNEL_EVENT_FIELDS:
                    raise ProbeFailure("HARDWARE_EVENT_CHANNEL_INVALID")
                xid_values = list(combined.pop("xid_errors"))
                event_values = combined
            else:
                assert self._xid_errors is not None
                assert self._hardware_events is not None
                xid_values = list(self._xid_errors())
                event_values = dict(self._hardware_events())
        except ProbeFailure:
            raise
        except Exception:
            raise ProbeFailure("HARDWARE_EVENT_PROVIDER_FAILED") from None
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in xid_values
        ):
            raise ProbeFailure("XID_CHANNEL_INVALID")
        required_event_fields = {"reset_detected", "device_lost", "hardware_error"}
        if set(event_values) != required_event_fields or any(
            not isinstance(event_values[field_name], bool)
            for field_name in required_event_fields
        ):
            raise ProbeFailure("HARDWARE_EVENT_CHANNEL_INVALID")
        return {
            "uuid": uuid,
            "memory_used": memory[0],
            "memory_free": memory[1],
            "memory_total": memory[2],
            "utilization_percent": _metric(
                row[4],
                allowed_units=frozenset({"%"}),
                failure_code="NVIDIA_UTILIZATION_FIELD_INVALID",
            ),
            "temperature_c": _metric(
                row[5],
                allowed_units=frozenset({"", "C"}),
                failure_code="NVIDIA_TEMPERATURE_FIELD_INVALID",
                unit_optional=True,
            ),
            "power_draw_w": _metric(
                row[6],
                allowed_units=frozenset({"W"}),
                failure_code="NVIDIA_POWER_FIELD_UNSUPPORTED",
            ),
            "power_limit_w": _metric(
                row[7],
                allowed_units=frozenset({"W"}),
                failure_code="NVIDIA_POWER_FIELD_UNSUPPORTED",
            ),
            "graphics_clock_mhz": _metric(
                row[8],
                allowed_units=frozenset({"MHz"}),
                failure_code="NVIDIA_CLOCK_FIELD_UNSUPPORTED",
            ),
            "memory_clock_mhz": _metric(
                row[9],
                allowed_units=frozenset({"MHz"}),
                failure_code="NVIDIA_CLOCK_FIELD_UNSUPPORTED",
            ),
            "performance_state": performance_state,
            "throttle_reasons": throttle_reasons,
            "throttle_state": throttle_state,
            "xid_errors": sorted(set(xid_values)),
            **event_values,
            "compute_processes": self._compute_processes(executable, uuid, managed),
        }

    def _compute_processes(
        self, executable: str, gpu_uuid: str, managed: set[int]
    ) -> list[dict[str, Any]]:
        output = self._run_nvidia(
            (
                executable,
                f"--id={self.gpu_index}",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader",
            )
        )
        result: list[dict[str, Any]] = []
        seen: set[int] = set()
        for row in csv.reader(output.splitlines()):
            if not row:
                continue
            if len(row) != 3:
                raise ProbeFailure("NVIDIA_PROCESS_QUERY_INVALID")
            pid_text, observed_uuid, raw_memory = (item.strip() for item in row)
            try:
                pid = int(pid_text)
            except ValueError:
                raise ProbeFailure("NVIDIA_PROCESS_QUERY_INVALID") from None
            if pid <= 0 or pid in seen or observed_uuid != gpu_uuid:
                raise ProbeFailure("NVIDIA_PROCESS_QUERY_INVALID")
            seen.add(pid)
            number, unit = _quantity(
                raw_memory,
                allowed_units=frozenset(_BYTE_UNITS),
                failure_code="NVIDIA_PROCESS_MEMORY_UNSUPPORTED",
            )
            result.append(
                {
                    "pid": pid,
                    "used_memory": {"value": number, "unit": unit},
                    "managed": pid in managed,
                }
            )
        return sorted(result, key=lambda item: item["pid"])

    def _proc_values(self, path: Path, failure_code: str) -> dict[str, str]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            raise ProbeFailure(failure_code) from None
        values: dict[str, str] = {}
        for line in lines:
            if ":" in line:
                key, raw = line.split(":", 1)
                values[key] = raw.strip()
            else:
                fields = line.split()
                if len(fields) == 2:
                    values[fields[0]] = fields[1]
        return values

    def _meminfo_kib(self, values: Mapping[str, str], key: str) -> int:
        raw = values.get(key)
        if raw is None:
            raise ProbeFailure("PROC_MEMINFO_INVALID")
        match = re.fullmatch(r"(\d+)\s+kB", raw)
        if match is None:
            raise ProbeFailure("PROC_MEMINFO_INVALID")
        return int(match.group(1)) * 1024

    def _managed_process_totals(self, managed: set[int]) -> dict[str, int | float]:
        rss = 0
        cpu_seconds = 0.0
        read_bytes = 0
        write_bytes = 0
        for pid in sorted(managed):
            root = self.proc_root / str(pid)
            try:
                statm = (root / "statm").read_text(encoding="utf-8").split()
                stat_text = (root / "stat").read_text(encoding="utf-8")
                io_values = self._proc_values(root / "io", "PROC_PROCESS_IO_INVALID")
            except FileNotFoundError:
                continue
            except OSError:
                raise ProbeFailure("PROC_PROCESS_CHANNEL_INVALID") from None
            if len(statm) < 2 or not statm[1].isdigit():
                raise ProbeFailure("PROC_PROCESS_MEMORY_INVALID")
            try:
                after_name = stat_text.rsplit(")", 1)[1].split()
            except IndexError:
                raise ProbeFailure("PROC_PROCESS_CPU_INVALID") from None
            if len(after_name) <= 12:
                raise ProbeFailure("PROC_PROCESS_CPU_INVALID")
            try:
                user_ticks = int(after_name[11])
                system_ticks = int(after_name[12])
                process_read = int(io_values["read_bytes"])
                process_write = int(io_values["write_bytes"])
            except (KeyError, ValueError):
                raise ProbeFailure("PROC_PROCESS_CHANNEL_INVALID") from None
            if min(user_ticks, system_ticks, process_read, process_write) < 0:
                raise ProbeFailure("PROC_PROCESS_CHANNEL_INVALID")
            rss += int(statm[1]) * self._page_size_bytes
            cpu_seconds += (user_ticks + system_ticks) / self._clock_ticks_per_second
            read_bytes += process_read
            write_bytes += process_write
        return {
            "managed_process_rss_bytes": rss,
            "managed_process_cpu_seconds": cpu_seconds,
            "managed_process_read_bytes": read_bytes,
            "managed_process_write_bytes": write_bytes,
        }

    def _optional_sensor(self, name: str) -> dict[str, Any]:
        provider = self._optional_sensors[name]
        if provider is None:
            return {
                "status": "unsupported",
                "value": None,
                "reason_code": "NOT_CONFIGURED",
            }
        try:
            value = provider()
        except Exception:
            raise ProbeFailure(f"{name.upper()}_TEMPERATURE_PROBE_FAILED") from None
        supported = value is not None
        with self._sensor_lock:
            previous = self._optional_sensor_states.setdefault(name, supported)
        if previous != supported:
            raise ProbeFailure(f"{name.upper()}_TEMPERATURE_STATUS_CHANGED")
        if not supported:
            return {
                "status": "unsupported",
                "value": None,
                "reason_code": "UNAVAILABLE_AT_FREEZE",
            }
        return {
            "status": "supported",
            "value": _finite_number(value, f"{name} temperature"),
            "reason_code": None,
        }

    def _check_gpu_thermal_limits(self) -> None:
        """Require a frozen supported thermal-limit channel to stay available.

        Exact thresholds belong in the separately sealed host profile.  A
        successful sample attests that the same provider remained readable and
        unchanged.  A cohort using the conservative fallback passes no
        provider and therefore declares this optional channel unavailable.
        """

        provider = self._gpu_thermal_limits
        if provider is None:
            return
        try:
            raw = provider()
        except Exception:
            if self._gpu_thermal_limits_supported:
                raise ProbeFailure("GPU_THERMAL_LIMIT_DISAPPEARED") from None
            raise ProbeFailure("GPU_THERMAL_LIMIT_PROBE_FAILED") from None
        supported = raw is not None
        previous = self._gpu_thermal_limits_supported
        if previous is not None and previous != supported:
            raise ProbeFailure("GPU_THERMAL_LIMIT_DISAPPEARED")
        self._gpu_thermal_limits_supported = supported
        if not supported:
            return
        if not isinstance(raw, Mapping):
            raise ProbeFailure("GPU_THERMAL_LIMIT_INVALID")
        required = {
            "maximum_operating_temperature_c",
            "slowdown_temperature_c",
            "shutdown_temperature_c",
            "target_temperature_c",
        }
        if set(raw) != required:
            raise ProbeFailure("GPU_THERMAL_LIMIT_INVALID")
        maximum = _finite_number(
            raw["maximum_operating_temperature_c"],
            "maximum operating temperature",
        )
        slowdown = _finite_number(raw["slowdown_temperature_c"], "slowdown temperature")
        shutdown = _finite_number(raw["shutdown_temperature_c"], "shutdown temperature")
        target_raw = raw["target_temperature_c"]
        target = (
            _finite_number(target_raw, "target temperature")
            if target_raw is not None
            else -1.0
        )
        if not maximum <= slowdown <= shutdown:
            raise ProbeFailure("GPU_THERMAL_LIMIT_INVALID")
        fingerprint = (maximum, slowdown, shutdown, target)
        if (
            self._gpu_thermal_limits_fingerprint is not None
            and fingerprint != self._gpu_thermal_limits_fingerprint
        ):
            raise ProbeFailure("GPU_THERMAL_LIMIT_DISAPPEARED")
        self._gpu_thermal_limits_fingerprint = fingerprint

    def _host_reading(self, managed: set[int]) -> dict[str, Any]:
        meminfo = self._proc_values(
            self.proc_root / "meminfo", "PROC_MEMINFO_UNAVAILABLE"
        )
        total_swap = self._meminfo_kib(meminfo, "SwapTotal")
        free_swap = self._meminfo_kib(meminfo, "SwapFree")
        if free_swap > total_swap:
            raise ProbeFailure("PROC_MEMINFO_INVALID")
        vmstat = self._proc_values(self.proc_root / "vmstat", "PROC_VMSTAT_UNAVAILABLE")
        try:
            swap_read_pages = int(vmstat["pswpin"])
            swap_write_pages = int(vmstat["pswpout"])
            load_text = (self.proc_root / "loadavg").read_text(encoding="utf-8")
            load_1m = float(load_text.split()[0])
            filesystem = self._statvfs(self.filesystem_path)
            filesystem_free = int(filesystem.f_bavail) * int(filesystem.f_frsize)
            growth = self._disk_growth_bytes()
            lease_active = self._lease_active()
        except (KeyError, OSError, TypeError, ValueError, IndexError):
            raise ProbeFailure("HOST_CHANNEL_INVALID") from None
        except Exception:
            raise ProbeFailure("HOST_PROVIDER_FAILED") from None
        if (
            swap_read_pages < 0
            or swap_write_pages < 0
            or not math.isfinite(load_1m)
            or load_1m < 0
            or filesystem_free < 0
            or isinstance(growth, bool)
            or not isinstance(growth, int)
            or growth < 0
            or not isinstance(lease_active, bool)
        ):
            raise ProbeFailure("HOST_CHANNEL_INVALID")
        return {
            "mem_available_bytes": self._meminfo_kib(meminfo, "MemAvailable"),
            "swap_used_bytes": total_swap - free_swap,
            "swap_read_bytes": swap_read_pages * self._page_size_bytes,
            "swap_write_bytes": swap_write_pages * self._page_size_bytes,
            "load_1m": load_1m,
            "filesystem_free_bytes": filesystem_free,
            **self._managed_process_totals(managed),
            "disk_growth_bytes": growth,
            "aptus_lease_active": lease_active,
            "cpu_temperature": self._optional_sensor("cpu"),
            "nvme_temperature": self._optional_sensor("nvme"),
        }

    def __call__(self) -> Mapping[str, Any]:
        try:
            self._check_gpu_thermal_limits()
            executable = self._executable()
            managed = self._managed_pid_set()
            return {
                "gpu": self._gpu_reading(executable, managed),
                "host": self._host_reading(managed),
            }
        except ProbeFailure:
            raise
        except Exception:
            raise ProbeFailure("LINUX_NVIDIA_HOST_PROBE_FAILED") from None


@dataclass(frozen=True)
class SamplingResult:
    start_monotonic_ns: int
    stop_monotonic_ns: int
    expected_sample_count: int
    samples: tuple[dict[str, Any], ...]
    missed_slots: tuple[int, ...]
    collector_healthy: bool
    failure_code: str | None

    @property
    def coverage(self) -> float:
        return len(self.samples) / self.expected_sample_count


class FixedRateSampler:
    """One-Hz monotonic sampler that skips elapsed slots instead of catching up."""

    def __init__(
        self,
        probe: HardwareProbe,
        watchdog_probe: Callable[[], Mapping[str, Any]],
        *,
        probe_timeout_seconds: float = 0.8,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
        wall_time: Callable[[], str] = utc_now,
    ) -> None:
        if (
            not math.isfinite(probe_timeout_seconds)
            or probe_timeout_seconds <= 0
            or probe_timeout_seconds >= SAMPLE_INTERVAL_SECONDS
        ):
            raise ValueError("probe timeout must be positive and below one second")
        self._probe = probe
        self._watchdog_probe = watchdog_probe
        self._probe_timeout_seconds = probe_timeout_seconds
        self._monotonic_ns = monotonic_ns
        self._sleep = sleep
        self._wall_time = wall_time

    def collect(
        self,
        *,
        experiment_run_id: str,
        start_monotonic_ns: int,
        stop_monotonic_ns: int,
    ) -> SamplingResult:
        expected = expected_slot_count(start_monotonic_ns, stop_monotonic_ns)
        samples: list[dict[str, Any]] = []
        missed: list[int] = []
        slot = 0
        failure_code: str | None = None
        while slot < expected:
            scheduled = start_monotonic_ns + slot * NANOSECONDS_PER_SECOND
            now = self._monotonic_ns()
            if now < scheduled:
                self._sleep((scheduled - now) / NANOSECONDS_PER_SECOND)
                now = self._monotonic_ns()
            if now >= scheduled + NANOSECONDS_PER_SECOND:
                current_slot = min(
                    (now - start_monotonic_ns) // NANOSECONDS_PER_SECOND,
                    expected,
                )
                missed.extend(range(slot, current_slot))
                slot = current_slot
                if slot >= expected:
                    break
                scheduled = start_monotonic_ns + slot * NANOSECONDS_PER_SECOND
            probe_started = self._monotonic_ns()
            try:
                reading = _bounded_probe(self._probe, self._probe_timeout_seconds)
                watchdog = _bounded_probe(
                    self._watchdog_probe,
                    min(self._probe_timeout_seconds, 0.2),
                )
                probe_finished = self._monotonic_ns()
                if not isinstance(reading, Mapping):
                    raise ProbeFailure("PROBE_INVALID")
                if not isinstance(watchdog, Mapping):
                    raise ProbeFailure("WATCHDOG_PROBE_INVALID")
                observed = probe_finished
                sample = construct_telemetry_sample(
                    sequence=len(samples),
                    experiment_run_id=experiment_run_id,
                    scheduled_slot=slot,
                    scheduled_monotonic_ns=scheduled,
                    observed_monotonic_ns=observed,
                    wall_time_utc=self._wall_time(),
                    probe_reading=reading,
                    collector={
                        "healthy": True,
                        "status_code": None,
                        "probe_duration_ns": max(0, probe_finished - probe_started),
                    },
                    watchdog=watchdog,
                )
            except ProbeFailure as error:
                failure_code = error.code
                break
            except (TelemetryValidationError, TypeError, ValueError):
                failure_code = "TELEMETRY_SAMPLE_INVALID"
                break
            samples.append(sample)
            slot += 1
            completed_at = self._monotonic_ns()
            while (
                slot < expected
                and start_monotonic_ns + slot * NANOSECONDS_PER_SECOND <= completed_at
            ):
                missed.append(slot)
                slot += 1
        if slot < expected:
            missed.extend(range(slot, expected))
        actual_stop = max(
            stop_monotonic_ns,
            max(
                (sample["observed_monotonic_ns"] for sample in samples),
                default=stop_monotonic_ns,
            ),
        )
        return SamplingResult(
            start_monotonic_ns=start_monotonic_ns,
            stop_monotonic_ns=actual_stop,
            expected_sample_count=expected,
            samples=tuple(samples),
            missed_slots=tuple(sorted(set(missed))),
            collector_healthy=failure_code is None,
            failure_code=failure_code,
        )


def expected_slot_count(start_monotonic_ns: int, stop_monotonic_ns: int) -> int:
    start = _require_nonnegative_integer(start_monotonic_ns, "start_monotonic_ns")
    stop = _require_nonnegative_integer(stop_monotonic_ns, "stop_monotonic_ns")
    if stop < start:
        raise TelemetryValidationError("telemetry stop precedes telemetry start")
    return (stop - start) // NANOSECONDS_PER_SECOND + 1


def _window_samples(
    samples: Sequence[Mapping[str, Any]],
    start_monotonic_ns: int,
    stop_monotonic_ns: int,
) -> list[dict[str, Any]]:
    expected = expected_slot_count(start_monotonic_ns, stop_monotonic_ns)
    validated = [validate_telemetry_sample(sample) for sample in samples]
    run_ids = {sample["experiment_run_id"] for sample in validated}
    if len(run_ids) > 1:
        raise TelemetryValidationError(
            "telemetry window contains multiple experiment_run_id values"
        )
    first_sequence = validated[0]["sequence"] if validated else 0
    previous_slot: int | None = None
    previous_time: int | None = None
    for index, sample in enumerate(validated):
        slot = sample["scheduled_slot"]
        sequence = sample["sequence"]
        scheduled = sample["scheduled_monotonic_ns"]
        observed = sample["observed_monotonic_ns"]
        if slot >= expected:
            raise TelemetryValidationError(
                "telemetry sample is outside the scheduled window"
            )
        if sequence != first_sequence + index:
            raise TelemetryValidationError(
                "telemetry sample sequences are not ordered and contiguous"
            )
        if previous_slot is not None and slot <= previous_slot:
            raise TelemetryValidationError(
                "telemetry scheduled slots are not strictly increasing"
            )
        if scheduled != start_monotonic_ns + slot * NANOSECONDS_PER_SECOND:
            raise TelemetryValidationError(
                "telemetry scheduled time does not match its window slot"
            )
        if observed < start_monotonic_ns or observed > stop_monotonic_ns:
            raise TelemetryValidationError(
                "telemetry observation is outside the window"
            )
        if previous_time is not None and observed < previous_time:
            raise TelemetryValidationError("telemetry observations move backward")
        previous_slot = slot
        previous_time = observed
    return validated


def telemetry_coverage(
    samples: Sequence[Mapping[str, Any]],
    start_monotonic_ns: int,
    stop_monotonic_ns: int,
) -> float:
    expected = expected_slot_count(start_monotonic_ns, stop_monotonic_ns)
    valid = _window_samples(samples, start_monotonic_ns, stop_monotonic_ns)
    return min(1.0, len(valid) / expected)


def maximum_gap_seconds(
    samples: Sequence[Mapping[str, Any]],
    start_monotonic_ns: int,
    stop_monotonic_ns: int,
) -> float:
    valid = _window_samples(samples, start_monotonic_ns, stop_monotonic_ns)
    points = [start_monotonic_ns]
    points.extend(sample["observed_monotonic_ns"] for sample in valid)
    points.append(stop_monotonic_ns)
    points.sort()
    return (
        max((right - left for left, right in zip(points, points[1:])), default=0)
        / NANOSECONDS_PER_SECOND
    )


def type7_quantile(values: Sequence[int | float], probability: float) -> int | float:
    if not values:
        raise TelemetryValidationError("a quantile requires at least one value")
    if not math.isfinite(probability) or probability < 0 or probability > 1:
        raise TelemetryValidationError("quantile probability must be in [0, 1]")
    ordered: list[Decimal] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TelemetryValidationError("quantile value must be a finite number")
        if isinstance(value, float) and not math.isfinite(value):
            raise TelemetryValidationError("quantile value must be a finite number")
        ordered.append(Decimal(str(value)))
    ordered.sort()
    decimal_probability = Decimal(str(probability))
    h = Decimal(len(ordered) - 1) * decimal_probability
    lower = int(h // 1)
    upper = lower if h == lower else lower + 1
    fraction = h - lower
    result = ordered[lower] + fraction * (ordered[upper] - ordered[lower])
    integral = result.to_integral_value()
    return int(integral) if result == integral else float(result)


def summarize_scalar(
    values: Sequence[int | float], *, free_resource: bool = False
) -> dict[str, int | float]:
    if not values:
        raise TelemetryValidationError("a scalar summary requires at least one value")
    normalized: list[int | float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TelemetryValidationError("summary value must be a finite number")
        if isinstance(value, float) and not math.isfinite(value):
            raise TelemetryValidationError("summary value must be a finite number")
        normalized.append(value)
    result: dict[str, int | float] = {
        "sample_count": len(normalized),
        "maximum": max(normalized),
        "median": type7_quantile(normalized, 0.5),
    }
    if free_resource:
        result.update(minimum=min(normalized), p05=type7_quantile(normalized, 0.05))
    else:
        result["p95"] = type7_quantile(normalized, 0.95)
    return result


def estimate_gpu_energy(
    samples: Sequence[Mapping[str, Any]],
    start_monotonic_ns: int,
    stop_monotonic_ns: int,
) -> dict[str, float | int] | None:
    valid = _window_samples(samples, start_monotonic_ns, stop_monotonic_ns)
    coverage = telemetry_coverage(valid, start_monotonic_ns, stop_monotonic_ns)
    gap = maximum_gap_seconds(valid, start_monotonic_ns, stop_monotonic_ns)
    if coverage < MINIMUM_QUALIFYING_COVERAGE or gap > MAXIMUM_QUALIFYING_GAP_SECONDS:
        return None
    joules = 0.0
    integrated_ns = 0
    for left, right in zip(valid, valid[1:]):
        interval_ns = right["observed_monotonic_ns"] - left["observed_monotonic_ns"]
        if interval_ns < 0:
            raise TelemetryValidationError("telemetry observations move backward")
        if interval_ns > 2 * NANOSECONDS_PER_SECOND:
            continue
        left_power = _finite_number(left["gpu"]["power_draw_w"], "gpu.power_draw_w")
        right_power = _finite_number(right["gpu"]["power_draw_w"], "gpu.power_draw_w")
        joules += (
            (left_power + right_power) * 0.5 * interval_ns / NANOSECONDS_PER_SECOND
        )
        integrated_ns += interval_ns
    window_ns = stop_monotonic_ns - start_monotonic_ns
    clipped_coverage = integrated_ns / window_ns if window_ns else 1.0
    return {
        "estimated_gpu_energy_joules": joules,
        "integrated_covered_duration_ns": integrated_ns,
        "scheduled_window_duration_ns": window_ns,
        "scheduled_slot_coverage": coverage,
        "clipped_time_support_coverage": min(1.0, clipped_coverage),
    }


def summarize_telemetry(
    samples: Sequence[Mapping[str, Any]],
    start_monotonic_ns: int,
    stop_monotonic_ns: int,
) -> dict[str, Any]:
    valid = _window_samples(samples, start_monotonic_ns, stop_monotonic_ns)
    result: dict[str, Any] = {
        "sample_count": len(valid),
        "expected_sample_count": expected_slot_count(
            start_monotonic_ns, stop_monotonic_ns
        ),
        "coverage": telemetry_coverage(valid, start_monotonic_ns, stop_monotonic_ns),
        "maximum_gap_seconds": maximum_gap_seconds(
            valid, start_monotonic_ns, stop_monotonic_ns
        ),
        "channels": {},
    }
    channel_specs = (
        (
            "gpu_memory_used_bytes",
            False,
            lambda item: item["gpu"]["memory"]["used"]["bytes"],
        ),
        (
            "gpu_memory_free_bytes",
            True,
            lambda item: item["gpu"]["memory"]["free"]["bytes"],
        ),
        (
            "gpu_memory_total_bytes",
            False,
            lambda item: item["gpu"]["memory"]["total"]["bytes"],
        ),
        ("gpu_temperature_c", False, lambda item: item["gpu"]["temperature_c"]),
        ("gpu_power_draw_w", False, lambda item: item["gpu"]["power_draw_w"]),
        ("gpu_power_limit_w", False, lambda item: item["gpu"]["power_limit_w"]),
        (
            "gpu_graphics_clock_mhz",
            False,
            lambda item: item["gpu"]["graphics_clock_mhz"],
        ),
        (
            "gpu_memory_clock_mhz",
            False,
            lambda item: item["gpu"]["memory_clock_mhz"],
        ),
        (
            "gpu_utilization_percent",
            False,
            lambda item: item["gpu"]["utilization_percent"],
        ),
        (
            "host_mem_available_bytes",
            True,
            lambda item: item["host"]["mem_available_bytes"],
        ),
        ("host_swap_used_bytes", False, lambda item: item["host"]["swap_used_bytes"]),
        ("host_swap_read_bytes", False, lambda item: item["host"]["swap_read_bytes"]),
        (
            "host_swap_write_bytes",
            False,
            lambda item: item["host"]["swap_write_bytes"],
        ),
        ("host_load_1m", False, lambda item: item["host"]["load_1m"]),
        (
            "filesystem_free_bytes",
            True,
            lambda item: item["host"]["filesystem_free_bytes"],
        ),
        (
            "managed_process_rss_bytes",
            False,
            lambda item: item["host"]["managed_process_rss_bytes"],
        ),
        (
            "managed_process_cpu_seconds",
            False,
            lambda item: item["host"]["managed_process_cpu_seconds"],
        ),
        (
            "managed_process_read_bytes",
            False,
            lambda item: item["host"]["managed_process_read_bytes"],
        ),
        (
            "managed_process_write_bytes",
            False,
            lambda item: item["host"]["managed_process_write_bytes"],
        ),
        ("disk_growth_bytes", False, lambda item: item["host"]["disk_growth_bytes"]),
        (
            "collector_probe_duration_ns",
            False,
            lambda item: item["collector"]["probe_duration_ns"],
        ),
    )
    for name, free_resource, accessor in channel_specs:
        summary = summarize_scalar(
            [accessor(sample) for sample in valid], free_resource=free_resource
        )
        summary["coverage"] = result["coverage"]
        summary["maximum_gap_seconds"] = result["maximum_gap_seconds"]
        result["channels"][name] = summary
    for field_name, channel_name in (
        ("cpu_temperature", "cpu_temperature_c"),
        ("nvme_temperature", "nvme_temperature_c"),
    ):
        sensor_states = {
            (
                sample["host"][field_name]["status"],
                sample["host"][field_name]["reason_code"],
            )
            for sample in valid
        }
        if len(sensor_states) != 1:
            raise TelemetryValidationError(
                f"{field_name} support state changed within the telemetry window"
            )
        supported = [
            sample["host"][field_name]
            for sample in valid
            if sample["host"][field_name]["status"] == "supported"
        ]
        if supported:
            optional_summary: dict[str, Any] = summarize_scalar(
                [item["value"] for item in supported]
            )
            optional_summary.update(
                status="supported",
                coverage=len(supported) / result["expected_sample_count"],
                maximum_gap_seconds=result["maximum_gap_seconds"],
            )
        else:
            first = valid[0]["host"][field_name]
            optional_summary = {
                "status": "unsupported",
                "reason_code": first["reason_code"],
                "sample_count": 0,
                "coverage": 0.0,
            }
        result["channels"][channel_name] = optional_summary
    result["estimated_gpu_energy"] = estimate_gpu_energy(
        valid, start_monotonic_ns, stop_monotonic_ns
    )
    return result


@dataclass(frozen=True)
class SafetyLimits:
    emergency_deadline_seconds: float
    remaining_disk_budget_bytes: int
    thermal_warning_c: float = 78.0
    thermal_warning_seconds: int = 30
    thermal_stop_c: float = 84.0
    thermal_stop_seconds: int = 5
    thermal_immediate_c: float = 89.0
    vram_warning_bytes: int = int(2.5 * GIB)
    vram_warning_seconds: int = 10
    vram_stop_bytes: int = 2 * GIB
    vram_stop_seconds: int = 5
    ram_warning_bytes: int = 12 * GIB
    ram_warning_seconds: int = 30
    ram_stop_bytes: int = 8 * GIB
    ram_stop_seconds: int = 5
    disk_warning_bytes: int = 48 * GIB
    disk_warning_seconds: int = 30
    disk_stop_bytes: int = 32 * GIB
    swap_warning_bytes_per_second: int = 16 * MIB
    swap_warning_seconds: int = 10
    swap_stop_bytes_per_second: int = 64 * MIB
    swap_stop_seconds: int = 10
    swap_secondary_stop_bytes_per_second: int = 16 * MIB
    swap_secondary_stop_seconds: int = 60
    qualifying_gap_seconds: float = MAXIMUM_QUALIFYING_GAP_SECONDS
    gap_warning_seconds: float = 3.0
    gap_stop_seconds: float = 5.0
    heartbeat_warning_seconds: float = 3.0
    heartbeat_stop_seconds: float = 5.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.emergency_deadline_seconds, bool)
            or not math.isfinite(self.emergency_deadline_seconds)
            or self.emergency_deadline_seconds <= 0
        ):
            raise ValueError("emergency deadline must be positive and finite")
        if (
            isinstance(self.remaining_disk_budget_bytes, bool)
            or not isinstance(self.remaining_disk_budget_bytes, int)
            or self.remaining_disk_budget_bytes < 0
        ):
            raise ValueError("remaining disk budget must be nonnegative")
        if not (
            self.thermal_warning_c < self.thermal_stop_c < self.thermal_immediate_c
            and self.vram_stop_bytes < self.vram_warning_bytes
            and self.ram_stop_bytes < self.ram_warning_bytes
            and self.disk_stop_bytes < self.disk_warning_bytes
            and self.qualifying_gap_seconds
            < self.gap_warning_seconds
            < self.gap_stop_seconds
            and self.heartbeat_warning_seconds < self.heartbeat_stop_seconds
        ):
            raise ValueError("safety threshold ordering is invalid")
        durations = (
            self.thermal_warning_seconds,
            self.thermal_stop_seconds,
            self.vram_warning_seconds,
            self.vram_stop_seconds,
            self.ram_warning_seconds,
            self.ram_stop_seconds,
            self.disk_warning_seconds,
            self.swap_warning_seconds,
            self.swap_stop_seconds,
            self.swap_secondary_stop_seconds,
        )
        if any(isinstance(value, bool) or value <= 0 for value in durations):
            raise ValueError("sustained safety durations must be positive")

    @classmethod
    def frozen_phase1(
        cls,
        *,
        emergency_deadline_seconds: float,
        remaining_disk_budget_bytes: int,
        initial_thermal_limits_available: bool = True,
    ) -> SafetyLimits:
        if initial_thermal_limits_available:
            return cls(
                emergency_deadline_seconds=emergency_deadline_seconds,
                remaining_disk_budget_bytes=remaining_disk_budget_bytes,
            )
        return cls(
            emergency_deadline_seconds=emergency_deadline_seconds,
            remaining_disk_budget_bytes=remaining_disk_budget_bytes,
            thermal_warning_c=75.0,
            thermal_stop_c=82.0,
            thermal_immediate_c=85.0,
        )


@dataclass(frozen=True)
class SafetyEvent:
    level: str
    reason_code: str
    monotonic_ns: int


class SafetyStateMachine:
    """Deterministic sustained-limit evaluator for one active experiment run."""

    def __init__(self, limits: SafetyLimits, *, run_started_monotonic_ns: int) -> None:
        self.limits = limits
        self.run_started_monotonic_ns = _require_nonnegative_integer(
            run_started_monotonic_ns, "run_started_monotonic_ns"
        )
        self._condition_started: dict[str, int] = {}
        self._warning_active: set[str] = set()
        self._last_sample_ns: int | None = None
        self._last_swap_total: int | None = None
        self._last_swap_ns: int | None = None
        self._swap_history: list[tuple[int, int]] = []
        self.stop_event: SafetyEvent | None = None

    def _stop(self, reason_code: str, now_ns: int) -> tuple[SafetyEvent, ...]:
        if self.stop_event is None:
            self.stop_event = SafetyEvent("stop", reason_code, now_ns)
            return (self.stop_event,)
        return ()

    def _sustained(
        self,
        key: str,
        condition: bool,
        now_ns: int,
        seconds: float,
        *,
        start_hint_ns: int | None = None,
    ) -> bool:
        if not condition:
            self._condition_started.pop(key, None)
            return False
        if key not in self._condition_started:
            self._condition_started[key] = (
                now_ns if start_hint_ns is None else start_hint_ns
            )
        return now_ns - self._condition_started[key] >= seconds * NANOSECONDS_PER_SECOND

    def _transition_event(
        self,
        key: str,
        level: str,
        reason_code: str,
        active: bool,
        now_ns: int,
    ) -> tuple[SafetyEvent, ...]:
        if not active:
            self._warning_active.discard(key)
            return ()
        if key in self._warning_active:
            return ()
        self._warning_active.add(key)
        return (SafetyEvent(level, reason_code, now_ns),)

    def _warning(
        self, key: str, reason_code: str, active: bool, now_ns: int
    ) -> tuple[SafetyEvent, ...]:
        return self._transition_event(key, "warning", reason_code, active, now_ns)

    def _deadline_exceeded(self, now_ns: int) -> bool:
        return now_ns - self.run_started_monotonic_ns >= (
            self.limits.emergency_deadline_seconds * NANOSECONDS_PER_SECOND
        )

    def observe_health(
        self,
        *,
        now_monotonic_ns: int,
        collector_alive: bool,
        watchdog_heartbeat_monotonic_ns: int,
        watchdog_alive: bool,
        ownership_certain: bool,
    ) -> tuple[SafetyEvent, ...]:
        now = _require_nonnegative_integer(now_monotonic_ns, "now_monotonic_ns")
        heartbeat = _require_nonnegative_integer(
            watchdog_heartbeat_monotonic_ns, "watchdog_heartbeat_monotonic_ns"
        )
        if heartbeat > now:
            raise TelemetryValidationError("watchdog heartbeat cannot be in the future")
        if self.stop_event is not None:
            return ()
        if not collector_alive:
            return self._stop("TELEMETRY_COLLECTOR_FAILURE", now)
        if not ownership_certain:
            return self._stop("OWNERSHIP_UNCERTAIN", now)
        if not watchdog_alive:
            return self._stop("WATCHDOG_HEARTBEAT_LOST", now)
        if self._deadline_exceeded(now):
            return self._stop("EMERGENCY_DEADLINE_EXCEEDED", now)
        heartbeat_age = (now - heartbeat) / NANOSECONDS_PER_SECOND
        if heartbeat_age >= self.limits.heartbeat_stop_seconds:
            return self._stop("WATCHDOG_HEARTBEAT_LOST", now)
        if self._last_sample_ns is not None:
            gap = (now - self._last_sample_ns) / NANOSECONDS_PER_SECOND
            if gap > self.limits.gap_stop_seconds:
                return self._stop("TELEMETRY_HARD_GAP", now)
        events: list[SafetyEvent] = []
        events.extend(
            self._warning(
                "heartbeat",
                "WATCHDOG_HEARTBEAT_WARNING",
                heartbeat_age >= self.limits.heartbeat_warning_seconds,
                now,
            )
        )
        if self._last_sample_ns is not None:
            gap = (now - self._last_sample_ns) / NANOSECONDS_PER_SECOND
            events.extend(
                self._transition_event(
                    "telemetry_qualifying_gap",
                    "capture-invalid",
                    "TELEMETRY_QUALIFYING_GAP",
                    gap > self.limits.qualifying_gap_seconds,
                    now,
                )
            )
            events.extend(
                self._warning(
                    "telemetry_gap_warning",
                    "TELEMETRY_QUALIFYING_GAP",
                    gap > self.limits.gap_warning_seconds,
                    now,
                )
            )
        return tuple(events)

    def observe_sample(
        self,
        sample: Mapping[str, Any],
        *,
        immediate_events: Iterable[str] = (),
    ) -> tuple[SafetyEvent, ...]:
        value = validate_telemetry_sample(sample)
        now = value["observed_monotonic_ns"]
        if self.stop_event is not None:
            return ()
        if self._last_sample_ns is not None and now < self._last_sample_ns:
            raise TelemetryValidationError("safety observations move backward")
        requested = set(immediate_events)
        unknown = requested - IMMEDIATE_STOP_REASON_CODES
        if unknown:
            raise TelemetryValidationError("an immediate safety reason code is unknown")
        gpu = value["gpu"]
        host = value["host"]
        if gpu["xid_errors"]:
            requested.add("CUDA_XID")
        if gpu["reset_detected"]:
            requested.add("CUDA_DEVICE_RESET")
        if gpu["device_lost"]:
            requested.add("CUDA_DEVICE_LOST")
        if gpu["hardware_error"]:
            requested.add("HARDWARE_ERROR")
        if any(reason in _THERMAL_STOP_REASONS for reason in gpu["throttle_reasons"]):
            requested.add("THERMAL_THROTTLE")
        if any(not process["managed"] for process in gpu["compute_processes"]):
            requested.add("UNRELATED_GPU_ACTIVITY")
        if not value["collector"]["healthy"]:
            requested.add("TELEMETRY_COLLECTOR_FAILURE")
        if not value["watchdog"]["healthy"]:
            requested.add("WATCHDOG_HEARTBEAT_LOST")
        if not value["watchdog"]["ownership_certain"]:
            requested.add("OWNERSHIP_UNCERTAIN")
        if self._deadline_exceeded(now):
            requested.add("EMERGENCY_DEADLINE_EXCEEDED")

        gap_seconds = 0.0
        if self._last_sample_ns is not None:
            gap_seconds = (now - self._last_sample_ns) / NANOSECONDS_PER_SECOND
        heartbeat_age = (
            now - value["watchdog"]["heartbeat_monotonic_ns"]
        ) / NANOSECONDS_PER_SECOND
        if heartbeat_age >= self.limits.heartbeat_stop_seconds:
            requested.add("WATCHDOG_HEARTBEAT_LOST")
        for reason_code in _IMMEDIATE_PRIORITY:
            if reason_code in requested:
                return self._stop(reason_code, now)
        if gap_seconds > self.limits.gap_stop_seconds:
            return self._stop("TELEMETRY_HARD_GAP", now)

        temperature = gpu["temperature_c"]
        free_vram = gpu["memory"]["free"]["bytes"]
        available_ram = host["mem_available_bytes"]
        free_disk = host["filesystem_free_bytes"]
        if temperature >= self.limits.thermal_immediate_c:
            return self._stop("THERMAL_STOP_IMMEDIATE", now)
        if free_disk < self.limits.disk_stop_bytes:
            return self._stop("DISK_FLOOR", now)
        if (
            free_disk
            < self.limits.disk_stop_bytes + self.limits.remaining_disk_budget_bytes
        ):
            return self._stop("DISK_BUDGET_INSUFFICIENT", now)

        thermal_stop = self._sustained(
            "thermal_stop",
            temperature >= self.limits.thermal_stop_c,
            now,
            self.limits.thermal_stop_seconds,
        )
        vram_stop = self._sustained(
            "vram_stop",
            free_vram < self.limits.vram_stop_bytes,
            now,
            self.limits.vram_stop_seconds,
        )
        ram_stop = self._sustained(
            "ram_stop",
            available_ram < self.limits.ram_stop_bytes,
            now,
            self.limits.ram_stop_seconds,
        )

        swap_total = host["swap_read_bytes"] + host["swap_write_bytes"]
        instantaneous_swap_rate: float | None = None
        interval_start: int | None = None
        if self._last_swap_total is not None and self._last_swap_ns is not None:
            elapsed_ns = now - self._last_swap_ns
            delta = swap_total - self._last_swap_total
            if elapsed_ns <= 0 or delta < 0:
                return self._stop("TELEMETRY_COLLECTOR_FAILURE", now)
            instantaneous_swap_rate = delta * NANOSECONDS_PER_SECOND / elapsed_ns
            interval_start = self._last_swap_ns
        self._last_swap_total = swap_total
        self._last_swap_ns = now
        self._swap_history.append((now, swap_total))
        history_floor = now - self.limits.swap_secondary_stop_seconds * (
            NANOSECONDS_PER_SECOND
        )
        while len(self._swap_history) > 1 and self._swap_history[1][0] <= history_floor:
            self._swap_history.pop(0)

        rolling_swap_rate: float | None = None
        rolling_window_ns = self.limits.swap_stop_seconds * NANOSECONDS_PER_SECOND
        target_ns = now - rolling_window_ns
        candidates = [item for item in self._swap_history if item[0] <= target_ns]
        if candidates:
            window_started_ns, window_started_total = candidates[-1]
            elapsed_ns = now - window_started_ns
            delta = swap_total - window_started_total
            if elapsed_ns < rolling_window_ns or delta < 0:
                return self._stop("TELEMETRY_COLLECTOR_FAILURE", now)
            rolling_swap_rate = delta * NANOSECONDS_PER_SECOND / elapsed_ns

        swap_stop = bool(
            rolling_swap_rate is not None
            and rolling_swap_rate >= self.limits.swap_stop_bytes_per_second
        )
        swap_secondary_stop = False
        if instantaneous_swap_rate is not None:
            swap_secondary_stop = self._sustained(
                "swap_secondary_stop",
                instantaneous_swap_rate
                >= self.limits.swap_secondary_stop_bytes_per_second,
                now,
                self.limits.swap_secondary_stop_seconds,
                start_hint_ns=interval_start,
            )
        if thermal_stop:
            return self._stop("THERMAL_STOP_SUSTAINED", now)
        if vram_stop:
            return self._stop("FREE_VRAM_FLOOR", now)
        if ram_stop:
            return self._stop("HOST_RAM_FLOOR", now)
        if swap_stop or swap_secondary_stop:
            return self._stop("SWAP_RATE_LIMIT", now)

        events: list[SafetyEvent] = []
        events.extend(
            self._transition_event(
                "qualifying_gap",
                "capture-invalid",
                "TELEMETRY_QUALIFYING_GAP",
                gap_seconds > self.limits.qualifying_gap_seconds,
                now,
            )
        )
        events.extend(
            self._warning(
                "gap_warning",
                "TELEMETRY_QUALIFYING_GAP",
                gap_seconds > self.limits.gap_warning_seconds,
                now,
            )
        )
        events.extend(
            self._warning(
                "heartbeat",
                "WATCHDOG_HEARTBEAT_WARNING",
                heartbeat_age >= self.limits.heartbeat_warning_seconds,
                now,
            )
        )
        thermal_warning = self._sustained(
            "thermal_warning",
            temperature >= self.limits.thermal_warning_c,
            now,
            self.limits.thermal_warning_seconds,
        )
        vram_warning = self._sustained(
            "vram_warning",
            free_vram < self.limits.vram_warning_bytes,
            now,
            self.limits.vram_warning_seconds,
        )
        ram_warning = self._sustained(
            "ram_warning",
            available_ram < self.limits.ram_warning_bytes,
            now,
            self.limits.ram_warning_seconds,
        )
        disk_warning = self._sustained(
            "disk_warning",
            free_disk < self.limits.disk_warning_bytes,
            now,
            self.limits.disk_warning_seconds,
        )
        swap_warning = False
        if rolling_swap_rate is not None:
            swap_warning = (
                rolling_swap_rate >= self.limits.swap_warning_bytes_per_second
            )
        for key, code, active in (
            ("thermal_warning_event", "THERMAL_WARNING_SUSTAINED", thermal_warning),
            ("vram_warning_event", "FREE_VRAM_WARNING", vram_warning),
            ("ram_warning_event", "HOST_RAM_WARNING", ram_warning),
            ("disk_warning_event", "DISK_WARNING", disk_warning),
            ("swap_warning_event", "SWAP_RATE_WARNING", swap_warning),
        ):
            events.extend(self._warning(key, code, active, now))
        self._last_sample_ns = now
        return tuple(events)


@dataclass(frozen=True)
class WindowValidation:
    valid: bool
    reason_codes: tuple[str, ...]
    summary: Mapping[str, Any] = field(default_factory=dict)


def _contiguous_window(
    samples: Sequence[Mapping[str, Any]], required_samples: int
) -> tuple[list[dict[str, Any]], list[str]]:
    if required_samples <= 0:
        raise ValueError("required_samples must be positive")
    reasons: list[str] = []
    if len(samples) != required_samples:
        return [], ["MISSING_REQUIRED_EVIDENCE"]
    try:
        valid = [validate_telemetry_sample(sample) for sample in samples]
    except TelemetryValidationError:
        return [], ["MISSING_REQUIRED_EVIDENCE"]
    experiment_run_id = valid[0]["experiment_run_id"]
    for left, right in zip(valid, valid[1:]):
        if (
            right["sequence"] != left["sequence"] + 1
            or right["experiment_run_id"] != experiment_run_id
        ):
            reasons.append("MISSING_REQUIRED_EVIDENCE")
            break
        if right["scheduled_slot"] != left["scheduled_slot"] + 1:
            reasons.append("TELEMETRY_QUALIFYING_GAP")
            break
        if (
            right["scheduled_monotonic_ns"] - left["scheduled_monotonic_ns"]
            != NANOSECONDS_PER_SECOND
        ):
            reasons.append("TELEMETRY_QUALIFYING_GAP")
            break
        observed_gap = (
            right["observed_monotonic_ns"] - left["observed_monotonic_ns"]
        ) / NANOSECONDS_PER_SECOND
        if observed_gap <= 0 or observed_gap > 2.5:
            reasons.append("TELEMETRY_QUALIFYING_GAP")
            break
    return valid, reasons


def _health_reasons(samples: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for sample in samples:
        gpu = sample["gpu"]
        host = sample["host"]
        if not sample["collector"]["healthy"]:
            reasons.append("TELEMETRY_COLLECTOR_FAILURE")
        if not sample["watchdog"]["healthy"]:
            reasons.append("WATCHDOG_HEARTBEAT_LOST")
        if not sample["watchdog"]["ownership_certain"]:
            reasons.append("OWNERSHIP_UNCERTAIN")
        heartbeat_age = (
            sample["observed_monotonic_ns"]
            - sample["watchdog"]["heartbeat_monotonic_ns"]
        ) / NANOSECONDS_PER_SECOND
        if heartbeat_age >= 5:
            reasons.append("WATCHDOG_HEARTBEAT_LOST")
        elif heartbeat_age >= 3:
            reasons.append("WATCHDOG_HEARTBEAT_WARNING")
        if any(reason in _THERMAL_STOP_REASONS for reason in gpu["throttle_reasons"]):
            reasons.append("THERMAL_THROTTLE")
        if gpu["xid_errors"]:
            reasons.append("CUDA_XID")
        if gpu["reset_detected"]:
            reasons.append("CUDA_DEVICE_RESET")
        if gpu["device_lost"]:
            reasons.append("CUDA_DEVICE_LOST")
        if gpu["hardware_error"]:
            reasons.append("HARDWARE_ERROR")
        if any(not process["managed"] for process in gpu["compute_processes"]):
            reasons.append("UNRELATED_GPU_ACTIVITY")
        if host["aptus_lease_active"]:
            reasons.append("OWNERSHIP_UNCERTAIN")
    return reasons


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def validate_idle_baseline(
    samples: Sequence[Mapping[str, Any]], *, required_samples: int = 600
) -> WindowValidation:
    valid, reasons = _contiguous_window(samples, required_samples)
    if not valid:
        return WindowValidation(False, _unique(reasons))
    reasons.extend(_health_reasons(valid))
    temperatures = [sample["gpu"]["temperature_c"] for sample in valid]
    free_vram = [sample["gpu"]["memory"]["free"]["bytes"] for sample in valid]
    powers = [sample["gpu"]["power_draw_w"] for sample in valid]
    summary = {
        "sample_count": len(valid),
        "gpu_temperature_median_c": type7_quantile(temperatures, 0.5),
        "gpu_temperature_p95_c": type7_quantile(temperatures, 0.95),
        "gpu_free_vram_median_bytes": type7_quantile(free_vram, 0.5),
        "gpu_power_draw_p95_w": type7_quantile(powers, 0.95),
        "gpu_utilization": summarize_scalar(
            [sample["gpu"]["utilization_percent"] for sample in valid]
        ),
    }
    return WindowValidation(not reasons, _unique(reasons), summary)


def _temperature_slope_per_minute(samples: Sequence[Mapping[str, Any]]) -> float:
    origin = samples[0]["observed_monotonic_ns"]
    xs = [
        (sample["observed_monotonic_ns"] - origin) / NANOSECONDS_PER_SECOND
        for sample in samples
    ]
    ys = [float(sample["gpu"]["temperature_c"]) for sample in samples]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator == 0:
        raise TelemetryValidationError("cooldown timestamps have zero variance")
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return numerator / denominator * 60


def validate_cooldown(
    samples: Sequence[Mapping[str, Any]],
    idle_baseline: Mapping[str, Any],
    *,
    required_samples: int = 120,
    minimum_zero_utilization_samples: int = 110,
) -> WindowValidation:
    valid, reasons = _contiguous_window(samples, required_samples)
    if not valid:
        return WindowValidation(False, _unique(reasons))
    evaluated = validate_cooldown_observations(
        valid,
        idle_baseline,
        required_samples=required_samples,
        minimum_zero_utilization_samples=minimum_zero_utilization_samples,
    )
    return WindowValidation(
        evaluated.valid and not reasons,
        _unique([*reasons, *evaluated.reason_codes]),
        evaluated.summary,
    )


def validate_cooldown_observations(
    valid: Sequence[Mapping[str, Any]],
    idle_baseline: Mapping[str, Any],
    *,
    required_samples: int = 120,
    minimum_zero_utilization_samples: int = 110,
    required_host_ram_bytes: int = 12 * GIB,
    required_disk_bytes: int = 48 * GIB,
    disk_reason_code: str = "DISK_WARNING",
    power_channel_supported: bool | None = None,
) -> WindowValidation:
    """Evaluate shared cooldown facts independent of their identity envelope."""

    if required_samples <= 0:
        raise ValueError("required_samples must be positive")
    if len(valid) != required_samples:
        return WindowValidation(False, ("MISSING_REQUIRED_EVIDENCE",))
    required_baseline_fields = {
        "gpu_temperature_median_c",
        "gpu_temperature_p95_c",
        "gpu_free_vram_median_bytes",
        "gpu_power_draw_p95_w",
    }
    if not required_baseline_fields.issubset(idle_baseline):
        return WindowValidation(False, ("MISSING_REQUIRED_EVIDENCE",))
    if (
        minimum_zero_utilization_samples < 0
        or minimum_zero_utilization_samples > required_samples
    ):
        raise ValueError(
            "minimum zero-utilization count is outside the cooldown window"
        )
    if (
        isinstance(required_host_ram_bytes, bool)
        or not isinstance(required_host_ram_bytes, int)
        or required_host_ram_bytes < 0
        or isinstance(required_disk_bytes, bool)
        or not isinstance(required_disk_bytes, int)
        or required_disk_bytes < 0
    ):
        raise ValueError("cooldown resource requirements must be nonnegative integers")
    if disk_reason_code not in {"DISK_WARNING", "DISK_BUDGET_INSUFFICIENT"}:
        raise ValueError("cooldown disk reason code is invalid")
    if power_channel_supported is None:
        power_channel_supported = idle_baseline["gpu_power_draw_p95_w"] is not None
    if not isinstance(power_channel_supported, bool):
        raise ValueError("power channel support must be boolean")
    reasons: list[str] = []
    for left, right in zip(valid, valid[1:]):
        if right["sequence"] != left["sequence"] + 1:
            reasons.append("MISSING_REQUIRED_EVIDENCE")
            break
        if right["scheduled_slot"] != left["scheduled_slot"] + 1:
            reasons.append("TELEMETRY_QUALIFYING_GAP")
            break
        if (
            right["scheduled_monotonic_ns"] - left["scheduled_monotonic_ns"]
            != NANOSECONDS_PER_SECOND
        ):
            reasons.append("TELEMETRY_QUALIFYING_GAP")
            break
        observed_gap = (
            right["observed_monotonic_ns"] - left["observed_monotonic_ns"]
        ) / NANOSECONDS_PER_SECOND
        if observed_gap <= 0 or observed_gap > MAXIMUM_QUALIFYING_GAP_SECONDS:
            reasons.append("TELEMETRY_QUALIFYING_GAP")
            break
    reasons.extend(_health_reasons(valid))
    temperatures = [float(sample["gpu"]["temperature_c"]) for sample in valid]
    idle_median = _finite_number(
        idle_baseline["gpu_temperature_median_c"], "idle temperature median"
    )
    idle_p95 = _finite_number(
        idle_baseline["gpu_temperature_p95_c"], "idle temperature p95"
    )
    temperature_ceiling = min(50.0, idle_median + 5.0, idle_p95 + 3.0)
    if any(value > temperature_ceiling for value in temperatures):
        reasons.append("THERMAL_WARNING_SUSTAINED")
    try:
        slope = _temperature_slope_per_minute(valid)
    except TelemetryValidationError:
        reasons.append("MISSING_REQUIRED_EVIDENCE")
        slope = math.nan
    if math.isfinite(slope) and slope >= 0.1:
        reasons.append("THERMAL_WARNING_SUSTAINED")
    zero_utilization = sum(
        sample["gpu"]["utilization_percent"] == 0 for sample in valid
    )
    if zero_utilization < minimum_zero_utilization_samples:
        reasons.append("UNRELATED_GPU_ACTIVITY")
    idle_free = _finite_number(
        idle_baseline["gpu_free_vram_median_bytes"], "idle free VRAM median"
    )
    free_values = [sample["gpu"]["memory"]["free"]["bytes"] for sample in valid]
    if any(
        abs(value - idle_free) > 128 * MIB or value < int(2.5 * GIB)
        for value in free_values
    ):
        reasons.append("FREE_VRAM_WARNING")
    if power_channel_supported:
        idle_power_p95 = _finite_number(
            idle_baseline["gpu_power_draw_p95_w"], "idle power p95", minimum=0
        )
        if any(sample["gpu"]["power_draw_w"] > idle_power_p95 + 10 for sample in valid):
            reasons.append("MISSING_REQUIRED_EVIDENCE")
    host_floor = max(12 * GIB, required_host_ram_bytes)
    disk_floor = max(48 * GIB, required_disk_bytes)
    if any(sample["host"]["mem_available_bytes"] < host_floor for sample in valid):
        reasons.append("HOST_RAM_WARNING")
    if any(sample["host"]["filesystem_free_bytes"] < disk_floor for sample in valid):
        reasons.append(disk_reason_code)
    swap_history: list[tuple[int, int]] = []
    for sample in valid:
        observed = sample["observed_monotonic_ns"]
        total = sample["host"]["swap_read_bytes"] + sample["host"]["swap_write_bytes"]
        if swap_history and total < swap_history[-1][1]:
            reasons.append("MISSING_REQUIRED_EVIDENCE")
            break
        swap_history.append((observed, total))
        target = observed - 10 * NANOSECONDS_PER_SECOND
        candidates = [item for item in swap_history if item[0] <= target]
        if not candidates:
            continue
        started_at, started_total = candidates[-1]
        elapsed = observed - started_at
        if elapsed <= 0:
            continue
        rolling_rate = (total - started_total) * NANOSECONDS_PER_SECOND / elapsed
        if rolling_rate >= 16 * MIB:
            reasons.append("SWAP_RATE_WARNING")
            break
    summary = {
        "sample_count": len(valid),
        "zero_utilization_sample_count": zero_utilization,
        "gpu_temperature_maximum_c": max(temperatures),
        "gpu_temperature_slope_c_per_minute": slope,
        "gpu_free_vram_minimum_bytes": min(free_values),
        "gpu_power_draw_maximum_w": max(
            sample["gpu"]["power_draw_w"] for sample in valid
        ),
        "required_host_ram_bytes": host_floor,
        "required_disk_bytes": disk_floor,
        "power_channel_supported": power_channel_supported,
    }
    return WindowValidation(not reasons, _unique(reasons), summary)
