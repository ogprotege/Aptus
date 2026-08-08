"""Background telemetry and safety adapter for the CUDA capture harness.

The sidecar deliberately owns only process-local threads.  It samples the
injected host probe on monotonic one-second slots, maintains an independent
watchdog heartbeat, and exposes the first fail-closed safety signal to the
capture harness.  It never attempts to cancel or signal a workload itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Self

from .contracts import utc_now
from .harness import SafetySignal, TelemetryCapture as HarnessTelemetryCapture
from .monitoring import (
    MINIMUM_QUALIFYING_COVERAGE,
    NANOSECONDS_PER_SECOND,
    SAMPLE_INTERVAL_SECONDS,
    HardwareProbe,
    LinuxNvidiaHostProbe,
    LinuxNvidiaJournalEventProvider,
    ProbeFailure,
    SafetyEvent,
    SafetyLimits,
    SafetyStateMachine,
    StatvfsDiskGrowthProvider,
    TelemetryValidationError,
    construct_telemetry_sample,
    detect_nvidia_thermal_limit_authority,
    maximum_gap_seconds,
    resolve_trusted_nvidia_smi,
    telemetry_coverage,
    validate_telemetry_sample,
)


_EXPERIMENT_RUN_ID = re.compile(r"^xrun_[0-9a-f]{32}$")
_BINDING_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_WATCHDOG_INTERVAL_SECONDS = 0.25
_MAX_SLEEP_SLICE_SECONDS = 0.1
_CONFIGURATION_FORMAT_VERSION = "aptus.cuda-telemetry-configuration.v1"
_QUALIFYING_PROFILE_ID = "phase1-frozen-qualifying"
_NONQUALIFYING_PROFILE_ID = "custom-nonqualifying-test-only"
_EVENT_LEVEL_ORDER = {"capture-invalid": 0, "warning": 1, "stop": 2}
_REQUIRED_SUPPORT_BINDINGS = frozenset(
    {
        "cpu_temperature",
        "gpu_thermal_limits",
        "hardware_events",
        "nvidia_smi_binary",
        "nvme_temperature",
        "xid_projection",
    }
)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        raise ValueError("telemetry configuration must be canonical JSON") from None
    return text.encode("utf-8")


def _json_record_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = json.loads(_canonical_json_bytes(value))
    if not isinstance(copied, dict):  # pragma: no cover - Mapping guarantees this.
        raise ValueError("telemetry configuration must be a JSON object")
    return copied


def _require_binding_text(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be a nonempty bounded string")
    return value


def _normalize_support_bindings(
    value: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("support_bindings must be a nonempty mapping")
    normalized: dict[str, str] = {}
    for raw_name, raw_binding in value.items():
        if not isinstance(raw_name, str) or _BINDING_NAME.fullmatch(raw_name) is None:
            raise ValueError("support binding names must be safe identifiers")
        binding = _require_binding_text(raw_binding, f"support binding {raw_name}")
        if raw_name in normalized:
            raise ValueError("support binding names must be unique")
        normalized[raw_name] = binding
    if set(normalized) != _REQUIRED_SUPPORT_BINDINGS:
        raise ValueError("support_bindings differ from required provider channels")
    return {name: normalized[name] for name in sorted(normalized)}


def _safety_limits_record(limits: SafetyLimits) -> dict[str, int | float]:
    return {item.name: getattr(limits, item.name) for item in fields(limits)}


def _configuration_record(
    *,
    safety_limits: SafetyLimits,
    readiness_timeout_seconds: float,
    join_timeout_seconds: float,
    qualifying: bool,
    initial_thermal_limits_available: bool | None,
    provider_name: str | None,
    provider_version: str | None,
    support_bindings: Mapping[str, str],
    ownership_binding: str | None,
    disk_growth_binding: str | None,
) -> dict[str, Any]:
    if qualifying:
        thermal_mode = (
            "reported-limits-bound"
            if initial_thermal_limits_available
            else "frozen-conservative-fallback"
        )
        qualification_reason = None
    else:
        thermal_mode = "custom-unbound"
        qualification_reason = "CUSTOM_OR_UNBOUND_TELEMETRY_PROFILE"
    payload: dict[str, Any] = {
        "format_version": _CONFIGURATION_FORMAT_VERSION,
        "lifecycle": {
            "join_timeout_seconds": join_timeout_seconds,
            "readiness_timeout_seconds": readiness_timeout_seconds,
        },
        "profile": {
            "id": _QUALIFYING_PROFILE_ID if qualifying else _NONQUALIFYING_PROFILE_ID,
            "qualifying": qualifying,
            "reason_code": qualification_reason,
        },
        "provenance": {
            "disk_growth_binding": disk_growth_binding,
            "ownership_binding": ownership_binding,
            "provider": {
                "name": provider_name,
                "version": provider_version,
            },
            "support_bindings": dict(support_bindings),
        },
        "safety_limits": _safety_limits_record(safety_limits),
        "sampling": {
            "interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "minimum_qualifying_coverage": MINIMUM_QUALIFYING_COVERAGE,
            "watchdog_interval_seconds": _WATCHDOG_INTERVAL_SECONDS,
        },
        "thermal_policy": {
            "initial_limits_available": initial_thermal_limits_available,
            "mode": thermal_mode,
        },
    }
    payload["configuration_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    return _json_record_copy(payload)


def _safety_event_records(
    events: tuple[SafetyEvent, ...],
) -> tuple[dict[str, Any], ...]:
    ordered = sorted(
        events,
        key=lambda event: (
            event.monotonic_ns,
            _EVENT_LEVEL_ORDER.get(event.level, len(_EVENT_LEVEL_ORDER)),
            event.level,
            event.reason_code,
        ),
    )
    return tuple(
        {
            "level": event.level,
            "monotonic_ns": event.monotonic_ns,
            "reason_code": event.reason_code,
        }
        for event in ordered
    )


@dataclass(frozen=True)
class SidecarTelemetryCapture(HarnessTelemetryCapture):
    """Harness-compatible capture plus the exact sidecar evidence envelope."""

    configuration: Mapping[str, Any] = field(default_factory=dict)
    safety_events: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Detached, non-final view for cooldown and admission evaluation."""

    samples: tuple[Mapping[str, Any], ...]
    configuration: Mapping[str, Any]
    safety_events: tuple[Mapping[str, Any], ...]
    failure_code: str | None


class TelemetrySidecarError(RuntimeError):
    """The sidecar could not establish a valid fail-closed lifecycle."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class BackgroundTelemetrySession:
    """Background implementation of the harness telemetry protocol.

    Direct construction is deliberately nonqualifying and exists for tests and
    diagnostics that need arbitrary limits. ``qualifying_production`` retains a
    frozen configuration builder for evaluator tests, but its injected session
    is also nonqualifying. Only a harness-authorized concrete provider may set
    the runtime qualifying authority.
    """

    def __init__(
        self,
        *,
        probe: HardwareProbe,
        safety_limits: SafetyLimits,
        ownership_certain: Callable[[], bool],
        readiness_timeout_seconds: float = 3.0,
        join_timeout_seconds: float = 2.0,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        wall_time: Callable[[], str] = utc_now,
        sleep: Callable[[float], None] = time.sleep,
        watchdog_tick: Callable[[], None] | None = None,
    ) -> None:
        if not callable(probe):
            raise TypeError("probe must be callable")
        if not isinstance(safety_limits, SafetyLimits):
            raise TypeError("safety_limits must be a SafetyLimits instance")
        if not callable(ownership_certain):
            raise TypeError("ownership_certain must be callable")
        for value, label in (
            (readiness_timeout_seconds, "readiness timeout"),
            (join_timeout_seconds, "join timeout"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{label} must be positive and finite")
        if not callable(monotonic_ns) or not callable(wall_time) or not callable(sleep):
            raise TypeError("clock and sleep boundaries must be callable")
        if watchdog_tick is not None and not callable(watchdog_tick):
            raise TypeError("watchdog_tick must be callable")

        self._probe = probe
        self._safety_limits = safety_limits
        self._ownership_provider = ownership_certain
        self._readiness_timeout_seconds = float(readiness_timeout_seconds)
        self._join_timeout_seconds = float(join_timeout_seconds)
        self._monotonic_ns = monotonic_ns
        self._wall_time = wall_time
        self._sleep = sleep
        self._watchdog_tick = watchdog_tick or (lambda: None)
        self._qualifying_profile = False
        self._qualifying_authority: object | None = None
        self._configuration = _configuration_record(
            safety_limits=safety_limits,
            readiness_timeout_seconds=float(readiness_timeout_seconds),
            join_timeout_seconds=float(join_timeout_seconds),
            qualifying=False,
            initial_thermal_limits_available=None,
            provider_name=None,
            provider_version=None,
            support_bindings={},
            ownership_binding=None,
            disk_growth_binding=None,
        )

        self._state_lock = threading.RLock()
        self._safety_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._startup_event = threading.Event()
        self._collector_started_event = threading.Event()

        self._started = False
        self._stopped = False
        self._sample_ready = False
        self._watchdog_ready = False
        self._experiment_run_id: str | None = None
        self._start_monotonic_ns: int | None = None
        self._requested_stop_ns: int | None = None
        self._heartbeat_monotonic_ns = 0
        self._ownership_certain = True
        self._samples: list[dict[str, Any]] = []
        self._safety_events: list[SafetyEvent] = []
        self._collector_failure_code: str | None = None
        self._watchdog_failure_code: str | None = None
        self._qualification_failure_code: str | None = None
        self._signal: SafetySignal | None = None
        self._safety: SafetyStateMachine | None = None
        self._collector_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None

    @classmethod
    def qualifying_production(
        cls,
        *,
        probe: HardwareProbe,
        ownership_certain: Callable[[], bool],
        emergency_deadline_seconds: float,
        remaining_disk_budget_bytes: int,
        initial_thermal_limits_available: bool,
        provider_name: str,
        provider_version: str,
        support_bindings: Mapping[str, str],
        ownership_binding: str,
        disk_growth_binding: str,
        readiness_timeout_seconds: float = 3.0,
        join_timeout_seconds: float = 2.0,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        wall_time: Callable[[], str] = utc_now,
        sleep: Callable[[float], None] = time.sleep,
        watchdog_tick: Callable[[], None] | None = None,
    ) -> Self:
        """Create a frozen-profile but runtime-nonqualifying injected session.

        The binding strings are protected-record provenance identifiers. They
        must identify the concrete probe/version, supported-channel freeze,
        ownership authority, and disk-growth authority used by this session.
        Their presence does not authorize this dependency-injected object to
        produce protocol-valid evidence.
        """

        if not isinstance(initial_thermal_limits_available, bool):
            raise ValueError("initial_thermal_limits_available must be a boolean")
        normalized_provider_name = _require_binding_text(provider_name, "provider_name")
        normalized_provider_version = _require_binding_text(
            provider_version, "provider_version"
        )
        normalized_support = _normalize_support_bindings(support_bindings)
        normalized_ownership = _require_binding_text(
            ownership_binding, "ownership_binding"
        )
        normalized_disk_growth = _require_binding_text(
            disk_growth_binding, "disk_growth_binding"
        )
        limits = SafetyLimits.frozen_phase1(
            emergency_deadline_seconds=emergency_deadline_seconds,
            remaining_disk_budget_bytes=remaining_disk_budget_bytes,
            initial_thermal_limits_available=initial_thermal_limits_available,
        )
        session = cls(
            probe=probe,
            safety_limits=limits,
            ownership_certain=ownership_certain,
            readiness_timeout_seconds=readiness_timeout_seconds,
            join_timeout_seconds=join_timeout_seconds,
            monotonic_ns=monotonic_ns,
            wall_time=wall_time,
            sleep=sleep,
            watchdog_tick=watchdog_tick,
        )
        session._configuration = _configuration_record(
            safety_limits=limits,
            readiness_timeout_seconds=session._readiness_timeout_seconds,
            join_timeout_seconds=session._join_timeout_seconds,
            qualifying=True,
            initial_thermal_limits_available=initial_thermal_limits_available,
            provider_name=normalized_provider_name,
            provider_version=normalized_provider_version,
            support_bindings=normalized_support,
            ownership_binding=normalized_ownership,
            disk_growth_binding=normalized_disk_growth,
        )
        return session

    @classmethod
    def _qualifying_for_harness(
        cls,
        *,
        harness: object,
        authority: object,
        filesystem_path: Path,
        gpu_index: int,
        nvidia_smi_path: str | None,
        unavailable_optional_sensors: tuple[str, ...],
        readiness_timeout_seconds: float = 3.0,
        join_timeout_seconds: float = 2.0,
    ) -> Self:
        """Construct every qualifying provider from one exact production harness."""

        if cls is not BackgroundTelemetrySession:
            raise TypeError("Qualifying telemetry subclasses are forbidden.")
        return _create_qualifying_session_for_harness(
            harness=harness,
            authority=authority,
            filesystem_path=filesystem_path,
            gpu_index=gpu_index,
            nvidia_smi_path=nvidia_smi_path,
            unavailable_optional_sensors=unavailable_optional_sensors,
            readiness_timeout_seconds=readiness_timeout_seconds,
            join_timeout_seconds=join_timeout_seconds,
        )

    def _authorized_for_harness(self, authority: object) -> bool:
        """Return whether this is the exact real-clock sidecar bound to a harness."""

        return _qualifying_session_is_authentic(self, authority)

    @staticmethod
    def _require_monotonic_ns(value: int, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TelemetrySidecarError(f"{label}_INVALID")
        return value

    def _clock(self) -> int:
        try:
            value = self._monotonic_ns()
        except BaseException:
            raise TelemetrySidecarError("MONOTONIC_CLOCK_FAILED") from None
        return self._require_monotonic_ns(value, "MONOTONIC_CLOCK")

    def _ownership(self) -> bool:
        try:
            value = self._ownership_provider()
        except BaseException:
            return False
        return value if isinstance(value, bool) else False

    def _record_signal(self, reason_code: str, detected_ns: int) -> None:
        try:
            signal = SafetySignal(reason_code, detected_ns)
        except (TypeError, ValueError):
            signal = SafetySignal("TELEMETRY_COLLECTOR_FAILURE", detected_ns)
        with self._state_lock:
            if self._signal is None:
                self._signal = signal

    def _record_events(self, events: tuple[SafetyEvent, ...]) -> None:
        if not events:
            return
        with self._state_lock:
            self._safety_events.extend(events)
        for event in events:
            if event.level == "stop":
                self._record_signal(event.reason_code, event.monotonic_ns)
            elif event.level == "capture-invalid":
                with self._state_lock:
                    if self._qualification_failure_code is None:
                        self._qualification_failure_code = event.reason_code

    def _record_collector_failure(self, code: str, detected_ns: int) -> None:
        with self._state_lock:
            if self._collector_failure_code is None:
                self._collector_failure_code = code
            self._startup_event.set()
        self._record_signal("TELEMETRY_COLLECTOR_FAILURE", detected_ns)

    def _record_watchdog_failure(self, code: str, detected_ns: int) -> None:
        with self._state_lock:
            if self._watchdog_failure_code is None:
                self._watchdog_failure_code = code
            self._startup_event.set()
        self._record_signal("WATCHDOG_HEARTBEAT_LOST", detected_ns)

    def _mark_sample_ready(self) -> None:
        with self._state_lock:
            self._sample_ready = True
            if self._watchdog_ready:
                self._startup_event.set()

    def _mark_watchdog_ready(self) -> None:
        with self._state_lock:
            self._watchdog_ready = True
            if self._sample_ready:
                self._startup_event.set()

    def _interruptible_sleep_until(self, target_ns: int) -> bool:
        while not self._stop_event.is_set():
            now = self._clock()
            if now >= target_ns:
                return True
            remaining = (target_ns - now) / NANOSECONDS_PER_SECOND
            try:
                self._sleep(min(remaining, _MAX_SLEEP_SLICE_SECONDS))
            except BaseException:
                raise TelemetrySidecarError("SLEEP_FAILED") from None
        return False

    def _observe_health(
        self,
        *,
        now_ns: int,
        collector_alive: bool,
        watchdog_alive: bool,
        ownership_certain: bool,
        heartbeat_ns: int,
    ) -> None:
        with self._safety_lock:
            safety = self._safety
            if safety is None:
                return
            events = safety.observe_health(
                now_monotonic_ns=now_ns,
                collector_alive=collector_alive,
                watchdog_heartbeat_monotonic_ns=min(heartbeat_ns, now_ns),
                watchdog_alive=watchdog_alive,
                ownership_certain=ownership_certain,
            )
        self._record_events(events)

    def _watchdog_loop(self) -> None:
        try:
            next_tick_ns = self._start_monotonic_ns
            if next_tick_ns is None:
                raise TelemetrySidecarError("WATCHDOG_START_STATE_INVALID")
            interval_ns = int(_WATCHDOG_INTERVAL_SECONDS * NANOSECONDS_PER_SECOND)
            while not self._stop_event.is_set():
                if not self._interruptible_sleep_until(next_tick_ns):
                    break
                self._watchdog_tick()
                now = self._clock()
                ownership = self._ownership()
                with self._state_lock:
                    self._heartbeat_monotonic_ns = now
                    self._ownership_certain = ownership
                    collector = self._collector_thread
                    collector_alive = not self._collector_started_event.is_set() or (
                        collector is not None and collector.is_alive()
                    )
                self._mark_watchdog_ready()
                self._observe_health(
                    now_ns=now,
                    collector_alive=collector_alive,
                    watchdog_alive=True,
                    ownership_certain=ownership,
                    heartbeat_ns=now,
                )
                next_tick_ns += interval_ns
                if next_tick_ns <= now:
                    next_tick_ns = now + interval_ns
        except BaseException:
            try:
                detected_ns = self._clock()
            except TelemetrySidecarError:
                detected_ns = self._start_monotonic_ns or 0
            self._record_watchdog_failure("WATCHDOG_THREAD_FAILED", detected_ns)

    def _collector_loop(self) -> None:
        self._collector_started_event.set()
        try:
            start_ns = self._start_monotonic_ns
            experiment_run_id = self._experiment_run_id
            if start_ns is None or experiment_run_id is None:
                raise TelemetrySidecarError("COLLECTOR_START_STATE_INVALID")
            interval_ns = SAMPLE_INTERVAL_SECONDS * NANOSECONDS_PER_SECOND
            slot = 0
            sequence = 0
            while not self._stop_event.is_set():
                scheduled_ns = start_ns + slot * interval_ns
                if not self._interruptible_sleep_until(scheduled_ns):
                    break
                now = self._clock()
                if now >= scheduled_ns + interval_ns:
                    slot = (now - start_ns) // interval_ns
                    scheduled_ns = start_ns + slot * interval_ns

                probe_started_ns = self._clock()
                try:
                    reading = self._probe()
                except ProbeFailure as error:
                    detected_ns = self._clock()
                    if error.code == "GPU_THERMAL_LIMIT_DISAPPEARED":
                        self._record_signal("THERMAL_LIMIT_DISAPPEARED", detected_ns)
                    self._record_collector_failure(error.code, detected_ns)
                    return
                except BaseException:
                    self._record_collector_failure("PROBE_FAILED", self._clock())
                    return
                observed_ns = self._clock()
                if self._stop_event.is_set():
                    break
                probe_duration_ns = observed_ns - probe_started_ns
                if probe_duration_ns < 0:
                    raise TelemetrySidecarError("MONOTONIC_CLOCK_REVERSED")

                ownership = self._ownership()
                with self._state_lock:
                    heartbeat_ns = min(self._heartbeat_monotonic_ns, observed_ns)
                    watchdog = self._watchdog_thread
                    watchdog_healthy = (
                        self._watchdog_failure_code is None
                        and watchdog is not None
                        and watchdog.is_alive()
                    )
                    self._ownership_certain = ownership
                try:
                    sample = construct_telemetry_sample(
                        sequence=sequence,
                        experiment_run_id=experiment_run_id,
                        scheduled_slot=slot,
                        scheduled_monotonic_ns=scheduled_ns,
                        observed_monotonic_ns=observed_ns,
                        wall_time_utc=self._wall_time(),
                        probe_reading=reading,
                        collector={
                            "healthy": True,
                            "status_code": None,
                            "probe_duration_ns": probe_duration_ns,
                        },
                        watchdog={
                            "healthy": watchdog_healthy,
                            "heartbeat_monotonic_ns": heartbeat_ns,
                            "ownership_certain": ownership,
                        },
                    )
                    validate_telemetry_sample(sample)
                except (TelemetryValidationError, TypeError, ValueError):
                    self._record_collector_failure(
                        "TELEMETRY_SAMPLE_INVALID", observed_ns
                    )
                    return
                except BaseException:
                    self._record_collector_failure(
                        "TELEMETRY_SAMPLE_CONSTRUCTION_FAILED", observed_ns
                    )
                    return

                with self._state_lock:
                    self._samples.append(sample)
                with self._safety_lock:
                    safety = self._safety
                    if safety is None:
                        raise TelemetrySidecarError("SAFETY_STATE_UNAVAILABLE")
                    events = safety.observe_sample(sample)
                self._record_events(events)
                if sequence == 0:
                    self._mark_sample_ready()
                sequence += 1

                slot += 1
                completed_ns = self._clock()
                if start_ns + slot * interval_ns <= completed_ns:
                    slot = (completed_ns - start_ns) // interval_ns + 1
        except BaseException:
            try:
                detected_ns = self._clock()
            except TelemetrySidecarError:
                detected_ns = self._start_monotonic_ns or 0
            self._record_collector_failure("COLLECTOR_THREAD_FAILED", detected_ns)

    def _join_threads(self) -> tuple[bool, bool]:
        collector = self._collector_thread
        watchdog = self._watchdog_thread
        if collector is not None:
            collector.join(self._join_timeout_seconds)
        if watchdog is not None:
            watchdog.join(self._join_timeout_seconds)
        return (
            collector is not None and collector.is_alive(),
            watchdog is not None and watchdog.is_alive(),
        )

    def _abort_start(self) -> None:
        self._stop_event.set()
        collector_alive, watchdog_alive = self._join_threads()
        now = self._start_monotonic_ns or 0
        if collector_alive:
            self._record_collector_failure("COLLECTOR_THREAD_LINGERING", now)
        if watchdog_alive:
            self._record_watchdog_failure("WATCHDOG_THREAD_LINGERING", now)

    def start(self, *, experiment_run_id: str, start_monotonic_ns: int) -> None:
        if not isinstance(experiment_run_id, str) or not _EXPERIMENT_RUN_ID.fullmatch(
            experiment_run_id
        ):
            raise TelemetrySidecarError("EXPERIMENT_RUN_ID_INVALID")
        start_ns = self._require_monotonic_ns(start_monotonic_ns, "START_MONOTONIC_NS")
        with self._state_lock:
            if self._started:
                raise TelemetrySidecarError("SESSION_ALREADY_STARTED")
            self._started = True
            self._experiment_run_id = experiment_run_id
            self._start_monotonic_ns = start_ns
            self._heartbeat_monotonic_ns = start_ns
            self._safety = SafetyStateMachine(
                self._safety_limits,
                run_started_monotonic_ns=start_ns,
            )
            self._collector_thread = threading.Thread(
                target=self._collector_loop,
                name="aptus-telemetry-collector",
                daemon=True,
            )
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                name="aptus-telemetry-watchdog",
                daemon=True,
            )

        # The watchdog exists before the first sample so readiness cannot claim
        # a healthy channel that has never emitted a heartbeat.
        self._watchdog_thread.start()
        self._collector_thread.start()
        if not self._startup_event.wait(self._readiness_timeout_seconds):
            self._record_collector_failure("READINESS_TIMEOUT", start_ns)
            self._abort_start()
            raise TelemetrySidecarError("READINESS_TIMEOUT")

        with self._state_lock:
            collector_failure = self._collector_failure_code
            watchdog_failure = self._watchdog_failure_code
            ready = self._sample_ready and self._watchdog_ready
        if collector_failure is not None or watchdog_failure is not None or not ready:
            code = collector_failure or watchdog_failure or "READINESS_SAMPLE_MISSING"
            self._abort_start()
            raise TelemetrySidecarError(code)

    def safety_signal(self) -> SafetySignal | None:
        """Return the first stop signal observed by either safety channel."""

        with self._state_lock:
            if self._signal is not None:
                return self._signal
            if not self._started or self._stopped or self._stop_event.is_set():
                return None
            collector = self._collector_thread
            watchdog = self._watchdog_thread
            heartbeat = self._heartbeat_monotonic_ns
            collector_alive = collector is not None and collector.is_alive()
            watchdog_alive = watchdog is not None and watchdog.is_alive()
        try:
            now = self._clock()
        except TelemetrySidecarError:
            now = self._start_monotonic_ns or 0
            self._record_collector_failure("MONOTONIC_CLOCK_FAILED", now)
            with self._state_lock:
                return self._signal
        ownership = self._ownership()
        try:
            self._observe_health(
                now_ns=now,
                collector_alive=collector_alive,
                watchdog_alive=watchdog_alive,
                ownership_certain=ownership,
                heartbeat_ns=heartbeat,
            )
        except (TelemetryValidationError, TypeError, ValueError):
            self._record_collector_failure("SAFETY_HEALTH_INVALID", now)
        with self._state_lock:
            return self._signal

    def configuration_record(self) -> dict[str, Any]:
        """Return a detached canonical record safe to persist before startup."""

        return _json_record_copy(self._configuration)

    def snapshot(self) -> TelemetrySnapshot:
        """Return detached current samples/events without stopping collection."""

        with self._state_lock:
            if not self._started:
                raise TelemetrySidecarError("SESSION_NOT_STARTED")
            samples = tuple(_json_record_copy(sample) for sample in self._samples)
            events = _safety_event_records(tuple(self._safety_events))
            failure_code = (
                self._collector_failure_code
                or self._watchdog_failure_code
                or self._qualification_failure_code
            )
        return TelemetrySnapshot(
            samples=samples,
            configuration=self.configuration_record(),
            safety_events=events,
            failure_code=failure_code,
        )

    def stop(self, *, stop_monotonic_ns: int) -> SidecarTelemetryCapture:
        stop_ns = self._require_monotonic_ns(stop_monotonic_ns, "STOP_MONOTONIC_NS")
        with self._state_lock:
            if not self._started:
                raise TelemetrySidecarError("SESSION_NOT_STARTED")
            if self._stopped:
                raise TelemetrySidecarError("SESSION_ALREADY_STOPPED")
            if (
                self._start_monotonic_ns is not None
                and stop_ns < self._start_monotonic_ns
            ):
                raise TelemetrySidecarError("STOP_PRECEDES_START")
            self._stopped = True
            self._requested_stop_ns = stop_ns
            self._stop_event.set()

        collector_alive, watchdog_alive = self._join_threads()
        if collector_alive:
            self._record_collector_failure("COLLECTOR_THREAD_LINGERING", stop_ns)
        if watchdog_alive:
            self._record_watchdog_failure("WATCHDOG_THREAD_LINGERING", stop_ns)

        with self._state_lock:
            samples = tuple(_json_record_copy(sample) for sample in self._samples)
            safety_events = _safety_event_records(tuple(self._safety_events))
            collector_failure = self._collector_failure_code
            watchdog_failure = self._watchdog_failure_code
            qualification_failure = self._qualification_failure_code
            ready = self._sample_ready and self._watchdog_ready

        validation_failure: str | None = None
        expected_run_id = self._experiment_run_id
        for expected_sequence, sample in enumerate(samples):
            try:
                validated = validate_telemetry_sample(sample)
                if (
                    validated["experiment_run_id"] != expected_run_id
                    or validated["sequence"] != expected_sequence
                ):
                    raise TelemetryValidationError("sample identity mismatch")
            except (TelemetryValidationError, TypeError, ValueError):
                validation_failure = "CAPTURE_SAMPLE_INVALID"
                break

        if validation_failure is None and samples:
            start_ns = self._start_monotonic_ns
            if start_ns is None:  # pragma: no cover - guarded by start().
                validation_failure = "CAPTURE_WINDOW_INVALID"
            else:
                try:
                    coverage = telemetry_coverage(samples, start_ns, stop_ns)
                    gap = maximum_gap_seconds(samples, start_ns, stop_ns)
                except (TelemetryValidationError, TypeError, ValueError):
                    validation_failure = "CAPTURE_WINDOW_INVALID"
                else:
                    if (
                        coverage < MINIMUM_QUALIFYING_COVERAGE
                        or gap > self._safety_limits.qualifying_gap_seconds
                    ):
                        qualification_failure = "TELEMETRY_QUALIFYING_GAP"

        failure_code = (
            collector_failure
            or watchdog_failure
            or validation_failure
            or qualification_failure
            or (None if ready and samples else "READINESS_SAMPLE_MISSING")
        )
        return SidecarTelemetryCapture(
            samples=samples,
            healthy=failure_code is None,
            failure_code=failure_code,
            configuration=self.configuration_record(),
            safety_events=safety_events,
        )

    @property
    def qualifying_profile(self) -> bool:
        return _qualifying_session_is_authentic(self)

    @property
    def collector_alive(self) -> bool:
        thread = self._collector_thread
        return thread is not None and thread.is_alive()

    @property
    def watchdog_alive(self) -> bool:
        thread = self._watchdog_thread
        return thread is not None and thread.is_alive()


def _install_qualifying_session_factory() -> tuple[
    Callable[..., BackgroundTelemetrySession],
    Callable[[object, object | None], bool],
]:
    """Mint runtime authority only after constructing every concrete provider."""

    registrations: weakref.WeakKeyDictionary[
        BackgroundTelemetrySession, dict[str, object]
    ] = weakref.WeakKeyDictionary()

    def create(
        *,
        harness: object,
        authority: object,
        filesystem_path: Path,
        gpu_index: int,
        nvidia_smi_path: str | None,
        unavailable_optional_sensors: tuple[str, ...],
        readiness_timeout_seconds: float,
        join_timeout_seconds: float,
    ) -> BackgroundTelemetrySession:
        # Imports are local because harness imports this module only when it
        # creates its telemetry session.
        from aptus.execution import JobService

        from .admission import Phase4CurrentAuthority
        from .harness import CaptureHarness
        from .phase4 import Phase4SourceFreezeVerification
        from .qualification import QualifyingRunContext

        context = getattr(harness, "qualification_context", None)
        service = getattr(harness, "job_service", None)
        try:
            harness_authorized = (
                CaptureHarness._authorized_for_qualifying_factory.__get__(
                    harness, CaptureHarness
                )(authority)
                is True
            )
        except (AttributeError, TypeError):
            harness_authorized = False
        if (
            type(harness) is not CaptureHarness
            or not harness_authorized
            or authority is None
            or getattr(harness, "_qualifying_authority", None) is not authority
            or type(context) is not QualifyingRunContext
            or context.production_qualifying is not True
            or type(service) is not JobService
            or getattr(harness, "_qualifying_job_service", None) is not service
            or type(getattr(harness, "_activation_authority", None))
            is not Phase4CurrentAuthority
            or type(getattr(harness, "_phase4_verification", None))
            is not Phase4SourceFreezeVerification
            or getattr(harness, "_monotonic_ns", None) is not time.monotonic_ns
            or getattr(harness, "_wall_time", None) is not utc_now
            or getattr(harness, "_sleep", None) is not time.sleep
        ):
            raise TypeError(
                "Qualifying telemetry requires an exact production harness."
            )
        if (
            isinstance(gpu_index, bool)
            or not isinstance(gpu_index, int)
            or gpu_index < 0
            or gpu_index != getattr(harness, "_phase4_gpu_index", None)
            or nvidia_smi_path != getattr(harness, "_phase4_nvidia_smi_path", None)
        ):
            raise ValueError("Qualifying GPU authority differs from Phase 4.")
        declarations = tuple(unavailable_optional_sensors)
        if len(declarations) != 2 or set(declarations) != {
            "cpu_temperature",
            "nvme_temperature",
        }:
            raise ValueError(
                "Qualifying telemetry requires the exact optional-sensor declaration."
            )
        filesystem = filesystem_path.resolve(strict=True)
        planned_slot = getattr(harness, "_planned_slot_context", None)
        try:
            admitted_filesystem = Path(planned_slot.run_proposal.bundle_path).resolve(
                strict=True
            )
        except (AttributeError, OSError, TypeError):
            raise ValueError(
                "Qualifying telemetry admission filesystem is unavailable."
            ) from None
        if (
            not filesystem.is_dir()
            or filesystem != admitted_filesystem
            or filesystem.stat().st_dev
            != getattr(harness, "_admission_filesystem_device", None)
        ):
            raise ValueError("Qualifying telemetry filesystem differs from admission.")

        trusted_nvidia = resolve_trusted_nvidia_smi(nvidia_smi_path)
        thermal = detect_nvidia_thermal_limit_authority(
            trusted_nvidia, gpu_index=gpu_index
        )
        journal = LinuxNvidiaJournalEventProvider.production()
        disk_growth = StatvfsDiskGrowthProvider.production(filesystem)
        if (
            getattr(journal, "production_authorized", False) is not True
            or getattr(disk_growth, "production_authorized", False) is not True
        ):
            raise TypeError("Qualifying telemetry providers lack production authority.")

        managed_pids = CaptureHarness.managed_pids.__get__(harness, CaptureHarness)
        managed_groups = CaptureHarness.managed_process_groups.__get__(
            harness, CaptureHarness
        )
        ownership_certain = CaptureHarness._qualifying_ownership_certain.__get__(
            harness, CaptureHarness
        )
        lease_active = JobService.campaign_lease_active.__get__(service, JobService)
        kernel_events = journal.snapshot
        probe = LinuxNvidiaHostProbe(
            filesystem_path=filesystem,
            managed_pids=managed_pids,
            managed_process_groups=managed_groups,
            kernel_events=kernel_events,
            lease_active=lease_active,
            disk_growth_bytes=disk_growth,
            gpu_index=gpu_index,
            nvidia_smi_path=trusted_nvidia.path,
            trusted_nvidia_executable=trusted_nvidia,
            cpu_temperature=None,
            nvme_temperature=None,
            gpu_thermal_limits=thermal.provider,
        )
        journal_bindings = journal.support_bindings()
        session = BackgroundTelemetrySession.qualifying_production(
            probe=probe,
            ownership_certain=ownership_certain,
            emergency_deadline_seconds=context.emergency_deadline_seconds,
            remaining_disk_budget_bytes=context.remaining_disk_budget_bytes,
            initial_thermal_limits_available=(thermal.status == "supported"),
            provider_name="linux-nvidia-host-probe",
            provider_version="aptus-cuda-campaign-v1",
            support_bindings={
                "cpu_temperature": "unsupported:reviewed-not-configured",
                "gpu_thermal_limits": thermal.support_binding,
                "hardware_events": journal_bindings["hardware_events"],
                "nvidia_smi_binary": f"sha256:{trusted_nvidia.binding_sha256}",
                "nvme_temperature": "unsupported:reviewed-not-configured",
                "xid_projection": journal_bindings["xid_projection"],
            },
            ownership_binding="factory-owned-job-service-process-group-v1",
            disk_growth_binding="factory-owned-statvfs-baseline-v1",
            readiness_timeout_seconds=readiness_timeout_seconds,
            join_timeout_seconds=join_timeout_seconds,
            monotonic_ns=time.monotonic_ns,
            wall_time=utc_now,
            sleep=time.sleep,
        )
        session._qualifying_profile = True
        session._qualifying_authority = authority
        registrations[session] = {
            "authority": authority,
            "configuration_bytes": _canonical_json_bytes(session._configuration),
            "disk_growth": disk_growth,
            "join_timeout_seconds": session._join_timeout_seconds,
            "journal": journal,
            "kernel_events": kernel_events,
            "lease_active": lease_active,
            "managed_groups": managed_groups,
            "managed_pids": managed_pids,
            "ownership_certain": ownership_certain,
            "probe": probe,
            "probe_command_runner": probe._command_runner,
            "probe_gpu_thermal_limits": probe._gpu_thermal_limits,
            "probe_filesystem_path": probe.filesystem_path,
            "probe_gpu_index": probe.gpu_index,
            "probe_nvidia_smi_path": probe._nvidia_smi_path,
            "probe_proc_root": probe.proc_root,
            "probe_statvfs": probe._statvfs,
            "readiness_timeout_seconds": session._readiness_timeout_seconds,
            "safety_limits": session._safety_limits,
            "trusted_nvidia": trusted_nvidia,
        }
        return session

    def is_authentic(value: object, authority: object | None = None) -> bool:
        if type(value) is not BackgroundTelemetrySession:
            return False
        registration = registrations.get(value)
        if registration is None:
            return False
        registered_authority = registration["authority"]
        probe = registration["probe"]
        journal = registration["journal"]
        disk_growth = registration["disk_growth"]
        try:
            current_configuration = _canonical_json_bytes(value._configuration)
        except (TypeError, ValueError):
            return False
        return bool(
            (authority is None or registered_authority is authority)
            and value._qualifying_profile is True
            and value._qualifying_authority is registered_authority
            and value._probe is probe
            and value._ownership_provider is registration["ownership_certain"]
            and value._safety_limits is registration["safety_limits"]
            and value._readiness_timeout_seconds
            == registration["readiness_timeout_seconds"]
            and value._join_timeout_seconds == registration["join_timeout_seconds"]
            and value._monotonic_ns is time.monotonic_ns
            and value._wall_time is utc_now
            and value._sleep is time.sleep
            and current_configuration == registration["configuration_bytes"]
            and type(probe) is LinuxNvidiaHostProbe
            and probe._managed_pids is registration["managed_pids"]
            and probe._managed_process_groups is registration["managed_groups"]
            and probe._lease_active is registration["lease_active"]
            and probe._kernel_events is registration["kernel_events"]
            and probe._disk_growth_bytes is disk_growth
            and probe._trusted_nvidia_executable is registration["trusted_nvidia"]
            and probe.filesystem_path == registration["probe_filesystem_path"]
            and probe.gpu_index == registration["probe_gpu_index"]
            and probe._nvidia_smi_path == registration["probe_nvidia_smi_path"]
            and probe._command_runner is registration["probe_command_runner"]
            and probe._gpu_thermal_limits is registration["probe_gpu_thermal_limits"]
            and probe.proc_root == registration["probe_proc_root"]
            and probe._statvfs is registration["probe_statvfs"]
            and probe._optional_sensors == {"cpu": None, "nvme": None}
            and journal.production_authorized is True
            and disk_growth.production_authorized is True
        )

    return create, is_authentic


(
    _create_qualifying_session_for_harness,
    _qualifying_session_is_authentic,
) = _install_qualifying_session_factory()
del _install_qualifying_session_factory


__all__ = [
    "BackgroundTelemetrySession",
    "SidecarTelemetryCapture",
    "TelemetrySidecarError",
    "TelemetrySnapshot",
]
