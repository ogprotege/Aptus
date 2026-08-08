"""Fail-closed pre-slot admission and atomic identity activation.

Admission is deliberately an operational layer, not a Phase-1 raw evidence
record.  Its observations therefore never carry an ``experiment_run_id``.
The execution-configuration identifier and opaque experiment-run identifier
are created only after the complete idle gate has passed and its authority has
been revalidated.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import sys
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from .contracts import (
    REASON_CODES,
    SCHEMA_VERSIONS,
    canonical_json_bytes,
    compact_canonical_json_bytes,
    deterministic_id,
    new_opaque_id,
    sha256_bytes,
    utc_now,
    validate_record,
)
from .monitoring import (
    GIB,
    NANOSECONDS_PER_SECOND,
    SAMPLE_INTERVAL_SECONDS,
    TelemetryValidationError,
    normalize_observation_facts,
    validate_cooldown_observations,
    validate_observation_facts,
)


ADMISSION_OBSERVATION_SCHEMA = "aptus.cuda-campaign-admission-observation.v1"
ADMISSION_DECISION_SCHEMA = "aptus.cuda-campaign-admission-decision.v1"
ACTIVATION_DECISION_SCHEMA = "aptus.cuda-campaign-activation-decision.v1"
STARTED_IDENTITY_TEMPLATE_SCHEMA = "aptus.cuda-campaign-started-identity-template.v1"
ACTIVATION_SEAL_SCHEMA = "aptus.cuda-campaign-activation-seal.v1"

REQUIRED_IDLE_OBSERVATIONS = 120
MAXIMUM_ADMISSION_ACQUISITION_SECONDS = 1800

_RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?\+00:00$")
MINIMUM_ZERO_UTILIZATION_OBSERVATIONS = 110
BASE_DISK_RESERVE_BYTES = 32 * GIB
HOST_RAM_RESERVE_BYTES = 8 * GIB
DOWNLOAD_FIXED_RESERVE_BYTES = 2 * GIB

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_CELL_ID = re.compile(r"^cell_[0-9a-f]{20}$")
_SLOT_ID = re.compile(r"^slot_[0-9a-f]{20}$")
_EXECUTION_ID = re.compile(r"^exec_[0-9a-f]{20}$")
_EXPERIMENT_RUN_ID = re.compile(r"^xrun_[0-9a-f]{32}$")
_PENDING_ARGV = ["PENDING_MANAGED_TRAIN_SUBMISSION"]

_PHASE4_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "phase4_source_freeze_sha256",
        "phase4_source_freeze_seal_sha256",
        "idle_baseline_samples_sha256",
        "telemetry_configuration_sha256",
        "host_binding_sha256",
        "current_host_binding_sha256",
        "current_boot_id_sha256",
        "journalctl_binding_sha256",
        "summary",
    }
)
_BASELINE_SUMMARY_FIELDS = frozenset(
    {
        "gpu_temperature_median_c",
        "gpu_temperature_p95_c",
        "gpu_free_vram_median_bytes",
        "gpu_power_draw_p95_w",
    }
)
_OBSERVATION_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "admission_context_sha256",
        "current_authority_sha256",
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
_ACTIVATION_FILE_NAMES = (
    "admission-decision.json",
    "admission-observations.json",
    "execution-configuration.json",
    "experiment-run-template.json",
    "started-identity-template.json",
    "activation-decision.json",
)
ACTIVATION_FILE_NAMES = _ACTIVATION_FILE_NAMES
ACTIVATION_SEAL_NAME = "ACTIVATED.json"
_ADMISSION_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "admitted",
        "reason_codes",
        "planned_attempt_slot_sha256",
        "admission_context_sha256",
        "resource_budget_sha256",
        "current_authority_sha256",
        "observations_sha256",
        "observation_count",
        "acquisition_started_monotonic_ns",
        "acquisition_finished_monotonic_ns",
        "acquisition_duration_ns",
        "summary",
        "production_authority",
        "production_observations",
    }
)
_ACTIVATION_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "activation_status",
        "admission_decision_sha256",
        "admission_context_sha256",
        "resource_budget_sha256",
        "current_authority_sha256",
        "execution_configuration_sha256",
        "experiment_run_template_sha256",
        "started_identity_template_sha256",
        "execution_configuration_id",
        "experiment_run_id",
        "activated_monotonic_ns",
        "activated_at_utc",
        "production_qualifying",
    }
)
_STARTED_TEMPLATE_FIELDS = frozenset(
    {
        "schema_version",
        "attempt_slot_id",
        "planned_attempt_slot_sha256",
        "execution_configuration_id",
        "experiment_run_id",
        "terminal_record_status",
    }
)


class AdmissionError(ValueError):
    """A pre-slot gate or activation invariant was violated."""


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AdmissionError(f"{label} must be a nonnegative integer")
    return value


def _positive_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise AdmissionError(f"{label} must be positive and finite")
    return float(value)


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise AdmissionError(f"{label} is invalid")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise AdmissionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _normalized_utc_timestamp(value: Any, label: str) -> str:
    if type(value) is not str or _RFC3339_UTC.fullmatch(value) is None:
        raise AdmissionError(f"{label} must be normalized RFC 3339 UTC with +00:00")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise AdmissionError(f"{label} is not a real calendar timestamp") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise AdmissionError(f"{label} must use UTC")
    return value


def _ceil_ratio(value: int, numerator: int, denominator: int) -> int:
    """Return ``ceil(value * numerator / denominator)`` using integers."""

    return (value * numerator + denominator - 1) // denominator


def _detached_json_object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise AdmissionError(f"{label} must be an exact JSON object")
    try:
        return json.loads(compact_canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise AdmissionError(f"{label} is not JSON-safe") from error


def _validate_phase4_binding(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _PHASE4_BINDING_FIELDS:
        raise AdmissionError("Phase-4 binding has the wrong exact fields")
    result = dict(value)
    if result["schema_version"] != "aptus.cuda-campaign-idle-baseline-binding.v1":
        raise AdmissionError("Phase-4 binding schema is unsupported")
    for name in _PHASE4_BINDING_FIELDS - {"schema_version", "summary"}:
        _digest(result[name], f"Phase-4 {name}")
    summary = result["summary"]
    if type(summary) is not dict or set(summary) != _BASELINE_SUMMARY_FIELDS:
        raise AdmissionError("Phase-4 idle summary has the wrong exact fields")
    for name, number in summary.items():
        if name == "gpu_power_draw_p95_w" and number is None:
            continue
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number < 0
        ):
            raise AdmissionError(f"Phase-4 idle summary {name} is invalid")
    result["summary"] = dict(summary)
    return result


@dataclass(frozen=True)
class FrozenResourceBudget:
    """Exact, content-bound admission budgets; no input has a default."""

    plan_id: str
    candidate_id: str
    bundle_fingerprint: str
    comparison_cell_id: str
    attempt_slot_id: str
    exact_artifact_bytes: int
    plan_required_disk_bytes: int
    largest_pilot_checkpoint_bytes: int
    final_export_bytes: int
    expected_copied_output_bytes: int
    expected_log_bytes: int
    expected_telemetry_bytes: int
    plan_required_host_ram_bytes: int

    def __post_init__(self) -> None:
        _identifier(self.plan_id, "plan_id")
        _identifier(self.candidate_id, "candidate_id")
        _digest(self.bundle_fingerprint, "bundle_fingerprint")
        if (
            not isinstance(self.comparison_cell_id, str)
            or _CELL_ID.fullmatch(self.comparison_cell_id) is None
        ):
            raise AdmissionError("comparison_cell_id is invalid")
        if (
            not isinstance(self.attempt_slot_id, str)
            or _SLOT_ID.fullmatch(self.attempt_slot_id) is None
        ):
            raise AdmissionError("attempt_slot_id is invalid")
        for name in (
            "exact_artifact_bytes",
            "plan_required_disk_bytes",
            "largest_pilot_checkpoint_bytes",
            "final_export_bytes",
            "expected_copied_output_bytes",
            "expected_log_bytes",
            "expected_telemetry_bytes",
            "plan_required_host_ram_bytes",
        ):
            value = _nonnegative_integer(getattr(self, name), name)
            if value == 0:
                raise AdmissionError(f"{name} must be positive")

    @property
    def download_bytes(self) -> int:
        return (
            _ceil_ratio(self.exact_artifact_bytes, 5, 4) + DOWNLOAD_FIXED_RESERVE_BYTES
        )

    @property
    def output_bytes(self) -> int:
        checkpoint_and_export = 4 * self.largest_pilot_checkpoint_bytes + (
            self.final_export_bytes
        )
        return _ceil_ratio(
            max(self.plan_required_disk_bytes, checkpoint_and_export), 5, 4
        )

    @property
    def vault_bytes(self) -> int:
        expected = (
            self.expected_copied_output_bytes
            + self.expected_log_bytes
            + self.expected_telemetry_bytes
        )
        return _ceil_ratio(expected, 11, 10)

    @property
    def admission_disk_bytes(self) -> int:
        return (
            BASE_DISK_RESERVE_BYTES
            + self.download_bytes
            + self.output_bytes
            + self.vault_bytes
        )

    @property
    def admission_host_ram_bytes(self) -> int:
        return self.plan_required_host_ram_bytes + HOST_RAM_RESERVE_BYTES

    def record(self) -> dict[str, Any]:
        return {
            "schema_version": "aptus.cuda-campaign-frozen-resource-budget.v1",
            "bindings": {
                "plan_id": self.plan_id,
                "candidate_id": self.candidate_id,
                "bundle_fingerprint": self.bundle_fingerprint,
                "comparison_cell_id": self.comparison_cell_id,
                "attempt_slot_id": self.attempt_slot_id,
            },
            "inputs": {
                "exact_artifact_bytes": self.exact_artifact_bytes,
                "plan_required_disk_bytes": self.plan_required_disk_bytes,
                "largest_pilot_checkpoint_bytes": self.largest_pilot_checkpoint_bytes,
                "final_export_bytes": self.final_export_bytes,
                "expected_copied_output_bytes": self.expected_copied_output_bytes,
                "expected_log_bytes": self.expected_log_bytes,
                "expected_telemetry_bytes": self.expected_telemetry_bytes,
                "plan_required_host_ram_bytes": self.plan_required_host_ram_bytes,
            },
            "derived": {
                "download_bytes": self.download_bytes,
                "output_bytes": self.output_bytes,
                "vault_bytes": self.vault_bytes,
                "base_disk_reserve_bytes": BASE_DISK_RESERVE_BYTES,
                "admission_disk_bytes": self.admission_disk_bytes,
                "host_ram_reserve_bytes": HOST_RAM_RESERVE_BYTES,
                "admission_host_ram_bytes": self.admission_host_ram_bytes,
            },
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(compact_canonical_json_bytes(self.record()))


@dataclass(frozen=True)
class ExecutionProposal:
    """Exact execution inputs before the deterministic execution ID exists."""

    exact_behavior_values: Mapping[str, Any]
    plan_id: str
    candidate_id: str
    bundle_fingerprint: str
    split_seed: int
    training_seed: int
    data_order_seed: int
    emergency_deadline_seconds: float

    def __post_init__(self) -> None:
        behavior = _detached_json_object(
            dict(self.exact_behavior_values), "exact_behavior_values"
        )
        deadline = _positive_number(
            self.emergency_deadline_seconds, "emergency_deadline_seconds"
        )
        if behavior.get("emergency_deadline_seconds") != (
            self.emergency_deadline_seconds
        ):
            raise AdmissionError("exact behavior must bind the same emergency deadline")
        _identifier(self.plan_id, "plan_id")
        _identifier(self.candidate_id, "candidate_id")
        _digest(self.bundle_fingerprint, "bundle_fingerprint")
        for name in ("split_seed", "training_seed", "data_order_seed"):
            _nonnegative_integer(getattr(self, name), name)
        object.__setattr__(self, "exact_behavior_values", MappingProxyType(behavior))
        object.__setattr__(self, "emergency_deadline_seconds", deadline)


@dataclass(frozen=True)
class RunProposal:
    """Exact run-template inputs before an opaque run ID is minted."""

    working_directory: str
    fresh_state_root: str
    bundle_path: str
    output_path: str
    bundle_manifest_sha256: str
    archive_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "working_directory",
            "fresh_state_root",
            "bundle_path",
            "output_path",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or "\x00" in value
                or not Path(value).is_absolute()
            ):
                raise AdmissionError(f"{name} must be an absolute safe path")
        _digest(self.bundle_manifest_sha256, "bundle_manifest_sha256")
        _digest(self.archive_sha256, "archive_sha256")


@dataclass(frozen=True)
class PlannedSlotContext:
    """All frozen proposal facts, with only the planned slot ID assigned."""

    campaign: Mapping[str, Any]
    comparison_cohort: Mapping[str, Any]
    comparison_cell: Mapping[str, Any]
    planned_attempt_slot: Mapping[str, Any]
    execution_proposal: ExecutionProposal
    run_proposal: RunProposal
    phase4_binding: Mapping[str, Any]
    resource_budget: FrozenResourceBudget

    def __post_init__(self) -> None:
        try:
            campaign = validate_record(dict(self.campaign), SCHEMA_VERSIONS["campaign"])
            cohort = validate_record(
                dict(self.comparison_cohort), SCHEMA_VERSIONS["comparison_cohort"]
            )
            cell = validate_record(
                dict(self.comparison_cell), SCHEMA_VERSIONS["comparison_cell"]
            )
            slot = validate_record(
                dict(self.planned_attempt_slot), SCHEMA_VERSIONS["attempt_slot"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AdmissionError("planned slot identity chain is invalid") from error
        if (
            slot["slot_status"] != "planned-not-started"
            or slot["execution_configuration_id"] is not None
            or slot["experiment_run_id"] is not None
            or slot["native_outcome"] is not None
            or slot["evidence_status"] != "not-started"
        ):
            raise AdmissionError("context must contain an unstarted canonical slot")
        if (
            cohort["campaign_id"] != campaign["campaign_id"]
            or cell["campaign_id"] != campaign["campaign_id"]
            or cell["comparison_cell_id"] not in cohort["member_cell_ids"]
            or slot["comparison_cohort_id"] != cohort["comparison_cohort_id"]
            or slot["comparison_cell_id"] != cell["comparison_cell_id"]
        ):
            raise AdmissionError("campaign, cohort, cell, and slot are misbound")
        proposal = self.execution_proposal
        if (
            proposal.training_seed != slot["scheduled_seed"]
            or proposal.data_order_seed != 1_000_000 + slot["scheduled_seed"]
            or (
                "split_seed" in cell["seed_policy"]
                and proposal.split_seed != cell["seed_policy"]["split_seed"]
            )
        ):
            raise AdmissionError("execution seed proposal is misbound")
        budget = self.resource_budget
        if (
            budget.plan_id != proposal.plan_id
            or budget.candidate_id != proposal.candidate_id
            or budget.bundle_fingerprint != proposal.bundle_fingerprint
            or budget.comparison_cell_id != cell["comparison_cell_id"]
            or budget.attempt_slot_id != slot["attempt_slot_id"]
        ):
            raise AdmissionError("resource budget identity is misbound")
        phase4 = _validate_phase4_binding(dict(self.phase4_binding))
        phase4_sha256 = sha256_bytes(compact_canonical_json_bytes(phase4))
        if (
            proposal.exact_behavior_values.get("resource_budget_sha256")
            != budget.sha256
            or proposal.exact_behavior_values.get("phase4_binding_sha256")
            != phase4_sha256
        ):
            raise AdmissionError(
                "exact behavior does not bind its resource budget and Phase-4 authority"
            )
        if phase4["host_binding_sha256"] != sha256_bytes(
            canonical_json_bytes(cell["host_binding"])
        ):
            raise AdmissionError("Phase-4 binding does not bind the comparison host")
        object.__setattr__(self, "campaign", MappingProxyType(campaign))
        object.__setattr__(self, "comparison_cohort", MappingProxyType(cohort))
        object.__setattr__(self, "comparison_cell", MappingProxyType(cell))
        object.__setattr__(self, "planned_attempt_slot", MappingProxyType(slot))
        object.__setattr__(self, "phase4_binding", MappingProxyType(phase4))

    @property
    def sha256(self) -> str:
        return sha256_bytes(compact_canonical_json_bytes(self.record()))

    def record(self) -> dict[str, Any]:
        return {
            "campaign": dict(self.campaign),
            "comparison_cohort": dict(self.comparison_cohort),
            "comparison_cell": dict(self.comparison_cell),
            "planned_attempt_slot": dict(self.planned_attempt_slot),
            "execution_proposal": {
                "exact_behavior_values": dict(
                    self.execution_proposal.exact_behavior_values
                ),
                "plan_id": self.execution_proposal.plan_id,
                "candidate_id": self.execution_proposal.candidate_id,
                "bundle_fingerprint": self.execution_proposal.bundle_fingerprint,
                "split_seed": self.execution_proposal.split_seed,
                "training_seed": self.execution_proposal.training_seed,
                "data_order_seed": self.execution_proposal.data_order_seed,
                "emergency_deadline_seconds": (
                    self.execution_proposal.emergency_deadline_seconds
                ),
            },
            "run_proposal": {
                "working_directory": self.run_proposal.working_directory,
                "fresh_state_root": self.run_proposal.fresh_state_root,
                "bundle_path": self.run_proposal.bundle_path,
                "output_path": self.run_proposal.output_path,
                "bundle_manifest_sha256": self.run_proposal.bundle_manifest_sha256,
                "archive_sha256": self.run_proposal.archive_sha256,
            },
            "phase4_binding": dict(self.phase4_binding),
            "resource_budget": self.resource_budget.record(),
            "resource_budget_sha256": self.resource_budget.sha256,
        }


def planned_slot_context_from_record(value: Any) -> PlannedSlotContext:
    """Reconstruct and exactly reproduce one retained pre-admission context."""

    expected_fields = {
        "campaign",
        "comparison_cohort",
        "comparison_cell",
        "planned_attempt_slot",
        "execution_proposal",
        "run_proposal",
        "phase4_binding",
        "resource_budget",
        "resource_budget_sha256",
    }
    if type(value) is not dict or set(value) != expected_fields:
        raise AdmissionError("retained planned-slot context fields are invalid")
    execution = value["execution_proposal"]
    run = value["run_proposal"]
    budget_record = value["resource_budget"]
    if (
        type(execution) is not dict
        or set(execution)
        != {
            "exact_behavior_values",
            "plan_id",
            "candidate_id",
            "bundle_fingerprint",
            "split_seed",
            "training_seed",
            "data_order_seed",
            "emergency_deadline_seconds",
        }
        or type(run) is not dict
        or set(run)
        != {
            "working_directory",
            "fresh_state_root",
            "bundle_path",
            "output_path",
            "bundle_manifest_sha256",
            "archive_sha256",
        }
        or type(budget_record) is not dict
        or set(budget_record) != {"schema_version", "bindings", "inputs", "derived"}
        or budget_record["schema_version"]
        != "aptus.cuda-campaign-frozen-resource-budget.v1"
        or type(budget_record["bindings"]) is not dict
        or type(budget_record["inputs"]) is not dict
    ):
        raise AdmissionError("retained planned-slot context payload is invalid")
    bindings = budget_record["bindings"]
    inputs = budget_record["inputs"]
    try:
        budget = FrozenResourceBudget(
            plan_id=bindings["plan_id"],
            candidate_id=bindings["candidate_id"],
            bundle_fingerprint=bindings["bundle_fingerprint"],
            comparison_cell_id=bindings["comparison_cell_id"],
            attempt_slot_id=bindings["attempt_slot_id"],
            exact_artifact_bytes=inputs["exact_artifact_bytes"],
            plan_required_disk_bytes=inputs["plan_required_disk_bytes"],
            largest_pilot_checkpoint_bytes=inputs["largest_pilot_checkpoint_bytes"],
            final_export_bytes=inputs["final_export_bytes"],
            expected_copied_output_bytes=inputs["expected_copied_output_bytes"],
            expected_log_bytes=inputs["expected_log_bytes"],
            expected_telemetry_bytes=inputs["expected_telemetry_bytes"],
            plan_required_host_ram_bytes=inputs["plan_required_host_ram_bytes"],
        )
        context = PlannedSlotContext(
            campaign=value["campaign"],
            comparison_cohort=value["comparison_cohort"],
            comparison_cell=value["comparison_cell"],
            planned_attempt_slot=value["planned_attempt_slot"],
            execution_proposal=ExecutionProposal(**execution),
            run_proposal=RunProposal(**run),
            phase4_binding=value["phase4_binding"],
            resource_budget=budget,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AdmissionError("retained planned-slot context is invalid") from error
    if (
        budget.record() != budget_record
        or value["resource_budget_sha256"] != budget.sha256
        or context.record() != value
    ):
        raise AdmissionError("retained planned-slot context does not reproduce")
    return context


class CurrentAuthority(Protocol):
    """Re-readable source/host authority used at gate and activation."""

    @property
    def production_qualifying(self) -> bool: ...

    def snapshot(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class InjectedAdmissionAuthority:
    """Test-only authority.  It can never produce qualifying activation."""

    reader: Callable[[], Mapping[str, Any]]

    @property
    def production_qualifying(self) -> bool:
        return False

    def snapshot(self) -> Mapping[str, Any]:
        return self.reader()


@dataclass(frozen=True)
class Phase4CurrentAuthority:
    """Production authority that always invokes the non-injected verifier."""

    directory: Path
    repository_root: Path
    campaign: Mapping[str, Any]
    comparison_cohort: Mapping[str, Any]
    comparison_cell: Mapping[str, Any]
    nvidia_smi_path: str | None = None
    gpu_index: int = 0

    @property
    def production_qualifying(self) -> bool:
        return True

    def snapshot(self) -> Mapping[str, Any]:
        # Kept local to avoid making the Phase-4 artifact depend on admission.
        from .phase4 import verify_phase4_source_freeze_artifact

        verification = verify_phase4_source_freeze_artifact(
            self.directory,
            repository_root=self.repository_root,
            campaign=self.campaign,
            comparison_cohort=self.comparison_cohort,
            comparison_cell=self.comparison_cell,
            nvidia_smi_path=self.nvidia_smi_path,
            gpu_index=self.gpu_index,
        )
        return dict(verification.baseline_binding)


class _ProductionAdmissionObservationBatch:
    """Capability-marked result of the concrete local-host collector."""

    __slots__ = (
        "_observations_payload",
        "acquisition_started_monotonic_ns",
        "acquisition_finished_monotonic_ns",
        "_authority",
        "_sealed",
        "__weakref__",
    )

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("Use collect_production_admission_observations().")

    def _initialize_from_collector(
        self,
        *,
        observations: Sequence[Mapping[str, Any]],
        acquisition_started_monotonic_ns: int,
        acquisition_finished_monotonic_ns: int,
        authority: object,
    ) -> None:
        validated = [
            validate_admission_observation(dict(item)) for item in observations
        ]
        object.__setattr__(
            self,
            "_observations_payload",
            compact_canonical_json_bytes(validated),
        )
        object.__setattr__(
            self,
            "acquisition_started_monotonic_ns",
            _nonnegative_integer(
                acquisition_started_monotonic_ns,
                "acquisition_started_monotonic_ns",
            ),
        )
        object.__setattr__(
            self,
            "acquisition_finished_monotonic_ns",
            _nonnegative_integer(
                acquisition_finished_monotonic_ns,
                "acquisition_finished_monotonic_ns",
            ),
        )
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("production admission batch is immutable")
        object.__setattr__(self, name, value)

    @property
    def observations(self) -> tuple[Mapping[str, Any], ...]:
        values = json.loads(self._observations_payload)
        return tuple(MappingProxyType(item) for item in values)

    def authorized_for(self, authority: CurrentAuthority) -> bool:
        return _production_batch_is_authentic(self, authority)


class _AdmissionOwnershipWatchdog:
    """Independent real-clock heartbeat for the pre-slot lease boundary."""

    def __init__(self, lease_active: Callable[[], bool]) -> None:
        self._lease_active = lease_active
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._heartbeat = time.monotonic_ns()
        self._healthy = True
        self._ownership_certain = False
        self._thread = threading.Thread(
            target=self._run,
            name="aptus-admission-ownership-watchdog",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=1.0):
            self._stop.set()
            self._thread.join(timeout=1.0)
            raise AdmissionError("admission watchdog did not become ready")
        with self._lock:
            if not self._healthy:
                raise AdmissionError("admission watchdog ownership probe failed")

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic_ns()
            try:
                ownership = not self._lease_active()
            except Exception:
                ownership = False
                with self._lock:
                    self._healthy = False
            with self._lock:
                self._heartbeat = now
                self._ownership_certain = ownership
                self._ready.set()
            self._stop.wait(0.25)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "healthy": self._healthy and self._thread.is_alive(),
                "heartbeat_monotonic_ns": self._heartbeat,
                "ownership_certain": self._ownership_certain,
            }

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise AdmissionError("admission watchdog did not join")


def _real_sleep_until(target_ns: int, deadline_ns: int) -> bool:
    while True:
        now = time.monotonic_ns()
        if now >= target_ns:
            return now <= deadline_ns
        if now > deadline_ns:
            return False
        time.sleep(min(0.1, (target_ns - now) / NANOSECONDS_PER_SECOND))


def _collect_production_admission_payload(
    context: PlannedSlotContext,
    *,
    authority: Phase4CurrentAuthority,
    filesystem_path: Path,
    job_service: Any,
    gpu_index: int = 0,
    nvidia_smi_path: str | None = None,
) -> tuple[list[dict[str, Any]], int, int, Phase4CurrentAuthority]:
    """Capture the only production-authorized pre-slot observation batch.

    Clocks, sleeping, NVIDIA identity, journal evidence, filesystem accounting,
    lease state, and the watchdog are all concrete here.  No injected probe or
    clock can acquire the private production capability.
    """

    if type(authority) is not Phase4CurrentAuthority:
        raise TypeError("production admission requires Phase4CurrentAuthority")
    if not sys.platform.startswith("linux"):
        raise AdmissionError("production CUDA admission requires Linux")
    from aptus.execution import JobService
    from .monitoring import (
        LinuxNvidiaHostProbe,
        LinuxNvidiaJournalEventProvider,
        ProbeFailure,
        StatvfsDiskGrowthProvider,
        detect_nvidia_thermal_limit_authority,
        resolve_trusted_nvidia_smi,
    )

    if type(job_service) is not JobService:
        raise TypeError("production admission requires the concrete JobService")
    lease_reader = job_service.campaign_lease_active
    if getattr(lease_reader, "__func__", None) is not JobService.campaign_lease_active:
        raise TypeError("production admission lease authority was replaced")
    if gpu_index != authority.gpu_index or nvidia_smi_path != authority.nvidia_smi_path:
        raise AdmissionError("collector GPU authority differs from Phase-4 authority")
    filesystem = filesystem_path.resolve(strict=True)
    if not filesystem.is_dir():
        raise AdmissionError("admission filesystem path must be a directory")
    initial_authority, authority_sha256 = authority_snapshot(authority)
    if initial_authority != dict(context.phase4_binding):
        raise AdmissionError("production authority differs from the planned slot")
    trusted_nvidia = resolve_trusted_nvidia_smi(nvidia_smi_path)
    thermal = detect_nvidia_thermal_limit_authority(trusted_nvidia, gpu_index=gpu_index)
    journal = LinuxNvidiaJournalEventProvider.production()
    disk_growth = StatvfsDiskGrowthProvider.production(filesystem)
    probe = LinuxNvidiaHostProbe(
        filesystem_path=filesystem,
        managed_pids=lambda: (),
        managed_process_groups=lambda: (),
        kernel_events=journal.snapshot,
        lease_active=lease_reader,
        disk_growth_bytes=disk_growth,
        gpu_index=gpu_index,
        nvidia_smi_path=trusted_nvidia.path,
        trusted_nvidia_executable=trusted_nvidia,
        cpu_temperature=None,
        nvme_temperature=None,
        gpu_thermal_limits=thermal.provider,
    )
    watchdog = _AdmissionOwnershipWatchdog(lease_reader)
    watchdog.start()
    started = time.monotonic_ns()
    deadline = started + (
        MAXIMUM_ADMISSION_ACQUISITION_SECONDS * NANOSECONDS_PER_SECOND
    )
    observations: list[dict[str, Any]] = []
    try:
        for index in range(REQUIRED_IDLE_OBSERVATIONS):
            scheduled = started + index * NANOSECONDS_PER_SECOND
            if not _real_sleep_until(scheduled, deadline):
                break
            try:
                probe_started = time.monotonic_ns()
                if probe_started >= scheduled + NANOSECONDS_PER_SECOND:
                    break
                reading = probe()
                probe_finished = time.monotonic_ns()
                watchdog_state = watchdog.snapshot()
                observed = time.monotonic_ns()
                if observed >= scheduled + NANOSECONDS_PER_SECOND:
                    break
                observations.append(
                    construct_admission_observation(
                        sequence=index,
                        admission_context_sha256=context.sha256,
                        current_authority_sha256=authority_sha256,
                        scheduled_slot=index,
                        scheduled_monotonic_ns=scheduled,
                        observed_monotonic_ns=observed,
                        wall_time_utc=utc_now(),
                        probe_reading=reading,
                        collector={
                            "healthy": True,
                            "status_code": None,
                            "probe_duration_ns": probe_finished - probe_started,
                        },
                        watchdog=watchdog_state,
                    )
                )
            except (ProbeFailure, AdmissionError):
                break
    finally:
        watchdog.stop()
    finished = time.monotonic_ns()
    final_authority, final_authority_sha256 = authority_snapshot(authority)
    if (
        final_authority != initial_authority
        or final_authority_sha256 != authority_sha256
    ):
        raise AdmissionError("production authority changed during collection")
    return observations, started, finished, authority


def _bind_production_observation_collector(
    collector: Callable[
        ..., tuple[list[dict[str, Any]], int, int, Phase4CurrentAuthority]
    ],
) -> tuple[Callable[..., _ProductionAdmissionObservationBatch], Callable[..., bool]]:
    """Keep the production capability registry outside importable globals."""

    authentic: weakref.WeakSet[_ProductionAdmissionObservationBatch] = weakref.WeakSet()

    def collect(
        context: PlannedSlotContext,
        *,
        authority: Phase4CurrentAuthority,
        filesystem_path: Path,
        job_service: Any,
        gpu_index: int = 0,
        nvidia_smi_path: str | None = None,
    ) -> _ProductionAdmissionObservationBatch:
        observations, started, finished, collected_authority = collector(
            context,
            authority=authority,
            filesystem_path=filesystem_path,
            job_service=job_service,
            gpu_index=gpu_index,
            nvidia_smi_path=nvidia_smi_path,
        )
        batch = object.__new__(_ProductionAdmissionObservationBatch)
        batch._initialize_from_collector(
            observations=observations,
            acquisition_started_monotonic_ns=started,
            acquisition_finished_monotonic_ns=finished,
            authority=collected_authority,
        )
        authentic.add(batch)
        return batch

    def is_authentic(value: object, authority: CurrentAuthority | None = None) -> bool:
        return bool(
            type(value) is _ProductionAdmissionObservationBatch
            and value in authentic
            and (authority is None or value._authority is authority)
        )

    return collect, is_authentic


(
    collect_production_admission_observations,
    _production_batch_is_authentic,
) = _bind_production_observation_collector(_collect_production_admission_payload)
del _bind_production_observation_collector


def authority_snapshot(authority: CurrentAuthority) -> tuple[dict[str, Any], str]:
    try:
        snapshot = _validate_phase4_binding(dict(authority.snapshot()))
    except (TypeError, ValueError) as error:
        raise AdmissionError("current Phase-4 authority is invalid") from error
    return snapshot, sha256_bytes(compact_canonical_json_bytes(snapshot))


def construct_admission_observation(
    *,
    sequence: int,
    admission_context_sha256: str,
    current_authority_sha256: str,
    scheduled_slot: int,
    scheduled_monotonic_ns: int,
    observed_monotonic_ns: int,
    wall_time_utc: str,
    probe_reading: Mapping[str, Any],
    collector: Mapping[str, Any],
    watchdog: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one strict admission observation without an experiment-run ID."""

    sequence = _nonnegative_integer(sequence, "sequence")
    scheduled_slot = _nonnegative_integer(scheduled_slot, "scheduled_slot")
    scheduled = _nonnegative_integer(scheduled_monotonic_ns, "scheduled_monotonic_ns")
    observed = _nonnegative_integer(observed_monotonic_ns, "observed_monotonic_ns")
    if observed < scheduled:
        raise AdmissionError("observation precedes its scheduled time")
    normalized_wall_time = _normalized_utc_timestamp(wall_time_utc, "wall_time_utc")
    if not isinstance(probe_reading, Mapping) or set(probe_reading) != {"gpu", "host"}:
        raise AdmissionError("probe reading has the wrong exact fields")
    try:
        result = {
            "schema_version": ADMISSION_OBSERVATION_SCHEMA,
            "sequence": sequence,
            "admission_context_sha256": _digest(
                admission_context_sha256, "admission_context_sha256"
            ),
            "current_authority_sha256": _digest(
                current_authority_sha256, "current_authority_sha256"
            ),
            "scheduled_slot": scheduled_slot,
            "scheduled_monotonic_ns": scheduled,
            "observed_monotonic_ns": observed,
            "wall_time_utc": normalized_wall_time,
            "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            **normalize_observation_facts(
                probe_reading=probe_reading,
                collector=collector,
                watchdog=watchdog,
                observed_monotonic_ns=observed,
            ),
        }
    except TelemetryValidationError as error:
        raise AdmissionError("admission observation probe facts are invalid") from error
    return validate_admission_observation(result)


def validate_admission_observation(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _OBSERVATION_FIELDS:
        raise AdmissionError("admission observation has the wrong exact fields")
    if value["schema_version"] != ADMISSION_OBSERVATION_SCHEMA:
        raise AdmissionError("admission observation schema is unsupported")
    sequence = _nonnegative_integer(value["sequence"], "sequence")
    slot = _nonnegative_integer(value["scheduled_slot"], "scheduled_slot")
    scheduled = _nonnegative_integer(
        value["scheduled_monotonic_ns"], "scheduled_monotonic_ns"
    )
    observed = _nonnegative_integer(
        value["observed_monotonic_ns"], "observed_monotonic_ns"
    )
    if observed < scheduled or value["sample_interval_seconds"] != 1:
        raise AdmissionError("admission observation cadence is invalid")
    _digest(value["admission_context_sha256"], "admission context")
    _digest(value["current_authority_sha256"], "current authority")
    _normalized_utc_timestamp(value["wall_time_utc"], "admission wall time")
    try:
        validate_observation_facts(
            {name: value[name] for name in ("gpu", "host", "collector", "watchdog")},
            observed_monotonic_ns=observed,
        )
    except TelemetryValidationError as error:
        raise AdmissionError("admission observation facts are invalid") from error
    if sequence < 0 or slot < 0:  # make type narrowing explicit
        raise AssertionError("unreachable")
    return json.loads(compact_canonical_json_bytes(value))


def _validate_idle_gate(
    observations: Sequence[Mapping[str, Any]],
    *,
    acquisition_started_monotonic_ns: int,
    acquisition_finished_monotonic_ns: int,
    context_sha256: str,
    authority_sha256: str,
    baseline: Mapping[str, Any],
    budget: FrozenResourceBudget,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Apply the frozen cooldown rules plus exact plan-specific resources."""

    if len(observations) != REQUIRED_IDLE_OBSERVATIONS:
        return ("MISSING_REQUIRED_EVIDENCE",), {"sample_count": len(observations)}
    try:
        valid = [validate_admission_observation(dict(item)) for item in observations]
    except (TypeError, ValueError):
        return ("MISSING_REQUIRED_EVIDENCE",), {"sample_count": len(observations)}
    reasons: list[str] = []
    if (
        [item["sequence"] for item in valid] != list(range(REQUIRED_IDLE_OBSERVATIONS))
        or [item["scheduled_slot"] for item in valid]
        != list(range(REQUIRED_IDLE_OBSERVATIONS))
        or any(
            item["scheduled_monotonic_ns"]
            != acquisition_started_monotonic_ns + index * NANOSECONDS_PER_SECOND
            or item["observed_monotonic_ns"] < item["scheduled_monotonic_ns"]
            or item["observed_monotonic_ns"]
            >= item["scheduled_monotonic_ns"] + NANOSECONDS_PER_SECOND
            or item["observed_monotonic_ns"] > acquisition_finished_monotonic_ns
            for index, item in enumerate(valid)
        )
    ):
        reasons.append("MISSING_REQUIRED_EVIDENCE")
    if any(
        item["admission_context_sha256"] != context_sha256
        or item["current_authority_sha256"] != authority_sha256
        for item in valid
    ):
        reasons.append("MISSING_REQUIRED_EVIDENCE")
    evaluated = validate_cooldown_observations(
        valid,
        baseline,
        required_samples=REQUIRED_IDLE_OBSERVATIONS,
        minimum_zero_utilization_samples=MINIMUM_ZERO_UTILIZATION_OBSERVATIONS,
        required_host_ram_bytes=budget.admission_host_ram_bytes,
        required_disk_bytes=budget.admission_disk_bytes,
        disk_reason_code="DISK_BUDGET_INSUFFICIENT",
        power_channel_supported=baseline["gpu_power_draw_p95_w"] is not None,
    )
    reasons.extend(evaluated.reason_codes)
    summary = {
        **dict(evaluated.summary),
        "minimum_mem_available_bytes": min(
            item["host"]["mem_available_bytes"] for item in valid
        ),
        "minimum_filesystem_free_bytes": min(
            item["host"]["filesystem_free_bytes"] for item in valid
        ),
    }
    return tuple(dict.fromkeys(reasons)), summary


@dataclass(frozen=True, eq=False)
class AdmissionResult:
    """Idle-gate result whose production authority is intentionally ephemeral.

    The serialized decision booleans describe how the decision was produced;
    they are never a capability.  Only the exact result registered by
    :func:`evaluate_pre_slot_admission` after consuming an authentic production
    observation batch can authorize production activation.
    """

    context: PlannedSlotContext
    observations: tuple[Mapping[str, Any], ...]
    decision: Mapping[str, Any]
    decision_sha256: str
    authority_snapshot: Mapping[str, Any]

    @property
    def admitted(self) -> bool:
        return bool(self.decision["admitted"])

    @property
    def execution_configuration_id(self) -> None:
        return None

    @property
    def experiment_run_id(self) -> None:
        return None

    @property
    def artifact_id(self) -> None:
        return None

    def authorized_for_production_activation(self, authority: CurrentAuthority) -> bool:
        """Return the closure-held authority for this exact transient result."""

        return _admission_result_is_authentic(self, authority)


def evaluate_pre_slot_admission(
    context: PlannedSlotContext,
    observations: Sequence[Mapping[str, Any]] | _ProductionAdmissionObservationBatch,
    *,
    authority: CurrentAuthority,
    acquisition_started_monotonic_ns: int | None = None,
    acquisition_finished_monotonic_ns: int | None = None,
) -> AdmissionResult:
    """Evaluate an idle gate without assigning any started-run identity."""

    production_observations = False
    if type(observations) is _ProductionAdmissionObservationBatch:
        batch = observations
        if not batch.authorized_for(authority):
            raise AdmissionError("production observations have the wrong authority")
        if (
            acquisition_started_monotonic_ns is not None
            and acquisition_started_monotonic_ns
            != batch.acquisition_started_monotonic_ns
        ) or (
            acquisition_finished_monotonic_ns is not None
            and acquisition_finished_monotonic_ns
            != batch.acquisition_finished_monotonic_ns
        ):
            raise AdmissionError("caller time differs from production acquisition")
        acquisition_started_monotonic_ns = batch.acquisition_started_monotonic_ns
        acquisition_finished_monotonic_ns = batch.acquisition_finished_monotonic_ns
        raw_observations: Sequence[Mapping[str, Any]] = batch.observations
        production_observations = type(authority) is Phase4CurrentAuthority
    else:
        raw_observations = observations
    if (
        acquisition_started_monotonic_ns is None
        or acquisition_finished_monotonic_ns is None
    ):
        raise AdmissionError("admission acquisition times are required")

    started = _nonnegative_integer(
        acquisition_started_monotonic_ns, "acquisition_started_monotonic_ns"
    )
    finished = _nonnegative_integer(
        acquisition_finished_monotonic_ns, "acquisition_finished_monotonic_ns"
    )
    if finished < started:
        raise AdmissionError("admission acquisition time moved backward")
    snapshot, authority_sha256 = authority_snapshot(authority)
    reasons: list[str] = []
    if snapshot != dict(context.phase4_binding):
        reasons.append("MISSING_REQUIRED_EVIDENCE")
    elapsed_ns = finished - started
    if elapsed_ns > MAXIMUM_ADMISSION_ACQUISITION_SECONDS * NANOSECONDS_PER_SECOND:
        reasons.append("EMERGENCY_DEADLINE_EXCEEDED")
    gate_reasons, summary = _validate_idle_gate(
        raw_observations,
        acquisition_started_monotonic_ns=started,
        acquisition_finished_monotonic_ns=finished,
        context_sha256=context.sha256,
        authority_sha256=authority_sha256,
        baseline=context.phase4_binding["summary"],
        budget=context.resource_budget,
    )
    reasons.extend(gate_reasons)
    reasons = list(dict.fromkeys(reasons))
    if any(code not in REASON_CODES or code == "NONE" for code in reasons):
        raise AdmissionError("admission decision produced a non-protocol reason code")
    validated_observations: list[dict[str, Any]] = []
    for item in raw_observations:
        try:
            validated_observations.append(validate_admission_observation(dict(item)))
        except (TypeError, ValueError):
            validated_observations = []
            break
    observation_payload = compact_canonical_json_bytes(validated_observations)
    record = {
        "schema_version": ADMISSION_DECISION_SCHEMA,
        "admitted": not reasons,
        "reason_codes": reasons,
        "planned_attempt_slot_sha256": sha256_bytes(
            canonical_json_bytes(dict(context.planned_attempt_slot))
        ),
        "admission_context_sha256": context.sha256,
        "resource_budget_sha256": context.resource_budget.sha256,
        "current_authority_sha256": authority_sha256,
        "observations_sha256": sha256_bytes(observation_payload),
        "observation_count": len(raw_observations),
        "acquisition_started_monotonic_ns": started,
        "acquisition_finished_monotonic_ns": finished,
        "acquisition_duration_ns": elapsed_ns,
        "summary": summary,
        "production_authority": bool(
            type(authority) is Phase4CurrentAuthority
            and authority.production_qualifying
        ),
        "production_observations": production_observations,
    }
    decision_sha256 = sha256_bytes(compact_canonical_json_bytes(record))
    frozen_observations = tuple(
        MappingProxyType(item) for item in validated_observations
    )
    return AdmissionResult(
        context=context,
        observations=frozen_observations,
        decision=MappingProxyType(record),
        decision_sha256=decision_sha256,
        authority_snapshot=MappingProxyType(snapshot),
    )


def _bind_admission_result_authority(
    evaluator: Callable[..., AdmissionResult],
) -> tuple[
    Callable[..., AdmissionResult],
    Callable[[object, CurrentAuthority | None], bool],
]:
    """Brand only results derived from an authentic production batch.

    The registry is process-local by design.  Activation persistence never
    serializes this authority; a restarted consumer must instead use the
    independent pinned ``verify_activated_slot`` capability boundary.
    """

    registry: weakref.WeakKeyDictionary[AdmissionResult, tuple[object, ...]] = (
        weakref.WeakKeyDictionary()
    )
    lock = threading.Lock()

    def evaluate(
        context: PlannedSlotContext,
        observations: Sequence[Mapping[str, Any]]
        | _ProductionAdmissionObservationBatch,
        *,
        authority: CurrentAuthority,
        acquisition_started_monotonic_ns: int | None = None,
        acquisition_finished_monotonic_ns: int | None = None,
    ) -> AdmissionResult:
        authentic_batch = bool(
            type(observations) is _ProductionAdmissionObservationBatch
            and _production_batch_is_authentic(observations, authority)
            and type(authority) is Phase4CurrentAuthority
        )
        result = evaluator(
            context,
            observations,
            authority=authority,
            acquisition_started_monotonic_ns=acquisition_started_monotonic_ns,
            acquisition_finished_monotonic_ns=acquisition_finished_monotonic_ns,
        )
        if authentic_batch:
            snapshot = (
                authority,
                result.context,
                result.observations,
                result.decision,
                result.decision_sha256,
                result.authority_snapshot,
            )
            with lock:
                registry[result] = snapshot
        return result

    def is_authentic(value: object, authority: CurrentAuthority | None = None) -> bool:
        if type(value) is not AdmissionResult:
            return False
        with lock:
            expected = registry.get(value)
        if expected is None:
            return False
        observed = (
            authority,
            value.context,
            value.observations,
            value.decision,
            value.decision_sha256,
            value.authority_snapshot,
        )
        return bool(
            authority is expected[0]
            and all(observed[index] is expected[index] for index in (1, 2, 3, 5))
            and observed[4] == expected[4]
            and type(authority) is Phase4CurrentAuthority
            and authority.production_qualifying
            and value.decision.get("production_authority") is True
            and value.decision.get("production_observations") is True
        )

    return evaluate, is_authentic


_original_admission_evaluator = evaluate_pre_slot_admission
(
    evaluate_pre_slot_admission,
    _admission_result_is_authentic,
) = _bind_admission_result_authority(_original_admission_evaluator)
del _bind_admission_result_authority, _original_admission_evaluator


class ActivationClock(Protocol):
    def monotonic_ns(self) -> int: ...

    def wall_time_utc(self) -> str: ...


@dataclass(frozen=True)
class InjectedActivationClock:
    monotonic_value: int
    wall_time_value: str

    def monotonic_ns(self) -> int:
        return self.monotonic_value

    def wall_time_utc(self) -> str:
        return self.wall_time_value


class _SystemActivationClock:
    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def wall_time_utc(self) -> str:
        return utc_now()


_SYSTEM_CLOCK = _SystemActivationClock()


@dataclass(frozen=True)
class ActivatedSlot:
    directory: Path
    admission_decision: Mapping[str, Any]
    activation_decision: Mapping[str, Any]
    execution_configuration: Mapping[str, Any]
    experiment_run_template: Mapping[str, Any]
    started_identity_template: Mapping[str, Any]
    production_qualifying: bool

    @property
    def execution_configuration_id(self) -> str:
        return str(self.execution_configuration["execution_configuration_id"])

    @property
    def experiment_run_id(self) -> str:
        return str(self.experiment_run_template["experiment_run_id"])


class VerifiedActivatedSlot:
    """Deep-verified, production-qualifying activation consumed by the harness."""

    __slots__ = (
        "directory",
        "_admission_decision_payload",
        "_activation_decision_payload",
        "_execution_configuration_payload",
        "_experiment_run_template_payload",
        "_started_identity_template_payload",
        "_observations_payload",
        "_sealed",
        "__weakref__",
    )

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("Use verify_activated_slot().")

    def _initialize_from_verifier(
        self,
        *,
        directory: Path,
        admission_decision: Mapping[str, Any],
        activation_decision: Mapping[str, Any],
        execution_configuration: Mapping[str, Any],
        experiment_run_template: Mapping[str, Any],
        started_identity_template: Mapping[str, Any],
        observations: tuple[Mapping[str, Any], ...],
    ) -> None:
        object.__setattr__(self, "directory", directory)
        for name, value in (
            ("_admission_decision_payload", admission_decision),
            ("_activation_decision_payload", activation_decision),
            ("_execution_configuration_payload", execution_configuration),
            ("_experiment_run_template_payload", experiment_run_template),
            ("_started_identity_template_payload", started_identity_template),
        ):
            object.__setattr__(self, name, compact_canonical_json_bytes(dict(value)))
        object.__setattr__(
            self,
            "_observations_payload",
            compact_canonical_json_bytes([dict(item) for item in observations]),
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("verified activation is immutable")
        object.__setattr__(self, name, value)

    def _mapping(self, name: str) -> Mapping[str, Any]:
        value = json.loads(getattr(self, name))
        return MappingProxyType(value)

    @property
    def admission_decision(self) -> Mapping[str, Any]:
        return self._mapping("_admission_decision_payload")

    @property
    def activation_decision(self) -> Mapping[str, Any]:
        return self._mapping("_activation_decision_payload")

    @property
    def execution_configuration(self) -> Mapping[str, Any]:
        return self._mapping("_execution_configuration_payload")

    @property
    def experiment_run_template(self) -> Mapping[str, Any]:
        return self._mapping("_experiment_run_template_payload")

    @property
    def started_identity_template(self) -> Mapping[str, Any]:
        return self._mapping("_started_identity_template_payload")

    @property
    def observations(self) -> tuple[Mapping[str, Any], ...]:
        values = json.loads(self._observations_payload)
        return tuple(MappingProxyType(item) for item in values)

    @property
    def production_qualifying(self) -> bool:
        return _verified_activation_is_authentic(self)

    def authorized_for_qualifying_harness(self) -> bool:
        """Return the private capability check used by the harness factory."""

        return _verified_activation_is_authentic(self)

    @property
    def execution_configuration_id(self) -> str:
        return str(self.execution_configuration["execution_configuration_id"])

    @property
    def experiment_run_id(self) -> str:
        return str(self.experiment_run_template["experiment_run_id"])


def _metadata_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_pinned_activation_file(
    directory_fd: int, name: str, *, maximum_bytes: int
) -> bytes:
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise AdmissionError("activation file is unavailable") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
        or before.st_size > maximum_bytes
    ):
        raise AdmissionError("activation file metadata is unsafe")
    try:
        fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise AdmissionError("activation file cannot be pinned") from error
    try:
        opened = os.fstat(fd)
        if _metadata_fingerprint(opened) != _metadata_fingerprint(before):
            raise AdmissionError("activation file changed while being pinned")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > maximum_bytes or len(payload) != before.st_size:
            raise AdmissionError("activation file size is invalid")
        after = os.fstat(fd)
        path_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _metadata_fingerprint(after) != _metadata_fingerprint(
            opened
        ) or _metadata_fingerprint(path_after) != _metadata_fingerprint(opened):
            raise AdmissionError("activation file changed during verification")
        return payload
    finally:
        os.close(fd)


def _read_activation_payloads(directory: Path) -> dict[str, bytes]:
    metadata = _private_directory_metadata(directory)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(directory, flags)
    except OSError as error:
        raise AdmissionError("activation directory cannot be pinned") from error
    try:
        opened = os.fstat(directory_fd)
        if _metadata_fingerprint(opened) != _metadata_fingerprint(metadata):
            raise AdmissionError("activation directory changed while being pinned")
        expected = {*_ACTIVATION_FILE_NAMES, ACTIVATION_SEAL_NAME}
        if set(os.listdir(directory_fd)) != expected:
            raise AdmissionError("activation file inventory is not exact")
        payloads: dict[str, bytes] = {}
        for name in (*_ACTIVATION_FILE_NAMES, ACTIVATION_SEAL_NAME):
            maximum = (
                64 * 1024 * 1024
                if name == "admission-observations.json"
                else 2 * 1024 * 1024
            )
            payloads[name] = _read_pinned_activation_file(
                directory_fd, name, maximum_bytes=maximum
            )
        if set(os.listdir(directory_fd)) != expected:
            raise AdmissionError("activation inventory changed during verification")
        current = directory.lstat()
        if _metadata_fingerprint(current) != _metadata_fingerprint(opened):
            raise AdmissionError("activation directory changed during verification")
        return payloads
    finally:
        os.close(directory_fd)


def _canonical_activation_json(payload: bytes, label: str) -> Any:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdmissionError(f"{label} is invalid JSON") from error
    if canonical_json_bytes(value) != payload:
        raise AdmissionError(f"{label} is not canonical JSON")
    return value


def _verify_activated_slot_payload(
    directory: Path,
    *,
    expected_context: PlannedSlotContext,
    authority: CurrentAuthority,
) -> dict[str, Any]:
    """Deep-verify a sealed activation and require production authority.

    Consumers must call this immediately before constructing the qualifying
    harness.  The returned object is derived from pinned bytes, never from the
    in-memory object returned by :func:`activate_admitted_slot`.
    """

    payloads = _read_activation_payloads(directory)
    values = {
        name: _canonical_activation_json(payload, name)
        for name, payload in payloads.items()
    }
    seal = values[ACTIVATION_SEAL_NAME]
    if (
        type(seal) is not dict
        or set(seal) != {"schema_version", "activation_decision_sha256", "files"}
        or seal["schema_version"] != ACTIVATION_SEAL_SCHEMA
        or type(seal["files"]) is not list
        or len(seal["files"]) != len(_ACTIVATION_FILE_NAMES)
    ):
        raise AdmissionError("activation seal fields are invalid")
    expected_entries = []
    for name in _ACTIVATION_FILE_NAMES:
        expected_entries.append(
            {
                "name": name,
                "size_bytes": len(payloads[name]),
                "sha256": sha256_bytes(payloads[name]),
            }
        )
    if seal["files"] != expected_entries or seal[
        "activation_decision_sha256"
    ] != sha256_bytes(payloads["activation-decision.json"]):
        raise AdmissionError("activation seal does not bind the exact files")

    decision = values["admission-decision.json"]
    observations = values["admission-observations.json"]
    execution = values["execution-configuration.json"]
    run = values["experiment-run-template.json"]
    started = values["started-identity-template.json"]
    activation = values["activation-decision.json"]
    if type(decision) is not dict or set(decision) != _ADMISSION_DECISION_FIELDS:
        raise AdmissionError("admission decision fields are invalid")
    if (
        decision["schema_version"] != ADMISSION_DECISION_SCHEMA
        or decision["admitted"] is not True
        or decision["reason_codes"] != []
        or decision["production_authority"] is not True
        or decision["production_observations"] is not True
    ):
        raise AdmissionError("admission decision is not production-admitted")
    if (
        type(observations) is not list
        or len(observations) != REQUIRED_IDLE_OBSERVATIONS
    ):
        raise AdmissionError("activation lacks the exact admission observations")
    validated_observations = tuple(
        validate_admission_observation(item) for item in observations
    )
    if decision["observations_sha256"] != sha256_bytes(
        compact_canonical_json_bytes(observations)
    ) or decision["observation_count"] != len(observations):
        raise AdmissionError("admission decision does not bind its observations")
    started_ns = _nonnegative_integer(
        decision["acquisition_started_monotonic_ns"],
        "acquisition_started_monotonic_ns",
    )
    finished_ns = _nonnegative_integer(
        decision["acquisition_finished_monotonic_ns"],
        "acquisition_finished_monotonic_ns",
    )
    if (
        finished_ns < started_ns
        or decision["acquisition_duration_ns"] != finished_ns - started_ns
        or finished_ns - started_ns
        > MAXIMUM_ADMISSION_ACQUISITION_SECONDS * NANOSECONDS_PER_SECOND
    ):
        raise AdmissionError("admission acquisition timing is invalid")
    context_digest = expected_context.sha256
    planned_digest = sha256_bytes(
        canonical_json_bytes(dict(expected_context.planned_attempt_slot))
    )
    if (
        decision["admission_context_sha256"] != context_digest
        or decision["planned_attempt_slot_sha256"] != planned_digest
        or decision["resource_budget_sha256"] != expected_context.resource_budget.sha256
    ):
        raise AdmissionError("admission decision differs from the expected context")
    snapshot, authority_digest = authority_snapshot(authority)
    if (
        type(authority) is not Phase4CurrentAuthority
        or snapshot != dict(expected_context.phase4_binding)
        or decision["current_authority_sha256"] != authority_digest
    ):
        raise AdmissionError("activation lacks current production authority")
    gate_reasons, expected_summary = _validate_idle_gate(
        validated_observations,
        acquisition_started_monotonic_ns=started_ns,
        acquisition_finished_monotonic_ns=finished_ns,
        context_sha256=context_digest,
        authority_sha256=authority_digest,
        baseline=expected_context.phase4_binding["summary"],
        budget=expected_context.resource_budget,
    )
    if gate_reasons or decision["summary"] != expected_summary:
        raise AdmissionError("admission decision does not reproduce")

    try:
        execution_record = validate_record(
            execution, SCHEMA_VERSIONS["execution_configuration"]
        )
        run_record = validate_record(run, SCHEMA_VERSIONS["experiment_run"])
    except (TypeError, ValueError) as error:
        raise AdmissionError("activated Phase-1 identity record is invalid") from error
    proposal = expected_context.execution_proposal
    expected_execution = {
        "schema_version": SCHEMA_VERSIONS["execution_configuration"],
        "execution_configuration_id": execution_record["execution_configuration_id"],
        "comparison_cell_id": expected_context.comparison_cell["comparison_cell_id"],
        "exact_behavior_values": dict(proposal.exact_behavior_values),
        "split_seed": proposal.split_seed,
        "training_seed": proposal.training_seed,
        "data_order_seed": proposal.data_order_seed,
        "plan_id": proposal.plan_id,
        "candidate_id": proposal.candidate_id,
        "bundle_fingerprint": proposal.bundle_fingerprint,
        "emergency_deadline_seconds": proposal.emergency_deadline_seconds,
    }
    if execution_record != expected_execution:
        raise AdmissionError("execution configuration differs from its proposal")
    experiment_run_id = run_record["experiment_run_id"]
    if _EXPERIMENT_RUN_ID.fullmatch(experiment_run_id) is None:
        raise AdmissionError("activated experiment run ID is invalid")
    run_proposal = expected_context.run_proposal
    decision_digest = sha256_bytes(compact_canonical_json_bytes(decision))
    expected_run = {
        "schema_version": SCHEMA_VERSIONS["experiment_run"],
        "experiment_run_id": experiment_run_id,
        "attempt_slot_id": expected_context.planned_attempt_slot["attempt_slot_id"],
        "execution_configuration_id": execution_record["execution_configuration_id"],
        "exact_argv": list(_PENDING_ARGV),
        "working_directory": run_proposal.working_directory,
        "fresh_state_root": run_proposal.fresh_state_root,
        "bundle_path": run_proposal.bundle_path,
        "output_path": run_proposal.output_path,
        "run_order": {
            "block": expected_context.planned_attempt_slot["block"],
            "position": expected_context.planned_attempt_slot["order_position"],
        },
        "observed_host_state": {
            "idle_baseline_sha256": sha256_bytes(
                canonical_json_bytes(dict(expected_context.phase4_binding))
            ),
            "admission_decision_sha256": decision_digest,
            "current_authority_sha256": authority_digest,
            "resource_budget_sha256": expected_context.resource_budget.sha256,
        },
        "plan_id": proposal.plan_id,
        "candidate_id": proposal.candidate_id,
        "bundle_fingerprint": proposal.bundle_fingerprint,
        "bundle_manifest_sha256": run_proposal.bundle_manifest_sha256,
        "archive_sha256": run_proposal.archive_sha256,
        "aptus_job_ids": [],
        "aptus_run_ids": [],
        "terminal_evidence": {"status": "pending"},
    }
    if run_record != expected_run:
        raise AdmissionError("experiment run template differs from its proposal")
    if type(started) is not dict or set(started) != _STARTED_TEMPLATE_FIELDS:
        raise AdmissionError("started identity template fields are invalid")
    expected_started = {
        "schema_version": STARTED_IDENTITY_TEMPLATE_SCHEMA,
        "attempt_slot_id": expected_context.planned_attempt_slot["attempt_slot_id"],
        "planned_attempt_slot_sha256": planned_digest,
        "execution_configuration_id": execution_record["execution_configuration_id"],
        "experiment_run_id": experiment_run_id,
        "terminal_record_status": "pending",
    }
    if started != expected_started:
        raise AdmissionError("started identity template is misbound")
    if type(activation) is not dict or set(activation) != _ACTIVATION_DECISION_FIELDS:
        raise AdmissionError("activation decision fields are invalid")
    if (
        activation["schema_version"] != ACTIVATION_DECISION_SCHEMA
        or activation["activation_status"] != "activated"
        or activation["production_qualifying"] is not True
        or activation["admission_decision_sha256"] != decision_digest
        or activation["admission_context_sha256"] != context_digest
        or activation["resource_budget_sha256"]
        != expected_context.resource_budget.sha256
        or activation["current_authority_sha256"] != authority_digest
        or activation["execution_configuration_sha256"]
        != sha256_bytes(canonical_json_bytes(execution_record))
        or activation["experiment_run_template_sha256"]
        != sha256_bytes(canonical_json_bytes(run_record))
        or activation["started_identity_template_sha256"]
        != sha256_bytes(canonical_json_bytes(started))
        or activation["execution_configuration_id"]
        != execution_record["execution_configuration_id"]
        or activation["experiment_run_id"] != experiment_run_id
    ):
        raise AdmissionError("activation decision is not production-bound")
    verified_activation_ns = _nonnegative_integer(
        activation["activated_monotonic_ns"], "activated_monotonic_ns"
    )
    _normalized_utc_timestamp(activation["activated_at_utc"], "activated_at_utc")
    if verified_activation_ns < finished_ns:
        raise AdmissionError("activation wall time is invalid")
    return {
        "directory": directory,
        "admission_decision": decision,
        "activation_decision": activation,
        "execution_configuration": execution_record,
        "experiment_run_template": run_record,
        "started_identity_template": started,
        "observations": validated_observations,
    }


def _bind_verified_activation_verifier(
    verifier: Callable[..., dict[str, Any]],
) -> tuple[Callable[..., VerifiedActivatedSlot], Callable[[object], bool]]:
    """Brand only results produced by the complete pinned verifier."""

    authentic: weakref.WeakSet[VerifiedActivatedSlot] = weakref.WeakSet()

    def verify(
        directory: Path,
        *,
        expected_context: PlannedSlotContext,
        authority: CurrentAuthority,
    ) -> VerifiedActivatedSlot:
        payload = verifier(
            directory,
            expected_context=expected_context,
            authority=authority,
        )
        verified = object.__new__(VerifiedActivatedSlot)
        verified._initialize_from_verifier(**payload)
        authentic.add(verified)
        return verified

    def is_authentic(value: object) -> bool:
        return bool(type(value) is VerifiedActivatedSlot and value in authentic)

    return verify, is_authentic


(
    verify_activated_slot,
    _verified_activation_is_authentic,
) = _bind_verified_activation_verifier(_verify_activated_slot_payload)
del _bind_verified_activation_verifier


@dataclass(frozen=True)
class RetainedActivatedSlot:
    """Deep-validated activation provenance without a current-host capability."""

    planned_slot_context: PlannedSlotContext
    admission_decision: Mapping[str, Any]
    observations: tuple[Mapping[str, Any], ...]
    execution_configuration: Mapping[str, Any]
    experiment_run_template: Mapping[str, Any]
    started_identity_template: Mapping[str, Any]
    activation_decision: Mapping[str, Any]


def validate_retained_activated_slot(
    payloads: Mapping[str, bytes],
    *,
    planned_slot_context_record: Any,
) -> RetainedActivatedSlot:
    """Verify all seven retained activation files and their complete identity chain."""

    expected_names = {*_ACTIVATION_FILE_NAMES, ACTIVATION_SEAL_NAME}
    if type(payloads) is not dict or set(payloads) != expected_names:
        raise AdmissionError("retained activation file inventory is not exact")
    if any(type(payload) is not bytes for payload in payloads.values()):
        raise AdmissionError("retained activation payloads must be exact bytes")
    values = {
        name: _canonical_activation_json(payloads[name], name)
        for name in expected_names
    }
    seal = values[ACTIVATION_SEAL_NAME]
    expected_entries = [
        {
            "name": name,
            "size_bytes": len(payloads[name]),
            "sha256": sha256_bytes(payloads[name]),
        }
        for name in _ACTIVATION_FILE_NAMES
    ]
    if (
        type(seal) is not dict
        or set(seal) != {"schema_version", "activation_decision_sha256", "files"}
        or seal["schema_version"] != ACTIVATION_SEAL_SCHEMA
        or seal["files"] != expected_entries
        or seal["activation_decision_sha256"]
        != sha256_bytes(payloads["activation-decision.json"])
    ):
        raise AdmissionError("retained activation seal is invalid")

    context = planned_slot_context_from_record(planned_slot_context_record)
    decision = values["admission-decision.json"]
    observations = values["admission-observations.json"]
    execution = values["execution-configuration.json"]
    run = values["experiment-run-template.json"]
    started = values["started-identity-template.json"]
    activation = values["activation-decision.json"]
    if (
        type(decision) is not dict
        or set(decision) != _ADMISSION_DECISION_FIELDS
        or decision["schema_version"] != ADMISSION_DECISION_SCHEMA
        or decision["admitted"] is not True
        or decision["reason_codes"] != []
        or decision["production_authority"] is not True
        or decision["production_observations"] is not True
        or type(observations) is not list
        or len(observations) != REQUIRED_IDLE_OBSERVATIONS
    ):
        raise AdmissionError("retained admission is not production-admitted")
    validated_observations = tuple(
        validate_admission_observation(item) for item in observations
    )
    if decision["observations_sha256"] != sha256_bytes(
        compact_canonical_json_bytes(observations)
    ) or decision["observation_count"] != len(observations):
        raise AdmissionError("retained admission observations are misbound")
    started_ns = _nonnegative_integer(
        decision["acquisition_started_monotonic_ns"],
        "acquisition_started_monotonic_ns",
    )
    finished_ns = _nonnegative_integer(
        decision["acquisition_finished_monotonic_ns"],
        "acquisition_finished_monotonic_ns",
    )
    if (
        finished_ns < started_ns
        or decision["acquisition_duration_ns"] != finished_ns - started_ns
        or finished_ns - started_ns
        > MAXIMUM_ADMISSION_ACQUISITION_SECONDS * NANOSECONDS_PER_SECOND
    ):
        raise AdmissionError("retained admission acquisition timing is invalid")
    context_digest = context.sha256
    planned_digest = sha256_bytes(
        canonical_json_bytes(dict(context.planned_attempt_slot))
    )
    authority_digest = sha256_bytes(
        compact_canonical_json_bytes(dict(context.phase4_binding))
    )
    if (
        decision["admission_context_sha256"] != context_digest
        or decision["planned_attempt_slot_sha256"] != planned_digest
        or decision["resource_budget_sha256"] != context.resource_budget.sha256
        or decision["current_authority_sha256"] != authority_digest
    ):
        raise AdmissionError("retained admission differs from its planned context")
    gate_reasons, expected_summary = _validate_idle_gate(
        validated_observations,
        acquisition_started_monotonic_ns=started_ns,
        acquisition_finished_monotonic_ns=finished_ns,
        context_sha256=context_digest,
        authority_sha256=authority_digest,
        baseline=context.phase4_binding["summary"],
        budget=context.resource_budget,
    )
    if gate_reasons or decision["summary"] != expected_summary:
        raise AdmissionError("retained admission decision does not reproduce")

    try:
        execution_record = validate_record(
            execution, SCHEMA_VERSIONS["execution_configuration"]
        )
        run_record = validate_record(run, SCHEMA_VERSIONS["experiment_run"])
    except (TypeError, ValueError) as error:
        raise AdmissionError("retained activated identity is invalid") from error
    proposal = context.execution_proposal
    expected_execution = {
        "schema_version": SCHEMA_VERSIONS["execution_configuration"],
        "execution_configuration_id": execution_record["execution_configuration_id"],
        "comparison_cell_id": context.comparison_cell["comparison_cell_id"],
        "exact_behavior_values": dict(proposal.exact_behavior_values),
        "split_seed": proposal.split_seed,
        "training_seed": proposal.training_seed,
        "data_order_seed": proposal.data_order_seed,
        "plan_id": proposal.plan_id,
        "candidate_id": proposal.candidate_id,
        "bundle_fingerprint": proposal.bundle_fingerprint,
        "emergency_deadline_seconds": proposal.emergency_deadline_seconds,
    }
    if execution_record != expected_execution:
        raise AdmissionError("retained execution differs from its proposal")
    experiment_run_id = run_record["experiment_run_id"]
    if _EXPERIMENT_RUN_ID.fullmatch(experiment_run_id) is None:
        raise AdmissionError("retained experiment run ID is invalid")
    run_proposal = context.run_proposal
    decision_digest = sha256_bytes(compact_canonical_json_bytes(decision))
    expected_run = {
        "schema_version": SCHEMA_VERSIONS["experiment_run"],
        "experiment_run_id": experiment_run_id,
        "attempt_slot_id": context.planned_attempt_slot["attempt_slot_id"],
        "execution_configuration_id": execution_record["execution_configuration_id"],
        "exact_argv": list(_PENDING_ARGV),
        "working_directory": run_proposal.working_directory,
        "fresh_state_root": run_proposal.fresh_state_root,
        "bundle_path": run_proposal.bundle_path,
        "output_path": run_proposal.output_path,
        "run_order": {
            "block": context.planned_attempt_slot["block"],
            "position": context.planned_attempt_slot["order_position"],
        },
        "observed_host_state": {
            "idle_baseline_sha256": sha256_bytes(
                canonical_json_bytes(dict(context.phase4_binding))
            ),
            "admission_decision_sha256": decision_digest,
            "current_authority_sha256": authority_digest,
            "resource_budget_sha256": context.resource_budget.sha256,
        },
        "plan_id": proposal.plan_id,
        "candidate_id": proposal.candidate_id,
        "bundle_fingerprint": proposal.bundle_fingerprint,
        "bundle_manifest_sha256": run_proposal.bundle_manifest_sha256,
        "archive_sha256": run_proposal.archive_sha256,
        "aptus_job_ids": [],
        "aptus_run_ids": [],
        "terminal_evidence": {"status": "pending"},
    }
    if run_record != expected_run:
        raise AdmissionError("retained run template differs from its proposal")
    expected_started = {
        "schema_version": STARTED_IDENTITY_TEMPLATE_SCHEMA,
        "attempt_slot_id": context.planned_attempt_slot["attempt_slot_id"],
        "planned_attempt_slot_sha256": planned_digest,
        "execution_configuration_id": execution_record["execution_configuration_id"],
        "experiment_run_id": experiment_run_id,
        "terminal_record_status": "pending",
    }
    if type(started) is not dict or started != expected_started:
        raise AdmissionError("retained started identity template is misbound")
    if type(activation) is not dict or set(activation) != _ACTIVATION_DECISION_FIELDS:
        raise AdmissionError("retained activation decision fields are invalid")
    _normalized_utc_timestamp(
        activation["activated_at_utc"], "retained activated_at_utc"
    )
    if (
        activation["schema_version"] != ACTIVATION_DECISION_SCHEMA
        or activation["activation_status"] != "activated"
        or activation["production_qualifying"] is not True
        or activation["admission_decision_sha256"] != decision_digest
        or activation["admission_context_sha256"] != context_digest
        or activation["resource_budget_sha256"] != context.resource_budget.sha256
        or activation["current_authority_sha256"] != authority_digest
        or activation["execution_configuration_sha256"]
        != sha256_bytes(canonical_json_bytes(execution_record))
        or activation["experiment_run_template_sha256"]
        != sha256_bytes(canonical_json_bytes(run_record))
        or activation["started_identity_template_sha256"]
        != sha256_bytes(canonical_json_bytes(started))
        or activation["execution_configuration_id"]
        != execution_record["execution_configuration_id"]
        or activation["experiment_run_id"] != experiment_run_id
        or _nonnegative_integer(
            activation["activated_monotonic_ns"], "activated_monotonic_ns"
        )
        < finished_ns
    ):
        raise AdmissionError("retained activation decision is misbound")
    return RetainedActivatedSlot(
        planned_slot_context=context,
        admission_decision=MappingProxyType(decision),
        observations=tuple(MappingProxyType(item) for item in validated_observations),
        execution_configuration=MappingProxyType(execution_record),
        experiment_run_template=MappingProxyType(run_record),
        started_identity_template=MappingProxyType(started),
        activation_decision=MappingProxyType(activation),
    )


def _build_activation_records(
    result: AdmissionResult,
    *,
    authority_sha256: str,
    experiment_run_id: str,
    activated_monotonic_ns: int,
    activated_at_utc: str,
    production_qualifying: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    context = result.context
    proposal = context.execution_proposal
    execution_identity = {
        "schema_version": SCHEMA_VERSIONS["execution_configuration"],
        "comparison_cell_id": context.comparison_cell["comparison_cell_id"],
        "exact_behavior_values": dict(proposal.exact_behavior_values),
        "split_seed": proposal.split_seed,
        "training_seed": proposal.training_seed,
        "data_order_seed": proposal.data_order_seed,
        "plan_id": proposal.plan_id,
        "candidate_id": proposal.candidate_id,
        "bundle_fingerprint": proposal.bundle_fingerprint,
    }
    execution = dict(execution_identity)
    execution["execution_configuration_id"] = deterministic_id(
        "exec_", execution_identity
    )
    execution["emergency_deadline_seconds"] = proposal.emergency_deadline_seconds
    execution = validate_record(execution, SCHEMA_VERSIONS["execution_configuration"])
    decision_digest = sha256_bytes(compact_canonical_json_bytes(dict(result.decision)))
    run_proposal = context.run_proposal
    run = validate_record(
        {
            "schema_version": SCHEMA_VERSIONS["experiment_run"],
            "experiment_run_id": experiment_run_id,
            "attempt_slot_id": context.planned_attempt_slot["attempt_slot_id"],
            "execution_configuration_id": execution["execution_configuration_id"],
            "exact_argv": list(_PENDING_ARGV),
            "working_directory": run_proposal.working_directory,
            "fresh_state_root": run_proposal.fresh_state_root,
            "bundle_path": run_proposal.bundle_path,
            "output_path": run_proposal.output_path,
            "run_order": {
                "block": context.planned_attempt_slot["block"],
                "position": context.planned_attempt_slot["order_position"],
            },
            "observed_host_state": {
                "idle_baseline_sha256": sha256_bytes(
                    canonical_json_bytes(dict(context.phase4_binding))
                ),
                "admission_decision_sha256": decision_digest,
                "current_authority_sha256": authority_sha256,
                "resource_budget_sha256": context.resource_budget.sha256,
            },
            "plan_id": proposal.plan_id,
            "candidate_id": proposal.candidate_id,
            "bundle_fingerprint": proposal.bundle_fingerprint,
            "bundle_manifest_sha256": run_proposal.bundle_manifest_sha256,
            "archive_sha256": run_proposal.archive_sha256,
            "aptus_job_ids": [],
            "aptus_run_ids": [],
            "terminal_evidence": {"status": "pending"},
        },
        SCHEMA_VERSIONS["experiment_run"],
    )
    started = {
        "schema_version": STARTED_IDENTITY_TEMPLATE_SCHEMA,
        "attempt_slot_id": context.planned_attempt_slot["attempt_slot_id"],
        "planned_attempt_slot_sha256": result.decision["planned_attempt_slot_sha256"],
        "execution_configuration_id": execution["execution_configuration_id"],
        "experiment_run_id": experiment_run_id,
        "terminal_record_status": "pending",
    }
    activation = {
        "schema_version": ACTIVATION_DECISION_SCHEMA,
        "activation_status": "activated",
        "admission_decision_sha256": decision_digest,
        "admission_context_sha256": context.sha256,
        "resource_budget_sha256": context.resource_budget.sha256,
        "current_authority_sha256": authority_sha256,
        "execution_configuration_sha256": sha256_bytes(canonical_json_bytes(execution)),
        "experiment_run_template_sha256": sha256_bytes(canonical_json_bytes(run)),
        "started_identity_template_sha256": sha256_bytes(canonical_json_bytes(started)),
        "execution_configuration_id": execution["execution_configuration_id"],
        "experiment_run_id": experiment_run_id,
        "activated_monotonic_ns": activated_monotonic_ns,
        "activated_at_utc": activated_at_utc,
        "production_qualifying": production_qualifying,
    }
    return execution, run, started, activation


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise AdmissionError("activation write made no progress")
        view = view[written:]


def _private_directory_metadata(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise AdmissionError("activation parent is unavailable") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
    ):
        raise AdmissionError("activation parent must be private and owned")
    return metadata


def _persist_activation(
    destination: Path,
    payloads: Mapping[str, bytes],
    *,
    seal: bytes,
    before_seal: Callable[[Path], None] | None,
) -> None:
    parent = destination.parent
    if destination.name in {"", ".", ".."} or "/" in destination.name:
        raise AdmissionError("activation destination is invalid")
    parent_metadata = _private_directory_metadata(parent)
    parent_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        parent_fd = os.open(parent, parent_flags)
    except OSError as error:
        raise AdmissionError("activation parent cannot be pinned") from error
    child_fd = -1
    try:
        pinned_parent = os.fstat(parent_fd)
        if (pinned_parent.st_dev, pinned_parent.st_ino) != (
            parent_metadata.st_dev,
            parent_metadata.st_ino,
        ):
            raise AdmissionError("activation parent changed while being pinned")
        try:
            os.mkdir(destination.name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise AdmissionError("activation destination already exists") from error
        except OSError as error:
            raise AdmissionError("activation destination cannot be created") from error
        child_flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        child_fd = os.open(destination.name, child_flags, dir_fd=parent_fd)
        child_metadata = os.fstat(child_fd)
        if (
            not stat.S_ISDIR(child_metadata.st_mode)
            or stat.S_IMODE(child_metadata.st_mode) != 0o700
            or child_metadata.st_uid != os.getuid()
        ):
            raise AdmissionError("activation destination is not private")
        for name in _ACTIVATION_FILE_NAMES:
            payload = payloads[name]
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(name, flags, 0o600, dir_fd=child_fd)
            try:
                _write_all(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
        if before_seal is not None:
            before_seal(destination)
        current_parent = parent.lstat()
        current_child = destination.lstat()
        if (current_parent.st_dev, current_parent.st_ino) != (
            pinned_parent.st_dev,
            pinned_parent.st_ino,
        ) or (current_child.st_dev, current_child.st_ino) != (
            child_metadata.st_dev,
            child_metadata.st_ino,
        ):
            raise AdmissionError("activation path changed before the seal")
        seal_fd = os.open(
            ACTIVATION_SEAL_NAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=child_fd,
        )
        try:
            _write_all(seal_fd, seal)
            os.fsync(seal_fd)
        finally:
            os.close(seal_fd)
        os.fsync(child_fd)
        os.fsync(parent_fd)
        final_child = destination.lstat()
        if (final_child.st_dev, final_child.st_ino) != (
            child_metadata.st_dev,
            child_metadata.st_ino,
        ):
            raise AdmissionError("activation path changed after the seal")
    except OSError as error:
        raise AdmissionError("activation persistence failed closed") from error
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        os.close(parent_fd)


def activate_admitted_slot(
    result: AdmissionResult,
    *,
    authority: CurrentAuthority,
    destination: Path,
    clock: ActivationClock = _SYSTEM_CLOCK,
    opaque_id_factory: Callable[[str], str] = new_opaque_id,
    _before_seal: Callable[[Path], None] | None = None,
) -> ActivatedSlot:
    """Revalidate, assign identities, and seal activation before work starts."""

    if type(result) is not AdmissionResult:
        raise AdmissionError("activation requires an exact admission result")
    if not result.admitted:
        raise AdmissionError("a failed admission decision cannot be activated")
    decision = dict(result.decision)
    if decision.get("schema_version") != ADMISSION_DECISION_SCHEMA:
        raise AdmissionError("admission decision schema is invalid")
    if decision["admission_context_sha256"] != result.context.sha256:
        raise AdmissionError("admission context changed before activation")
    if sha256_bytes(compact_canonical_json_bytes(decision)) != result.decision_sha256:
        raise AdmissionError("admission decision changed before activation")
    observations = [dict(item) for item in result.observations]
    if (
        sha256_bytes(compact_canonical_json_bytes(observations))
        != decision["observations_sha256"]
    ):
        raise AdmissionError("admission observations changed before activation")
    snapshot, authority_sha256 = authority_snapshot(authority)
    if (
        snapshot != dict(result.authority_snapshot)
        or authority_sha256 != decision["current_authority_sha256"]
        or snapshot != dict(result.context.phase4_binding)
    ):
        raise AdmissionError("current authority changed before activation")
    activated_ns = _nonnegative_integer(
        clock.monotonic_ns(), "activation monotonic time"
    )
    activated_utc = _normalized_utc_timestamp(
        clock.wall_time_utc(), "activation wall time"
    )
    authenticated_production_result = result.authorized_for_production_activation(
        authority
    )
    serialized_production_claim = bool(
        decision["production_authority"] is True
        and decision["production_observations"] is True
    )
    if serialized_production_claim and not authenticated_production_result:
        raise AdmissionError(
            "serialized production admission lacks transient activation authority"
        )
    production_qualifying = bool(
        authenticated_production_result
        and type(authority) is Phase4CurrentAuthority
        and authority.production_qualifying
        and decision["production_authority"]
        and decision["production_observations"]
        and clock is _SYSTEM_CLOCK
        and opaque_id_factory is new_opaque_id
        and _before_seal is None
    )
    if (
        production_qualifying
        and activated_ns < decision["acquisition_finished_monotonic_ns"]
    ):
        raise AdmissionError("production activation precedes admission completion")
    experiment_run_id = opaque_id_factory("xrun")
    if (
        not isinstance(experiment_run_id, str)
        or _EXPERIMENT_RUN_ID.fullmatch(experiment_run_id) is None
    ):
        raise AdmissionError("opaque run identity factory returned an invalid ID")
    execution, run, started, activation = _build_activation_records(
        result,
        authority_sha256=authority_sha256,
        experiment_run_id=experiment_run_id,
        activated_monotonic_ns=activated_ns,
        activated_at_utc=activated_utc,
        production_qualifying=production_qualifying,
    )
    payload_values = {
        "admission-decision.json": decision,
        "admission-observations.json": observations,
        "execution-configuration.json": execution,
        "experiment-run-template.json": run,
        "started-identity-template.json": started,
        "activation-decision.json": activation,
    }
    payloads = {
        name: canonical_json_bytes(value) for name, value in payload_values.items()
    }
    seal_value = {
        "schema_version": ACTIVATION_SEAL_SCHEMA,
        "activation_decision_sha256": sha256_bytes(
            payloads["activation-decision.json"]
        ),
        "files": [
            {
                "name": name,
                "size_bytes": len(payloads[name]),
                "sha256": sha256_bytes(payloads[name]),
            }
            for name in _ACTIVATION_FILE_NAMES
        ],
    }
    _persist_activation(
        destination,
        payloads,
        seal=canonical_json_bytes(seal_value),
        before_seal=_before_seal,
    )
    return ActivatedSlot(
        directory=destination,
        admission_decision=MappingProxyType(decision),
        activation_decision=MappingProxyType(activation),
        execution_configuration=MappingProxyType(execution),
        experiment_run_template=MappingProxyType(run),
        started_identity_template=MappingProxyType(started),
        production_qualifying=production_qualifying,
    )


__all__ = [
    "ACTIVATION_FILE_NAMES",
    "ACTIVATION_DECISION_SCHEMA",
    "ACTIVATION_SEAL_NAME",
    "ADMISSION_DECISION_SCHEMA",
    "ADMISSION_OBSERVATION_SCHEMA",
    "AdmissionError",
    "AdmissionResult",
    "ActivatedSlot",
    "VerifiedActivatedSlot",
    "ExecutionProposal",
    "FrozenResourceBudget",
    "InjectedActivationClock",
    "InjectedAdmissionAuthority",
    "MAXIMUM_ADMISSION_ACQUISITION_SECONDS",
    "Phase4CurrentAuthority",
    "PlannedSlotContext",
    "REQUIRED_IDLE_OBSERVATIONS",
    "RunProposal",
    "RetainedActivatedSlot",
    "activate_admitted_slot",
    "authority_snapshot",
    "collect_production_admission_observations",
    "construct_admission_observation",
    "evaluate_pre_slot_admission",
    "planned_slot_context_from_record",
    "validate_admission_observation",
    "validate_retained_activated_slot",
    "verify_activated_slot",
]
