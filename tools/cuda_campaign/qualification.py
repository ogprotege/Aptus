"""Fail-closed identity and qualification helpers for managed CUDA attempts."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .admission import PlannedSlotContext, VerifiedActivatedSlot
from .contracts import (
    NATIVE_OUTCOMES,
    REASON_CODES,
    SCHEMA_VERSIONS,
    canonical_json_bytes,
    compact_canonical_json_bytes,
    sha256_bytes,
    validate_record,
)
from .runtime_events import RuntimeBoundary
from .monitoring import (
    MAXIMUM_QUALIFYING_GAP_SECONDS,
    MINIMUM_QUALIFYING_COVERAGE,
    SAMPLE_INTERVAL_SECONDS,
    SafetyLimits,
    WindowValidation,
    summarize_scalar,
    summarize_telemetry,
    validate_telemetry_sample,
)


QUALIFYING_ACTION_ORDER = (
    "dependency",
    "model-data",
    "preflight",
    "pilot",
    "train",
)
REQUIRED_QUALIFYING_ARTIFACT_ROLES = frozenset(
    {
        "plan",
        "bundle-manifest",
        "validation-report",
        "pilot-metrics",
        "training-metrics",
        "final-export-manifest",
        "bundle-archive",
    }
)
REQUIRED_QUALIFYING_AUTHORITY_ROLES = frozenset(
    {
        "campaign-record",
        "comparison-cohort-record",
        "comparison-cell-record",
        "phase4-source-freeze",
        "phase4-source-freeze-seal",
        "phase4-idle-baseline-samples",
        "planned-slot-context",
        "activation-admission-decision",
        "activation-admission-observations",
        "activation-execution-configuration",
        "activation-experiment-run-template",
        "activation-started-identity-template",
        "activation-decision",
        "activation-seal",
    }
)
IDLE_BASELINE_BINDING_SCHEMA = "aptus.cuda-campaign-idle-baseline-binding.v1"
_JOB_ID = re.compile(r"^job_[0-9a-f]{32}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_LINUX_BOOT_CLOCK = re.compile(r"^linux-boot-sha256:([0-9a-f]{64})$")
PENDING_TRAIN_ARGV = ["PENDING_MANAGED_TRAIN_SUBMISSION"]
_BASELINE_FIELDS = frozenset(
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
_TELEMETRY_CONFIGURATION_FIELDS = frozenset(
    {
        "configuration_sha256",
        "format_version",
        "lifecycle",
        "profile",
        "provenance",
        "safety_limits",
        "sampling",
        "thermal_policy",
    }
)
_TELEMETRY_SUPPORT_BINDINGS = frozenset(
    {
        "cpu_temperature",
        "gpu_thermal_limits",
        "hardware_events",
        "nvidia_smi_binary",
        "nvme_temperature",
        "xid_projection",
    }
)


class QualificationError(ValueError):
    """A would-be qualifying attempt lacks a frozen identity or evidence fact."""


def validate_qualifying_terminal_timing(
    record: dict[str, Any], *, expected_boot_sha256: str
) -> dict[str, Any]:
    binding = record.get("monotonic_clock_binding")
    match = _LINUX_BOOT_CLOCK.fullmatch(binding or "")
    if match is None or match.group(1) != expected_boot_sha256:
        raise QualificationError(
            "terminal job monotonic clock differs from the Phase-4 boot"
        )
    queued = record.get("queued_monotonic_ns")
    child_started = record.get("child_process_started_monotonic_ns")
    child_finished = record.get("child_process_finished_monotonic_ns")
    terminal = record.get("terminal_monotonic_ns")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (queued, terminal)
    ):
        raise QualificationError("terminal job monotonic boundaries are invalid")
    if (child_started is None) != (child_finished is None):
        raise QualificationError(
            "terminal child boundaries must be an all-or-none pair"
        )
    if child_started is not None and (
        isinstance(child_started, bool)
        or not isinstance(child_started, int)
        or child_started < 0
        or isinstance(child_finished, bool)
        or not isinstance(child_finished, int)
        or child_finished < child_started
        or child_started < queued
        or terminal < child_finished
    ):
        raise QualificationError("terminal child monotonic boundaries are invalid")
    if terminal < queued:
        raise QualificationError("terminal boundary precedes its queued boundary")
    return {
        "monotonic_clock_binding": binding,
        "queued_monotonic_ns": queued,
        "child_process_started_monotonic_ns": child_started,
        "child_process_finished_monotonic_ns": child_finished,
        "terminal_monotonic_ns": terminal,
        "child_runtime_duration_ns": (
            child_finished - child_started if child_started is not None else None
        ),
        "queue_to_terminal_duration_ns": terminal - queued,
    }


def terminal_timing_summary(
    terminal_jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not terminal_jobs:
        raise QualificationError("qualifying timing requires terminal jobs")
    bindings = {item.get("monotonic_clock_binding") for item in terminal_jobs}
    if len(bindings) != 1:
        raise QualificationError("qualifying terminal clocks differ")
    for earlier, later in zip(terminal_jobs, terminal_jobs[1:]):
        if earlier["terminal_monotonic_ns"] > later["queued_monotonic_ns"]:
            raise QualificationError("qualifying terminal actions overlap or reorder")
    return {
        "monotonic_clock_binding": next(iter(bindings)),
        "submitted_jobs_span_ns": (
            terminal_jobs[-1]["terminal_monotonic_ns"]
            - terminal_jobs[0]["queued_monotonic_ns"]
        ),
        "child_runtime_duration_ns_by_action": {
            item["action"]: item["child_runtime_duration_ns"] for item in terminal_jobs
        },
        "queue_to_terminal_duration_ns_by_action": {
            item["action"]: item["queue_to_terminal_duration_ns"]
            for item in terminal_jobs
        },
    }


def _detached_mapping(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise QualificationError(f"{label} must be an exact JSON object")
    return validate_record(value, value.get("schema_version"))


def validate_idle_baseline_binding(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _BASELINE_FIELDS:
        raise QualificationError("idle baseline binding has the wrong exact fields")
    if value["schema_version"] != IDLE_BASELINE_BINDING_SCHEMA:
        raise QualificationError("idle baseline binding schema is unsupported")
    for field in (
        "phase4_source_freeze_sha256",
        "phase4_source_freeze_seal_sha256",
        "idle_baseline_samples_sha256",
        "telemetry_configuration_sha256",
        "host_binding_sha256",
        "current_host_binding_sha256",
        "current_boot_id_sha256",
        "journalctl_binding_sha256",
    ):
        if _DIGEST.fullmatch(value[field] or "") is None:
            raise QualificationError(f"idle baseline {field} is invalid")
    summary = value["summary"]
    if type(summary) is not dict or set(summary) != _BASELINE_SUMMARY_FIELDS:
        raise QualificationError("idle baseline summary has the wrong exact fields")
    for field, number in summary.items():
        if field == "gpu_power_draw_p95_w" and number is None:
            continue
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or number < 0
        ):
            raise QualificationError(f"idle baseline summary field {field} is invalid")
    return {
        "schema_version": value["schema_version"],
        "phase4_source_freeze_sha256": value["phase4_source_freeze_sha256"],
        "phase4_source_freeze_seal_sha256": value["phase4_source_freeze_seal_sha256"],
        "idle_baseline_samples_sha256": value["idle_baseline_samples_sha256"],
        "telemetry_configuration_sha256": value["telemetry_configuration_sha256"],
        "host_binding_sha256": value["host_binding_sha256"],
        "current_host_binding_sha256": value["current_host_binding_sha256"],
        "current_boot_id_sha256": value["current_boot_id_sha256"],
        "journalctl_binding_sha256": value["journalctl_binding_sha256"],
        "summary": dict(summary),
    }


@dataclass(frozen=True, init=False)
class QualifyingRunContext:
    """Post-admission identities required before a managed attempt starts.

    Production instances can only be assembled from a planned slot that has no
    execution/run identity and the capability-bearing result of
    :func:`verify_activated_slot`.  The explicit test constructor remains
    non-qualifying and cannot be consumed by the production harness factory.
    """

    campaign: dict[str, Any]
    comparison_cohort: dict[str, Any]
    comparison_cell: dict[str, Any]
    planned_attempt_slot: dict[str, Any]
    execution_configuration: dict[str, Any]
    experiment_run_template: dict[str, Any]
    idle_baseline_binding: dict[str, Any]
    planned_slot_context: PlannedSlotContext | None
    verified_activation: VerifiedActivatedSlot | None
    _production_qualifying: bool

    def __init__(
        self,
        planned_slot_context: PlannedSlotContext,
        verified_activation: VerifiedActivatedSlot,
    ) -> None:
        if type(planned_slot_context) is not PlannedSlotContext:
            raise TypeError("planned_slot_context must be a PlannedSlotContext")
        if (
            type(verified_activation) is not VerifiedActivatedSlot
            or verified_activation.production_qualifying is not True
            or not verified_activation.authorized_for_qualifying_harness()
        ):
            raise TypeError(
                "qualifying context requires a verified production activation"
            )
        execution = dict(verified_activation.execution_configuration)
        run = dict(verified_activation.experiment_run_template)
        started = dict(verified_activation.started_identity_template)
        slot = dict(planned_slot_context.planned_attempt_slot)
        if (
            started.get("attempt_slot_id") != slot.get("attempt_slot_id")
            or started.get("execution_configuration_id")
            != execution.get("execution_configuration_id")
            or started.get("experiment_run_id") != run.get("experiment_run_id")
            or started.get("terminal_record_status") != "pending"
        ):
            raise QualificationError("verified activation started identity is misbound")
        object.__setattr__(self, "campaign", dict(planned_slot_context.campaign))
        object.__setattr__(
            self,
            "comparison_cohort",
            dict(planned_slot_context.comparison_cohort),
        )
        object.__setattr__(
            self,
            "comparison_cell",
            dict(planned_slot_context.comparison_cell),
        )
        object.__setattr__(self, "planned_attempt_slot", slot)
        object.__setattr__(self, "execution_configuration", execution)
        object.__setattr__(self, "experiment_run_template", run)
        object.__setattr__(
            self,
            "idle_baseline_binding",
            dict(planned_slot_context.phase4_binding),
        )
        object.__setattr__(self, "planned_slot_context", planned_slot_context)
        object.__setattr__(self, "verified_activation", verified_activation)
        object.__setattr__(self, "_production_qualifying", True)
        self.__post_init__()

    @classmethod
    def _for_test(
        cls,
        campaign: dict[str, Any],
        comparison_cohort: dict[str, Any],
        comparison_cell: dict[str, Any],
        planned_attempt_slot: dict[str, Any],
        execution_configuration: dict[str, Any],
        experiment_run_template: dict[str, Any],
        idle_baseline_binding: dict[str, Any],
    ) -> "QualifyingRunContext":
        """Build an injectable context that is never production qualifying."""

        self = object.__new__(cls)
        object.__setattr__(self, "campaign", campaign)
        object.__setattr__(self, "comparison_cohort", comparison_cohort)
        object.__setattr__(self, "comparison_cell", comparison_cell)
        object.__setattr__(self, "planned_attempt_slot", planned_attempt_slot)
        object.__setattr__(self, "execution_configuration", execution_configuration)
        object.__setattr__(self, "experiment_run_template", experiment_run_template)
        object.__setattr__(self, "idle_baseline_binding", idle_baseline_binding)
        object.__setattr__(self, "planned_slot_context", None)
        object.__setattr__(self, "verified_activation", None)
        object.__setattr__(self, "_production_qualifying", False)
        self.__post_init__()
        return self

    def __post_init__(self) -> None:
        campaign = _detached_mapping(self.campaign, "campaign")
        cohort = _detached_mapping(self.comparison_cohort, "comparison cohort")
        cell = _detached_mapping(self.comparison_cell, "comparison cell")
        slot = _detached_mapping(self.planned_attempt_slot, "planned attempt slot")
        execution = _detached_mapping(
            self.execution_configuration, "execution configuration"
        )
        run = _detached_mapping(self.experiment_run_template, "experiment run template")
        baseline = validate_idle_baseline_binding(self.idle_baseline_binding)
        if campaign["schema_version"] != SCHEMA_VERSIONS["campaign"]:
            raise QualificationError("campaign schema is wrong")
        if cohort["schema_version"] != SCHEMA_VERSIONS["comparison_cohort"]:
            raise QualificationError("comparison cohort schema is wrong")
        if cell["schema_version"] != SCHEMA_VERSIONS["comparison_cell"]:
            raise QualificationError("comparison cell schema is wrong")
        if slot["schema_version"] != SCHEMA_VERSIONS["attempt_slot"]:
            raise QualificationError("planned attempt slot schema is wrong")
        if execution["schema_version"] != SCHEMA_VERSIONS["execution_configuration"]:
            raise QualificationError("execution configuration schema is wrong")
        if run["schema_version"] != SCHEMA_VERSIONS["experiment_run"]:
            raise QualificationError("experiment run template schema is wrong")
        if (
            slot["slot_status"] != "planned-not-started"
            or slot["execution_configuration_id"] is not None
            or slot["experiment_run_id"] is not None
            or slot["native_outcome"] is not None
            or slot["evidence_status"] != "not-started"
        ):
            raise QualificationError(
                "qualifying context requires a canonical planned-not-started slot"
            )
        if slot["comparison_cell_id"] != execution["comparison_cell_id"]:
            raise QualificationError("slot and execution configuration cells differ")
        if (
            cohort["campaign_id"] != campaign["campaign_id"]
            or cell["campaign_id"] != campaign["campaign_id"]
            or cell["comparison_cell_id"] not in cohort["member_cell_ids"]
            or slot["comparison_cohort_id"] != cohort["comparison_cohort_id"]
            or slot["comparison_cell_id"] != cell["comparison_cell_id"]
            or execution["comparison_cell_id"] != cell["comparison_cell_id"]
        ):
            raise QualificationError(
                "campaign, cohort, cell, slot, and execution membership is misbound"
            )
        if (
            cell["method"] not in campaign["allowed_methods"]
            or cell["placement"] != campaign["allowed_placement"]
            or cell["world_size"] != campaign["allowed_world_size"]
            or execution["training_seed"] != slot["scheduled_seed"]
            or execution["data_order_seed"] != 1_000_000 + slot["scheduled_seed"]
            or (
                "split_seed" in cell["seed_policy"]
                and cell["seed_policy"]["split_seed"] != execution["split_seed"]
            )
        ):
            raise QualificationError(
                "campaign method, placement, world size, or seed binding is invalid"
            )
        if (
            run["attempt_slot_id"] != slot["attempt_slot_id"]
            or run["execution_configuration_id"]
            != execution["execution_configuration_id"]
        ):
            raise QualificationError("experiment run template identity is misbound")
        for field in ("plan_id", "candidate_id", "bundle_fingerprint"):
            if run[field] != execution[field]:
                raise QualificationError(f"experiment run template {field} is misbound")
        if (
            run["aptus_job_ids"]
            or run["aptus_run_ids"]
            or run["terminal_evidence"] != {"status": "pending"}
        ):
            raise QualificationError(
                "experiment run template must not contain post-start evidence"
            )
        if run["exact_argv"] != PENDING_TRAIN_ARGV:
            raise QualificationError(
                "pre-start experiment run exact_argv must use the pending sentinel"
            )
        for field in (
            "working_directory",
            "fresh_state_root",
            "bundle_path",
            "output_path",
        ):
            path = Path(run[field])
            if not path.is_absolute() or "\x00" in run[field]:
                raise QualificationError(f"experiment run template {field} is invalid")
        observed = run["observed_host_state"]
        baseline_digest = sha256_bytes(canonical_json_bytes(baseline))
        if type(observed) is not dict or observed.get("idle_baseline_sha256") != (
            baseline_digest
        ):
            raise QualificationError(
                "experiment run host state does not bind the idle baseline"
            )
        if self._production_qualifying:
            planned = self.planned_slot_context
            verified = self.verified_activation
            if planned is None or verified is None:
                raise QualificationError(
                    "production context lacks its admission activation"
                )
            if (
                campaign != dict(planned.campaign)
                or cohort != dict(planned.comparison_cohort)
                or cell != dict(planned.comparison_cell)
                or slot != dict(planned.planned_attempt_slot)
                or execution != dict(verified.execution_configuration)
                or run != dict(verified.experiment_run_template)
                or baseline != dict(planned.phase4_binding)
            ):
                raise QualificationError(
                    "production context differs from its verified activation"
                )
            expected_observed = {
                "idle_baseline_sha256": baseline_digest,
                "admission_decision_sha256": sha256_bytes(
                    compact_canonical_json_bytes(dict(verified.admission_decision))
                ),
                "current_authority_sha256": verified.activation_decision[
                    "current_authority_sha256"
                ],
                "resource_budget_sha256": planned.resource_budget.sha256,
            }
            if observed != expected_observed:
                raise QualificationError(
                    "experiment run host state differs from its admission"
                )
        if baseline["host_binding_sha256"] != sha256_bytes(
            canonical_json_bytes(cell["host_binding"])
        ):
            raise QualificationError(
                "idle baseline does not bind the comparison-cell host"
            )
        if type(run["run_order"]) is not dict or run["run_order"] != {
            "block": slot["block"],
            "position": slot["order_position"],
        }:
            raise QualificationError("experiment run order differs from its slot")
        object.__setattr__(self, "campaign", MappingProxyType(campaign))
        object.__setattr__(self, "comparison_cohort", MappingProxyType(cohort))
        object.__setattr__(self, "comparison_cell", MappingProxyType(cell))
        object.__setattr__(self, "planned_attempt_slot", MappingProxyType(slot))
        object.__setattr__(self, "execution_configuration", MappingProxyType(execution))
        object.__setattr__(self, "experiment_run_template", MappingProxyType(run))
        object.__setattr__(self, "idle_baseline_binding", MappingProxyType(baseline))

    @property
    def production_qualifying(self) -> bool:
        verified = self.verified_activation
        return bool(
            self._production_qualifying
            and type(verified) is VerifiedActivatedSlot
            and verified.production_qualifying is True
            and verified.authorized_for_qualifying_harness()
        )

    @property
    def experiment_run_id(self) -> str:
        return str(self.experiment_run_template["experiment_run_id"])

    @property
    def attempt_slot_id(self) -> str:
        return str(self.planned_attempt_slot["attempt_slot_id"])

    @property
    def emergency_deadline_seconds(self) -> float:
        return float(self.execution_configuration["emergency_deadline_seconds"])

    @property
    def remaining_disk_budget_bytes(self) -> int:
        value = self.execution_configuration["exact_behavior_values"].get(
            "remaining_disk_budget_bytes"
        )
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise QualificationError(
                "execution configuration lacks remaining_disk_budget_bytes"
            )
        return value

    @property
    def idle_baseline_summary(self) -> dict[str, Any]:
        return dict(self.idle_baseline_binding["summary"])

    def source_bindings(self) -> dict[str, str]:
        bindings = {
            "campaign_sha256": sha256_bytes(canonical_json_bytes(dict(self.campaign))),
            "comparison_cohort_sha256": sha256_bytes(
                canonical_json_bytes(dict(self.comparison_cohort))
            ),
            "comparison_cell_sha256": sha256_bytes(
                canonical_json_bytes(dict(self.comparison_cell))
            ),
            "planned_attempt_slot_sha256": sha256_bytes(
                canonical_json_bytes(dict(self.planned_attempt_slot))
            ),
            "execution_configuration_sha256": sha256_bytes(
                canonical_json_bytes(dict(self.execution_configuration))
            ),
            "experiment_run_template_sha256": sha256_bytes(
                canonical_json_bytes(dict(self.experiment_run_template))
            ),
            "idle_baseline_binding_sha256": sha256_bytes(
                canonical_json_bytes(dict(self.idle_baseline_binding))
            ),
        }
        if self.production_qualifying:
            assert self.planned_slot_context is not None
            assert self.verified_activation is not None
            bindings.update(
                planned_slot_context_sha256=self.planned_slot_context.sha256,
                admission_decision_sha256=sha256_bytes(
                    canonical_json_bytes(
                        dict(self.verified_activation.admission_decision)
                    )
                ),
                admission_decision_semantic_sha256=sha256_bytes(
                    compact_canonical_json_bytes(
                        dict(self.verified_activation.admission_decision)
                    )
                ),
                admission_observations_sha256=sha256_bytes(
                    canonical_json_bytes(
                        [dict(item) for item in self.verified_activation.observations]
                    )
                ),
                activation_decision_sha256=sha256_bytes(
                    canonical_json_bytes(
                        dict(self.verified_activation.activation_decision)
                    )
                ),
                started_identity_template_sha256=sha256_bytes(
                    canonical_json_bytes(
                        dict(self.verified_activation.started_identity_template)
                    )
                ),
            )
        return bindings

    def finalize_records(
        self,
        terminal_job_records: list[dict[str, Any]],
        *,
        native_outcome: str,
        evidence_status: str,
        reason_code: str,
        evidence_role_sha256: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if native_outcome not in NATIVE_OUTCOMES or reason_code not in REASON_CODES:
            raise QualificationError("terminal attempt disposition is invalid")
        if evidence_status not in {"protocol-valid", "capture-invalid"}:
            raise QualificationError("terminal attempt evidence status is invalid")
        job_ids: list[str] = []
        run_ids: list[str] = []
        terminal_jobs: list[dict[str, Any]] = []
        _require_context_bound_action_records(self, terminal_job_records)
        expected_boot_sha256 = self.idle_baseline_binding["current_boot_id_sha256"]
        for index, record in enumerate(terminal_job_records):
            if type(record) is not dict:
                raise QualificationError("terminal job evidence must be an object")
            job_id = record.get("job_id", record.get("id"))
            if job_id is None:
                if (
                    index != len(terminal_job_records) - 1
                    or record.get("native_outcome") != native_outcome
                ):
                    raise QualificationError("jobless terminal disposition is misbound")
                continue
            if (
                not isinstance(job_id, str)
                or _JOB_ID.fullmatch(job_id) is None
                or record.get("id", job_id) != job_id
                or job_id in job_ids
            ):
                raise QualificationError("terminal job identity is invalid")
            state = record.get("state")
            action = record.get("action")
            if action not in QUALIFYING_ACTION_ORDER:
                raise QualificationError("terminal job action is invalid")
            if state not in {"completed", "failed", "cancelled"}:
                raise QualificationError("terminal job state is invalid")
            run_id = record.get("run_id")
            if run_id is not None:
                if not isinstance(run_id, str) or not run_id or run_id in run_ids:
                    raise QualificationError("terminal Aptus run identity is invalid")
                run_ids.append(run_id)
            return_code = record.get("return_code")
            if isinstance(return_code, bool) or not isinstance(
                return_code, (int, type(None))
            ):
                raise QualificationError("terminal job return code is invalid")
            job_ids.append(job_id)
            timing = validate_qualifying_terminal_timing(
                record, expected_boot_sha256=expected_boot_sha256
            )
            terminal_jobs.append(
                {
                    "job_id": job_id,
                    "run_id": run_id,
                    "action": action,
                    "state": state,
                    "return_code": return_code,
                    **timing,
                }
            )
        timing_summary = (
            terminal_timing_summary(terminal_jobs) if terminal_jobs else None
        )
        terminal_evidence: dict[str, Any] = {
            "native_outcome": native_outcome,
            "evidence_status": evidence_status,
            "reason_code": reason_code,
            "jobs": terminal_jobs,
            "timing": timing_summary,
        }
        if evidence_role_sha256 is not None:
            passing_roles = (
                {
                    "attempt-slot-record",
                    "execution-configuration-record",
                    "idle-baseline-binding",
                    "telemetry-configuration",
                    "telemetry-summary",
                    "cooldown-summary",
                }
                | REQUIRED_QUALIFYING_ARTIFACT_ROLES
                | REQUIRED_QUALIFYING_AUTHORITY_ROLES
            )
            nonpass_roles = {
                "attempt-slot-record",
                "execution-configuration-record",
                "idle-baseline-binding",
                "telemetry-configuration",
                "telemetry-summary",
                "plan",
                "bundle-manifest",
                "validation-report",
                "bundle-archive",
            } | REQUIRED_QUALIFYING_AUTHORITY_ROLES
            if any(
                record.get("action") == "pilot"
                and record.get("state") == "completed"
                and record.get("return_code") == 0
                for record in terminal_job_records
            ):
                nonpass_roles.add("pilot-metrics")
            expected_roles = (
                passing_roles
                if native_outcome == "passed" and evidence_status == "protocol-valid"
                else nonpass_roles
            )
            if (
                type(evidence_role_sha256) is not dict
                or set(evidence_role_sha256) != expected_roles
                or "experiment-run-record" in evidence_role_sha256
            ):
                raise QualificationError("terminal evidence digest roles are invalid")
            if any(
                not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None
                for digest in evidence_role_sha256.values()
            ):
                raise QualificationError("terminal evidence digest is invalid")
            terminal_evidence["evidence_role_sha256"] = dict(
                sorted(evidence_role_sha256.items())
            )
        run = dict(self.experiment_run_template)
        submitted_records = [
            record
            for record in terminal_job_records
            if isinstance(record.get("command"), list)
            and isinstance(record.get("bundle_dir"), str)
        ]
        last_submitted = submitted_records[-1] if submitted_records else None
        run.update(
            aptus_job_ids=job_ids,
            aptus_run_ids=run_ids,
            exact_argv=(
                list(last_submitted["command"])
                if last_submitted is not None
                else list(PENDING_TRAIN_ARGV)
            ),
            working_directory=(
                str(last_submitted["bundle_dir"])
                if last_submitted is not None
                else str(run["working_directory"])
            ),
            terminal_evidence=terminal_evidence,
        )
        finalized_run = validate_record(run, SCHEMA_VERSIONS["experiment_run"])
        slot = dict(self.planned_attempt_slot)
        slot.update(
            slot_status="started",
            execution_configuration_id=self.execution_configuration[
                "execution_configuration_id"
            ],
            experiment_run_id=self.experiment_run_id,
            native_outcome=native_outcome,
            evidence_status=evidence_status,
            reason_code=reason_code,
        )
        finalized_slot = validate_record(slot, SCHEMA_VERSIONS["attempt_slot"])
        return finalized_slot, finalized_run


def validate_passing_runtime_boundaries(
    boundaries: list[RuntimeBoundary],
    *,
    pilot_job_id: str,
    train_job_id: str,
) -> None:
    """Require the exact successful runtime-emitted Phase-1 boundary order."""

    expected = (
        ("pilot.phase-started", "pilot-phase-1", pilot_job_id),
        ("pilot.phase-finished", "pilot-phase-1", pilot_job_id),
        ("pilot.phase-started", "pilot-phase-2", pilot_job_id),
        ("pilot.phase-finished", "pilot-phase-2", pilot_job_id),
        ("training.started", "training", train_job_id),
        ("export.started", "final-export", train_job_id),
        ("export.finished", "final-export", train_job_id),
        ("training.finished", "training", train_job_id),
        ("verification.started", "parent-verification", train_job_id),
        ("verification.finished", "parent-verification", train_job_id),
    )
    observed = tuple((item.event_type, item.phase, item.job_id) for item in boundaries)
    if observed != expected:
        raise QualificationError("runtime boundary sequence is incomplete or reordered")
    prior = -1
    for item in boundaries:
        if item.monotonic_ns < prior:
            raise QualificationError("runtime boundary monotonic time moved backward")
        prior = item.monotonic_ns
        if item.event_type.endswith("finished") and (
            item.native_outcome != "passed" or item.reason_code != "NONE"
        ):
            raise QualificationError("passing runtime boundary is not a pass")


def _require_context_bound_action_records(
    context: QualifyingRunContext, action_records: list[dict[str, Any]]
) -> None:
    """Cross-bind every retained terminal to the pre-start run declaration."""

    if not 1 <= len(action_records) <= len(QUALIFYING_ACTION_ORDER):
        raise QualificationError("qualifying terminal action count is invalid")
    run = context.experiment_run_template
    execution = context.execution_configuration
    bundle = Path(str(run["bundle_path"])).resolve()
    output_root = Path(str(run["output_path"])).resolve()
    if output_root != bundle / "runs":
        raise QualificationError("qualifying output root is not bundle/runs")
    interpreter: str | None = None
    levels = {
        "dependency": "dependency",
        "model-data": "model-data",
        "preflight": "measured-preflight",
        "pilot": "pilot",
    }
    for index, (expected_action, record) in enumerate(
        zip(QUALIFYING_ACTION_ORDER, action_records)
    ):
        if type(record) is not dict or record.get("action") != expected_action:
            raise QualificationError("qualifying terminal action is misbound")
        job_id = record.get("job_id", record.get("id"))
        if job_id is None:
            if index != len(action_records) - 1 or record.get("native_outcome") not in {
                "refused",
                "guard-blocked",
                "unknown",
            }:
                raise QualificationError(
                    "jobless qualifying action is not the stopping record"
                )
            continue
        try:
            record_bundle = Path(str(record["bundle_dir"])).resolve()
        except (KeyError, OSError, RuntimeError, ValueError):
            raise QualificationError("qualifying terminal bundle is invalid") from None
        if (
            record_bundle != bundle
            or record.get("artifact_fingerprint") != execution["bundle_fingerprint"]
            or record.get("plan_id") != execution["plan_id"]
            or record.get("candidate_id") != execution["candidate_id"]
            or record.get("bundle_manifest_sha256") != run["bundle_manifest_sha256"]
        ):
            raise QualificationError("qualifying terminal bundle identity is misbound")
        command = record.get("command")
        if (
            type(command) is not list
            or not command
            or any(type(item) is not str or not item for item in command)
        ):
            raise QualificationError("qualifying terminal command is invalid")
        if interpreter is None:
            interpreter = command[0]
        elif command[0] != interpreter:
            raise QualificationError("qualifying terminal interpreter changed")
        if expected_action in levels:
            expected = [interpreter, "validate.py", "--level", levels[expected_action]]
            if command != expected:
                raise QualificationError("qualifying validation command is misbound")
            if (
                record.get("run_id") is not None
                or record.get("run_output_dir") is not None
            ):
                raise QualificationError("non-training terminal has a run output")
            continue
        run_id = record.get("run_id")
        output_value = record.get("run_output_dir")
        if not isinstance(run_id, str) or not isinstance(output_value, str):
            raise QualificationError("training terminal lacks its run output")
        run_output = Path(output_value).resolve()
        if run_output.parent != output_root or run_output.name != run_id:
            raise QualificationError("training output is outside its declared root")
        relative_output = str(Path("runs") / run_id)
        if command[-2:] != ["--output-dir", relative_output]:
            raise QualificationError("training command output is misbound")
        prefix = command[:-2]
        direct = prefix in (
            [interpreter, "train.py", "--confirm-full-train"],
            [interpreter, "run.py", "--confirm-full-train"],
            [
                interpreter,
                "run.py",
                "--confirm-full-train",
                "--defer-parent-promotion",
            ],
        )
        accelerated = (
            len(prefix) == 8
            and prefix[:4]
            == [interpreter, "-m", "accelerate.commands.accelerate_cli", "launch"]
            and prefix[4:6] == ["--config_file", "config/accelerate.yaml"]
            and prefix[6:] == ["train.py", "--confirm-full-train"]
        )
        if not direct and not accelerated:
            raise QualificationError("training command is not canonical")


def validate_qualifying_telemetry_configuration(
    value: Any, *, context: QualifyingRunContext
) -> dict[str, Any]:
    """Verify the exact self-digested frozen sidecar configuration."""

    if type(value) is not dict or set(value) != _TELEMETRY_CONFIGURATION_FIELDS:
        raise QualificationError("telemetry configuration has the wrong exact fields")
    digest = value["configuration_sha256"]
    unsigned = dict(value)
    unsigned.pop("configuration_sha256")
    if (
        not isinstance(digest, str)
        or _DIGEST.fullmatch(digest) is None
        or sha256_bytes(compact_canonical_json_bytes(unsigned)) != digest
    ):
        raise QualificationError("telemetry configuration digest is invalid")
    if value["format_version"] != "aptus.cuda-telemetry-configuration.v1":
        raise QualificationError("telemetry configuration version is unsupported")
    profile = value["profile"]
    if type(profile) is not dict or profile != {
        "id": "phase1-frozen-qualifying",
        "qualifying": True,
        "reason_code": None,
    }:
        raise QualificationError("telemetry profile is not qualifying")
    thermal = value["thermal_policy"]
    if (
        type(thermal) is not dict
        or set(thermal) != {"initial_limits_available", "mode"}
        or not isinstance(thermal["initial_limits_available"], bool)
        or thermal["mode"]
        != (
            "reported-limits-bound"
            if thermal["initial_limits_available"]
            else "frozen-conservative-fallback"
        )
    ):
        raise QualificationError("telemetry thermal policy is invalid")
    expected_limits = SafetyLimits.frozen_phase1(
        emergency_deadline_seconds=context.emergency_deadline_seconds,
        remaining_disk_budget_bytes=context.remaining_disk_budget_bytes,
        initial_thermal_limits_available=thermal["initial_limits_available"],
    )
    expected_limit_record = {
        item.name: getattr(expected_limits, item.name)
        for item in fields(expected_limits)
    }
    if value["safety_limits"] != expected_limit_record:
        raise QualificationError("telemetry safety limits differ from Phase 1")
    sampling = value["sampling"]
    if type(sampling) is not dict or sampling != {
        "interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "minimum_qualifying_coverage": MINIMUM_QUALIFYING_COVERAGE,
        "watchdog_interval_seconds": 0.25,
    }:
        raise QualificationError("telemetry sampling policy differs from Phase 1")
    provenance = value["provenance"]
    if type(provenance) is not dict or set(provenance) != {
        "disk_growth_binding",
        "ownership_binding",
        "provider",
        "support_bindings",
    }:
        raise QualificationError("telemetry provider provenance is invalid")
    provider = provenance["provider"]
    support = provenance["support_bindings"]
    required_text = (
        provenance["disk_growth_binding"],
        provenance["ownership_binding"],
        provider.get("name") if type(provider) is dict else None,
        provider.get("version") if type(provider) is dict else None,
    )
    if any(not isinstance(item, str) or not item for item in required_text):
        raise QualificationError("telemetry provider provenance is incomplete")
    if (
        type(support) is not dict
        or set(support) != _TELEMETRY_SUPPORT_BINDINGS
        or any(not isinstance(item, str) or not item for item in support.values())
    ):
        raise QualificationError("telemetry support bindings are incomplete")
    lifecycle = value["lifecycle"]
    if (
        type(lifecycle) is not dict
        or set(lifecycle) != {"join_timeout_seconds", "readiness_timeout_seconds"}
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or item <= 0
            for item in lifecycle.values()
        )
    ):
        raise QualificationError("telemetry lifecycle policy is invalid")
    return dict(value)


def build_segment_summaries(
    samples: list[dict[str, Any]],
    ledger_records: list[dict[str, Any]],
    *,
    allow_open_terminal_prefix: bool = False,
) -> list[dict[str, Any]]:
    """Derive bounded per-boundary telemetry summaries without changing samples."""

    validated_samples = [validate_telemetry_sample(item) for item in samples]
    pairs = {
        "command.started": "command.finished",
        "pilot.phase-started": "pilot.phase-finished",
        "training.started": "training.finished",
        "export.started": "export.finished",
        "verification.started": "verification.finished",
        "cooldown.started": "cooldown.finished",
    }
    active: dict[tuple[Any, ...], dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for record in ledger_records:
        event_type = record.get("event_type")
        key = (
            record.get("phase"),
            record.get("action"),
            record.get("subject_kind"),
            record.get("subject_id"),
        )
        if event_type in pairs:
            active[key] = record
            continue
        starts = [name for name, finish in pairs.items() if finish == event_type]
        if not starts or key not in active:
            continue
        started = active.pop(key)
        start_ns = started["monotonic_ns"]
        stop_ns = record["monotonic_ns"]
        window = [
            sample
            for sample in validated_samples
            if start_ns <= sample["observed_monotonic_ns"] <= stop_ns
        ]
        telemetry: dict[str, Any] | None = None
        if window:
            telemetry = {
                "sample_count": len(window),
                "scheduled_slots": [item["scheduled_slot"] for item in window],
                "gpu_utilization_percent": summarize_scalar(
                    [item["gpu"]["utilization_percent"] for item in window]
                ),
                "gpu_temperature_c": summarize_scalar(
                    [item["gpu"]["temperature_c"] for item in window]
                ),
                "gpu_power_draw_w": summarize_scalar(
                    [item["gpu"]["power_draw_w"] for item in window]
                ),
                "gpu_memory_used_bytes": summarize_scalar(
                    [item["gpu"]["memory"]["used"]["bytes"] for item in window]
                ),
                "gpu_memory_free_bytes": summarize_scalar(
                    [item["gpu"]["memory"]["free"]["bytes"] for item in window],
                    free_resource=True,
                ),
                "host_mem_available_bytes": summarize_scalar(
                    [item["host"]["mem_available_bytes"] for item in window],
                    free_resource=True,
                ),
                "managed_process_rss_bytes": summarize_scalar(
                    [item["host"]["managed_process_rss_bytes"] for item in window]
                ),
            }
        summaries.append(
            {
                "started_event_type": started["event_type"],
                "finished_event_type": event_type,
                "phase": key[0],
                "action": key[1],
                "subject_kind": key[2],
                "subject_id": key[3],
                "started_monotonic_ns": start_ns,
                "finished_monotonic_ns": stop_ns,
                "duration_ns": stop_ns - start_ns,
                "telemetry": telemetry,
            }
        )
    if active and not allow_open_terminal_prefix:
        raise QualificationError("segment boundary set is incomplete")
    return summaries


@dataclass(frozen=True)
class QualificationDecision:
    valid: bool
    reason_codes: tuple[str, ...]
    telemetry_summary: dict[str, Any]
    segment_summaries: tuple[dict[str, Any], ...]
    cooldown_summary: dict[str, Any]


def evaluate_passing_qualification(
    *,
    context: QualifyingRunContext,
    action_records: list[dict[str, Any]],
    selected_artifact_roles: set[str],
    runtime_boundaries: list[RuntimeBoundary],
    telemetry_samples: list[dict[str, Any]],
    telemetry_start_monotonic_ns: int,
    telemetry_stop_monotonic_ns: int,
    telemetry_configuration: dict[str, Any],
    safety_events: list[dict[str, Any]],
    cooldown: WindowValidation,
    ledger_records: list[dict[str, Any]],
) -> QualificationDecision:
    """Return a protocol-valid decision only for one complete passing attempt."""

    reasons: list[str] = []
    actions = [record.get("action") for record in action_records]
    if tuple(actions) != QUALIFYING_ACTION_ORDER or any(
        record.get("state") != "completed" or record.get("return_code") != 0
        for record in action_records
    ):
        reasons.append("MISSING_REQUIRED_EVIDENCE")
    try:
        _require_context_bound_action_records(context, action_records)
    except QualificationError:
        reasons.append("MISSING_REQUIRED_EVIDENCE")
    missing_roles = REQUIRED_QUALIFYING_ARTIFACT_ROLES - selected_artifact_roles
    if missing_roles:
        reasons.append("MISSING_REQUIRED_EVIDENCE")
    try:
        validate_qualifying_telemetry_configuration(
            telemetry_configuration, context=context
        )
    except QualificationError:
        reasons.append("MISSING_REQUIRED_EVIDENCE")
    if safety_events:
        reasons.append(
            safety_events[0].get("reason_code", "MISSING_REQUIRED_EVIDENCE")
            if type(safety_events[0]) is dict
            else "MISSING_REQUIRED_EVIDENCE"
        )
    job_by_action = {
        record.get("action"): record.get("job_id", record.get("id"))
        for record in action_records
    }
    try:
        validate_passing_runtime_boundaries(
            runtime_boundaries,
            pilot_job_id=job_by_action.get("pilot"),
            train_job_id=job_by_action.get("train"),
        )
    except QualificationError:
        reasons.append("MISSING_REQUIRED_EVIDENCE")
    try:
        telemetry_summary = summarize_telemetry(
            telemetry_samples,
            telemetry_start_monotonic_ns,
            telemetry_stop_monotonic_ns,
        )
    except (TypeError, ValueError):
        telemetry_summary = {}
        reasons.append("TELEMETRY_QUALIFYING_GAP")
    else:
        expected_slots = range(telemetry_summary["expected_sample_count"])
        present_slots = {item["scheduled_slot"] for item in telemetry_samples}
        telemetry_summary["missing_scheduled_slots"] = [
            slot for slot in expected_slots if slot not in present_slots
        ]
        if (
            telemetry_summary["coverage"] < MINIMUM_QUALIFYING_COVERAGE
            or telemetry_summary["maximum_gap_seconds"] > MAXIMUM_QUALIFYING_GAP_SECONDS
        ):
            reasons.append("TELEMETRY_QUALIFYING_GAP")
    try:
        segments = build_segment_summaries(telemetry_samples, ledger_records)
    except (QualificationError, TypeError, ValueError):
        segments = []
        reasons.append("MISSING_REQUIRED_EVIDENCE")
    if not cooldown.valid:
        reasons.extend(cooldown.reason_codes or ("MISSING_REQUIRED_EVIDENCE",))
    valid_reason_codes = tuple(dict.fromkeys(reasons))
    return QualificationDecision(
        valid=not valid_reason_codes,
        reason_codes=valid_reason_codes,
        telemetry_summary=telemetry_summary,
        segment_summaries=tuple(segments),
        cooldown_summary=dict(cooldown.summary),
    )


__all__ = [
    "IDLE_BASELINE_BINDING_SCHEMA",
    "QUALIFYING_ACTION_ORDER",
    "REQUIRED_QUALIFYING_ARTIFACT_ROLES",
    "REQUIRED_QUALIFYING_AUTHORITY_ROLES",
    "QualificationError",
    "QualificationDecision",
    "QualifyingRunContext",
    "validate_idle_baseline_binding",
    "validate_passing_runtime_boundaries",
    "validate_qualifying_telemetry_configuration",
    "build_segment_summaries",
    "evaluate_passing_qualification",
]
