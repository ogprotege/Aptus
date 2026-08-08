from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

from aptus.execution import JobService
from tools.cuda_campaign import admission as admission_module
from tools.cuda_campaign.admission import (
    ACTIVATION_FILE_NAMES,
    ACTIVATION_SEAL_NAME,
    AdmissionError,
    AdmissionResult,
    ExecutionProposal,
    FrozenResourceBudget,
    InjectedActivationClock,
    InjectedAdmissionAuthority,
    Phase4CurrentAuthority,
    PlannedSlotContext,
    RunProposal,
    VerifiedActivatedSlot,
    activate_admitted_slot,
    authority_snapshot,
    collect_production_admission_observations,
    construct_admission_observation,
    evaluate_pre_slot_admission,
    validate_retained_activated_slot,
    verify_activated_slot,
)
from tools.cuda_campaign.contracts import (
    SCHEMA_VERSIONS,
    canonical_json_bytes,
    compact_canonical_json_bytes,
    deterministic_id,
    sha256_bytes,
)
from tools.cuda_campaign.monitoring import GIB, MIB, NANOSECONDS_PER_SECOND


DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64


def _identity(
    record: dict[str, object],
    *,
    field: str,
    prefix: str,
    fields: tuple[str, ...],
) -> dict[str, object]:
    record[field] = deterministic_id(prefix, {name: record[name] for name in fields})
    return record


def _campaign() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["campaign"],
        "protocol_schema_version": "aptus.cuda-campaign-protocol.v1",
        "program_key": "rtx-3050-local",
        "phase_sequence": list(range(11)),
        "host_class": "single-rtx-3050-8gib",
        "allowed_methods": ["lora", "qlora"],
        "allowed_placement": "single",
        "allowed_world_size": 1,
    }
    return _identity(
        value,
        field="campaign_id",
        prefix="campaign_",
        fields=tuple(value),
    )


def _cell(campaign_id: str) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["comparison_cell"],
        "campaign_id": campaign_id,
        "source_binding": {"commit": "f" * 40},
        "host_binding": {"host_id": "host_" + "6" * 32},
        "environment_binding": {"python": "3.12"},
        "model_binding": {"revision": "e" * 40},
        "dataset_and_split_binding": {"sha256": DIGEST},
        "method": "lora",
        "precision": "bf16",
        "quantization": None,
        "placement": "single",
        "world_size": 1,
        "sequence_length": 256,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 8,
        "training_budget": {"optimizer_steps": 128},
        "checkpoint_rule": {"cadence": 64},
        "adapter_targets": ["q_proj", "v_proj"],
        "seed_policy": {"split_seed": 424242},
        "cache_policy": {"model": "warm"},
        "cooldown_policy": {"samples": 120},
        "safety_policy": {"temperature_c": 84},
        "capture_policy": {"sample_interval_seconds": 1},
        "retention_policy_id": "cuda-v02-public-claim-evidence-24m-v1",
    }
    return _identity(
        value,
        field="comparison_cell_id",
        prefix="cell_",
        fields=tuple(value),
    )


def _cohort(campaign_id: str, cell_id: str) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["comparison_cohort"],
        "campaign_id": campaign_id,
        "question": "Does the anchor repeat?",
        "held_controls": {"placement": "single"},
        "varied_dimensions": ["training_seed"],
        "member_cell_ids": [cell_id],
        "attempt_counts": {"anchor": 5},
        "seed_schedule": {"training": [101]},
        "block_schedule": [{"block": 1}],
        "stopping_rule": {"rule": "no-replacement"},
        "promotion_rule": {"required": 5},
        "no_replacement_rule": True,
        "aggregate_rule": {"median": "type-7"},
    }
    return _identity(
        value,
        field="comparison_cohort_id",
        prefix="cohort_",
        fields=tuple(value),
    )


def _slot(cohort_id: str, cell_id: str) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["attempt_slot"],
        "comparison_cohort_id": cohort_id,
        "comparison_cell_id": cell_id,
        "block": 0,
        "ordinal": 1,
        "role": "anchor",
        "order_position": 0,
        "scheduled_seed": 101,
        "slot_status": "planned-not-started",
        "execution_configuration_id": None,
        "experiment_run_id": None,
        "native_outcome": None,
        "evidence_status": "not-started",
        "reason_code": "PRIOR_STOP_RULE",
    }
    return _identity(
        value,
        field="attempt_slot_id",
        prefix="slot_",
        fields=(
            "schema_version",
            "comparison_cohort_id",
            "comparison_cell_id",
            "block",
            "ordinal",
            "role",
            "order_position",
            "scheduled_seed",
        ),
    )


def _phase4(
    cell: dict[str, object], *, power_p95_w: float | None = 20.0
) -> dict[str, object]:
    return {
        "schema_version": "aptus.cuda-campaign-idle-baseline-binding.v1",
        "phase4_source_freeze_sha256": "1" * 64,
        "phase4_source_freeze_seal_sha256": "2" * 64,
        "idle_baseline_samples_sha256": "3" * 64,
        "telemetry_configuration_sha256": "4" * 64,
        "host_binding_sha256": sha256_bytes(canonical_json_bytes(cell["host_binding"])),
        "current_host_binding_sha256": "5" * 64,
        "current_boot_id_sha256": "6" * 64,
        "journalctl_binding_sha256": "7" * 64,
        "summary": {
            "gpu_temperature_median_c": 40.0,
            "gpu_temperature_p95_c": 41.0,
            "gpu_free_vram_median_bytes": 7 * GIB,
            "gpu_power_draw_p95_w": power_p95_w,
        },
    }


def _context(
    *,
    plan_ram: int = 8 * GIB,
    plan_disk: int = 2 * GIB,
    power_p95_w: float | None = 20.0,
) -> PlannedSlotContext:
    campaign = _campaign()
    cell = _cell(str(campaign["campaign_id"]))
    cohort = _cohort(str(campaign["campaign_id"]), str(cell["comparison_cell_id"]))
    slot = _slot(str(cohort["comparison_cohort_id"]), str(cell["comparison_cell_id"]))
    budget = FrozenResourceBudget(
        plan_id="plan_example",
        candidate_id="candidate_example",
        bundle_fingerprint=DIGEST,
        comparison_cell_id=str(cell["comparison_cell_id"]),
        attempt_slot_id=str(slot["attempt_slot_id"]),
        exact_artifact_bytes=1 * GIB,
        plan_required_disk_bytes=plan_disk,
        largest_pilot_checkpoint_bytes=512 * MIB,
        final_export_bytes=1 * GIB,
        expected_copied_output_bytes=1 * GIB,
        expected_log_bytes=64 * MIB,
        expected_telemetry_bytes=64 * MIB,
        plan_required_host_ram_bytes=plan_ram,
    )
    phase4 = _phase4(cell, power_p95_w=power_p95_w)
    behavior = {
        "emergency_deadline_seconds": 300,
        "resource_budget_sha256": budget.sha256,
        "phase4_binding_sha256": sha256_bytes(compact_canonical_json_bytes(phase4)),
    }
    return PlannedSlotContext(
        campaign=campaign,
        comparison_cohort=cohort,
        comparison_cell=cell,
        planned_attempt_slot=slot,
        execution_proposal=ExecutionProposal(
            exact_behavior_values=behavior,
            plan_id="plan_example",
            candidate_id="candidate_example",
            bundle_fingerprint=DIGEST,
            split_seed=424242,
            training_seed=101,
            data_order_seed=1_000_101,
            emergency_deadline_seconds=300,
        ),
        run_proposal=RunProposal(
            working_directory="/private/work",
            fresh_state_root="/private/state",
            bundle_path="/private/bundle",
            output_path="/private/bundle/runs",
            bundle_manifest_sha256=OTHER_DIGEST,
            archive_sha256="c" * 64,
        ),
        phase4_binding=phase4,
        resource_budget=budget,
    )


def _probe(
    *,
    mem_available_bytes: int,
    filesystem_free_bytes: int,
    compute_processes: list[dict[str, object]] | None = None,
    lease: bool = False,
) -> dict[str, object]:
    return {
        "gpu": {
            "uuid": "GPU-private-test",
            "memory_used": {"value": str(1 * GIB), "unit": "B"},
            "memory_free": {"value": str(7 * GIB), "unit": "B"},
            "memory_total": {"value": str(8 * GIB), "unit": "B"},
            "utilization_percent": 0.0,
            "temperature_c": 40.0,
            "power_draw_w": 20.0,
            "power_limit_w": 130.0,
            "graphics_clock_mhz": 210.0,
            "memory_clock_mhz": 405.0,
            "performance_state": "P8",
            "throttle_reasons": [],
            "throttle_state": "0x0000000000000000",
            "xid_errors": [],
            "reset_detected": False,
            "device_lost": False,
            "hardware_error": False,
            "compute_processes": compute_processes or [],
        },
        "host": {
            "mem_available_bytes": mem_available_bytes,
            "swap_used_bytes": 0,
            "swap_read_bytes": 0,
            "swap_write_bytes": 0,
            "load_1m": 0.25,
            "filesystem_free_bytes": filesystem_free_bytes,
            "managed_process_rss_bytes": 0,
            "managed_process_cpu_seconds": 0.0,
            "managed_process_read_bytes": 0,
            "managed_process_write_bytes": 0,
            "disk_growth_bytes": 0,
            "aptus_lease_active": lease,
            "cpu_temperature": {
                "status": "unsupported",
                "value": None,
                "reason_code": "NOT_CONFIGURED",
            },
            "nvme_temperature": {
                "status": "unsupported",
                "value": None,
                "reason_code": "NOT_CONFIGURED",
            },
        },
    }


def _observations(
    context: PlannedSlotContext,
    authority: InjectedAdmissionAuthority,
    *,
    count: int = 120,
    mem_available_bytes: int = 64 * GIB,
    filesystem_free_bytes: int = 100 * GIB,
    compute_processes: list[dict[str, object]] | None = None,
    lease: bool = False,
) -> list[dict[str, object]]:
    _, authority_sha256 = authority_snapshot(authority)
    result = []
    for index in range(count):
        timestamp = index * NANOSECONDS_PER_SECOND
        result.append(
            construct_admission_observation(
                sequence=index,
                admission_context_sha256=context.sha256,
                current_authority_sha256=authority_sha256,
                scheduled_slot=index,
                scheduled_monotonic_ns=timestamp,
                observed_monotonic_ns=timestamp,
                wall_time_utc="2026-08-08T12:00:00+00:00",
                probe_reading=_probe(
                    mem_available_bytes=mem_available_bytes,
                    filesystem_free_bytes=filesystem_free_bytes,
                    compute_processes=compute_processes,
                    lease=lease,
                ),
                collector={
                    "healthy": True,
                    "status_code": None,
                    "probe_duration_ns": 1000,
                },
                watchdog={
                    "healthy": True,
                    "heartbeat_monotonic_ns": timestamp,
                    "ownership_certain": True,
                },
            )
        )
    return result


def _evaluate(
    context: PlannedSlotContext,
    authority: InjectedAdmissionAuthority,
    observations: list[dict[str, object]],
    *,
    finished_seconds: int = 119,
):
    return evaluate_pre_slot_admission(
        context,
        observations,
        authority=authority,
        acquisition_started_monotonic_ns=0,
        acquisition_finished_monotonic_ns=(finished_seconds * NANOSECONDS_PER_SECOND),
    )


class FrozenBudgetTests(unittest.TestCase):
    def test_exact_integer_ceiling_formulas_and_bindings(self) -> None:
        context = _context()
        budget = context.resource_budget
        self.assertEqual(budget.download_bytes, 3_489_660_928)
        self.assertEqual(budget.output_bytes, 4_026_531_840)
        self.assertEqual(budget.vault_bytes, 1_328_755_508)
        self.assertEqual(
            budget.admission_disk_bytes,
            32 * GIB + budget.download_bytes + budget.output_bytes + budget.vault_bytes,
        )
        self.assertEqual(budget.admission_host_ram_bytes, 16 * GIB)
        self.assertEqual(
            budget.record()["bindings"]["attempt_slot_id"],
            context.planned_attempt_slot["attempt_slot_id"],
        )

    def test_missing_or_zero_physical_input_fails_instead_of_defaulting(self) -> None:
        context = _context()
        values = {
            name: getattr(context.resource_budget, name)
            for name in context.resource_budget.__dataclass_fields__
        }
        values.pop("expected_telemetry_bytes")
        with self.assertRaises(TypeError):
            FrozenResourceBudget(**values)
        values["expected_telemetry_bytes"] = 0
        with self.assertRaisesRegex(AdmissionError, "must be positive"):
            FrozenResourceBudget(**values)

    def test_exact_behavior_must_bind_budget_and_phase4(self) -> None:
        base = _context()
        behavior = dict(base.execution_proposal.exact_behavior_values)
        behavior["resource_budget_sha256"] = "f" * 64
        with self.assertRaisesRegex(AdmissionError, "does not bind"):
            PlannedSlotContext(
                campaign=base.campaign,
                comparison_cohort=base.comparison_cohort,
                comparison_cell=base.comparison_cell,
                planned_attempt_slot=base.planned_attempt_slot,
                execution_proposal=ExecutionProposal(
                    exact_behavior_values=behavior,
                    plan_id=base.execution_proposal.plan_id,
                    candidate_id=base.execution_proposal.candidate_id,
                    bundle_fingerprint=base.execution_proposal.bundle_fingerprint,
                    split_seed=base.execution_proposal.split_seed,
                    training_seed=base.execution_proposal.training_seed,
                    data_order_seed=base.execution_proposal.data_order_seed,
                    emergency_deadline_seconds=base.execution_proposal.emergency_deadline_seconds,
                ),
                run_proposal=base.run_proposal,
                phase4_binding=base.phase4_binding,
                resource_budget=base.resource_budget,
            )


class AdmissionGateTests(unittest.TestCase):
    def test_exactly_120_safe_observations_admit_without_started_identity(self) -> None:
        context = _context()
        authority = InjectedAdmissionAuthority(lambda: dict(context.phase4_binding))
        observations = _observations(context, authority)
        result = _evaluate(context, authority, observations)
        self.assertTrue(result.admitted)
        self.assertIsNone(result.execution_configuration_id)
        self.assertIsNone(result.experiment_run_id)
        self.assertIsNone(result.artifact_id)
        self.assertEqual(
            dict(result.context.planned_attempt_slot),
            dict(context.planned_attempt_slot),
        )
        self.assertNotIn("experiment_run_id", observations[0])
        self.assertNotIn("xrun_", json.dumps(dict(result.decision)))

    def test_119_and_121_observations_fail(self) -> None:
        for count in (119, 121):
            with self.subTest(count=count):
                context = _context()
                authority = InjectedAdmissionAuthority(
                    lambda: dict(context.phase4_binding)
                )
                result = _evaluate(
                    context, authority, _observations(context, authority, count=count)
                )
                self.assertFalse(result.admitted)
                self.assertIn(
                    "MISSING_REQUIRED_EVIDENCE", result.decision["reason_codes"]
                )

    def test_acquisition_over_1800_seconds_fails(self) -> None:
        context = _context()
        authority = InjectedAdmissionAuthority(lambda: dict(context.phase4_binding))
        result = _evaluate(
            context,
            authority,
            _observations(context, authority),
            finished_seconds=1801,
        )
        self.assertFalse(result.admitted)
        self.assertIn("EMERGENCY_DEADLINE_EXCEEDED", result.decision["reason_codes"])

    def test_60_gib_plan_is_not_admitted_on_62_gib_available(self) -> None:
        context = _context(plan_ram=60 * GIB)
        authority = InjectedAdmissionAuthority(lambda: dict(context.phase4_binding))
        result = _evaluate(
            context,
            authority,
            _observations(context, authority, mem_available_bytes=62 * GIB),
        )
        self.assertFalse(result.admitted)
        self.assertIn("HOST_RAM_WARNING", result.decision["reason_codes"])
        self.assertEqual(
            result.decision["summary"]["required_host_ram_bytes"], 68 * GIB
        )

    def test_exact_disk_budget_foreign_cuda_and_lease_fail(self) -> None:
        context = _context()
        authority = InjectedAdmissionAuthority(lambda: dict(context.phase4_binding))
        observations = _observations(
            context,
            authority,
            filesystem_free_bytes=context.resource_budget.admission_disk_bytes - 1,
            compute_processes=[
                {
                    "pid": 91,
                    "used_memory": {"value": "1", "unit": "MiB"},
                    "managed": False,
                }
            ],
            lease=True,
        )
        result = _evaluate(context, authority, observations)
        self.assertFalse(result.admitted)
        self.assertIn("DISK_BUDGET_INSUFFICIENT", result.decision["reason_codes"])
        self.assertIn("UNRELATED_GPU_ACTIVITY", result.decision["reason_codes"])
        self.assertIn("OWNERSHIP_UNCERTAIN", result.decision["reason_codes"])

    def test_power_channel_unavailable_is_frozen_and_not_fabricated(self) -> None:
        context = _context(power_p95_w=None)
        authority = InjectedAdmissionAuthority(lambda: dict(context.phase4_binding))
        result = _evaluate(context, authority, _observations(context, authority))
        self.assertTrue(result.admitted, result.decision["reason_codes"])
        self.assertFalse(result.decision["summary"]["power_channel_supported"])

    def test_malformed_sequence_maps_only_to_a_frozen_reason_code(self) -> None:
        context = _context()
        authority = InjectedAdmissionAuthority(lambda: dict(context.phase4_binding))
        observations = _observations(context, authority)
        observations[1]["sequence"] = 0
        result = _evaluate(context, authority, observations)
        self.assertFalse(result.admitted)
        self.assertEqual(result.decision["reason_codes"], ["MISSING_REQUIRED_EVIDENCE"])

    def test_observations_are_exactly_bound_to_the_acquisition_window(self) -> None:
        context = _context()
        authority = InjectedAdmissionAuthority(lambda: dict(context.phase4_binding))
        observations = _observations(context, authority)
        offset = 10_000 * NANOSECONDS_PER_SECOND
        for observation in observations:
            observation["scheduled_monotonic_ns"] += offset
            observation["observed_monotonic_ns"] += offset
            observation["watchdog"]["heartbeat_monotonic_ns"] += offset
        result = _evaluate(context, authority, observations)
        self.assertFalse(result.admitted)
        self.assertEqual(result.decision["reason_codes"], ["MISSING_REQUIRED_EVIDENCE"])

        observations = _observations(context, authority)
        observations[-1]["observed_monotonic_ns"] += 1
        observations[-1]["watchdog"]["heartbeat_monotonic_ns"] += 1
        result = _evaluate(context, authority, observations)
        self.assertFalse(result.admitted)
        self.assertEqual(result.decision["reason_codes"], ["MISSING_REQUIRED_EVIDENCE"])

    def test_swap_warning_uses_complete_rolling_ten_second_window(self) -> None:
        context = _context()
        authority = InjectedAdmissionAuthority(lambda: dict(context.phase4_binding))
        early_burst = _observations(context, authority)
        for index, observation in enumerate(early_burst):
            value = 32 * MIB if index >= 1 else 0
            observation["host"]["swap_read_bytes"] = value
        safe = _evaluate(context, authority, early_burst)
        self.assertTrue(safe.admitted, safe.decision["reason_codes"])

        sustained = _observations(context, authority)
        for index, observation in enumerate(sustained):
            observation["host"]["swap_read_bytes"] = index * 16 * MIB
        rejected = _evaluate(context, authority, sustained)
        self.assertFalse(rejected.admitted)
        self.assertIn("SWAP_RATE_WARNING", rejected.decision["reason_codes"])

    def test_injected_authority_cannot_enter_production_collector(self) -> None:
        context = _context()
        authority = InjectedAdmissionAuthority(lambda: dict(context.phase4_binding))
        with self.assertRaisesRegex(TypeError, "Phase4CurrentAuthority"):
            collect_production_admission_observations(
                context,
                authority=authority,  # type: ignore[arg-type]
                filesystem_path=Path("/"),
                job_service=object(),
            )

    def test_imported_batch_internals_cannot_mint_production_authority(self) -> None:
        self.assertFalse(hasattr(admission_module, "_PRODUCTION_OBSERVATION_TOKEN"))
        context = _context()
        authority = InjectedAdmissionAuthority(lambda: dict(context.phase4_binding))
        observations = _observations(context, authority)
        batch_type = admission_module._ProductionAdmissionObservationBatch
        with self.assertRaisesRegex(TypeError, "collect_production"):
            batch_type()
        forged = object.__new__(batch_type)
        forged._initialize_from_collector(
            observations=observations,
            acquisition_started_monotonic_ns=0,
            acquisition_finished_monotonic_ns=119 * NANOSECONDS_PER_SECOND,
            authority=authority,
        )
        self.assertFalse(forged.authorized_for(authority))
        with self.assertRaisesRegex(AdmissionError, "wrong authority"):
            evaluate_pre_slot_admission(context, forged, authority=authority)

    def test_observation_construction_requires_normalized_real_utc(self) -> None:
        context = _context()
        authority = InjectedAdmissionAuthority(lambda: dict(context.phase4_binding))
        _, authority_sha256 = authority_snapshot(authority)
        for value in (
            "2026-02-30T12:00:00+00:00",
            "2026-08-08T12:00:00Z",
            "2026-08-08T08:00:00-04:00",
        ):
            with self.subTest(value=value), self.assertRaises(AdmissionError):
                construct_admission_observation(
                    sequence=0,
                    admission_context_sha256=context.sha256,
                    current_authority_sha256=authority_sha256,
                    scheduled_slot=0,
                    scheduled_monotonic_ns=0,
                    observed_monotonic_ns=0,
                    wall_time_utc=value,
                    probe_reading=_probe(
                        mem_available_bytes=64 * GIB,
                        filesystem_free_bytes=100 * GIB,
                    ),
                    collector={
                        "healthy": True,
                        "status_code": None,
                        "probe_duration_ns": 1,
                    },
                    watchdog={
                        "healthy": True,
                        "heartbeat_monotonic_ns": 0,
                        "ownership_certain": True,
                    },
                )


class ActivationTests(unittest.TestCase):
    def _retained_fixture(self) -> tuple[dict[str, bytes], dict[str, object]]:
        from tests.tools.test_cuda_campaign_storage import (
            protocol_activation_authority,
        )

        fixture = protocol_activation_authority()
        context_record = fixture.pop("planned-slot-context")
        roles = {
            "admission-decision.json": "activation-admission-decision",
            "admission-observations.json": "activation-admission-observations",
            "execution-configuration.json": "activation-execution-configuration",
            "experiment-run-template.json": "activation-experiment-run-template",
            "started-identity-template.json": "activation-started-identity-template",
            "activation-decision.json": "activation-decision",
            ACTIVATION_SEAL_NAME: "activation-seal",
        }
        return (
            {name: fixture[role] for name, role in roles.items()},
            context_record,
        )

    def _reseal_activation(self, payloads: dict[str, bytes]) -> None:
        seal = {
            "schema_version": "aptus.cuda-campaign-activation-seal.v1",
            "activation_decision_sha256": sha256_bytes(
                payloads["activation-decision.json"]
            ),
            "files": [
                {
                    "name": name,
                    "size_bytes": len(payloads[name]),
                    "sha256": sha256_bytes(payloads[name]),
                }
                for name in ACTIVATION_FILE_NAMES
            ],
        }
        payloads[ACTIVATION_SEAL_NAME] = canonical_json_bytes(seal)

    def _bind_changed_observations(
        self,
        payloads: dict[str, bytes],
        observations: list[dict[str, object]],
    ) -> None:
        payloads["admission-observations.json"] = canonical_json_bytes(observations)
        decision = json.loads(payloads["admission-decision.json"])
        decision["observations_sha256"] = sha256_bytes(
            compact_canonical_json_bytes(observations)
        )
        payloads["admission-decision.json"] = canonical_json_bytes(decision)
        decision_sha256 = sha256_bytes(compact_canonical_json_bytes(decision))
        run = json.loads(payloads["experiment-run-template.json"])
        run["observed_host_state"]["admission_decision_sha256"] = decision_sha256
        payloads["experiment-run-template.json"] = canonical_json_bytes(run)
        activation = json.loads(payloads["activation-decision.json"])
        activation["admission_decision_sha256"] = decision_sha256
        activation["experiment_run_template_sha256"] = sha256_bytes(
            payloads["experiment-run-template.json"]
        )
        payloads["activation-decision.json"] = canonical_json_bytes(activation)
        self._reseal_activation(payloads)

    def _admitted(self):
        context = _context()
        state = {"value": dict(context.phase4_binding)}
        authority = InjectedAdmissionAuthority(lambda: deepcopy(state["value"]))
        result = _evaluate(context, authority, _observations(context, authority))
        self.assertTrue(result.admitted)
        return context, state, authority, result

    def _activate_test_only(
        self,
        parent: Path,
        name: str,
        authority: InjectedAdmissionAuthority,
        result,
    ):
        return activate_admitted_slot(
            result,
            authority=authority,
            destination=parent / name,
            clock=InjectedActivationClock(1, "2026-08-08T12:05:00+00:00"),
            opaque_id_factory=lambda _: "xrun_" + "d" * 32,
        )

    def test_forged_production_decision_booleans_cannot_authorize_activation(
        self,
    ) -> None:
        self.assertFalse(hasattr(admission_module, "_ADMISSION_RESULT_TOKEN"))
        context = _context()
        authority = Phase4CurrentAuthority(
            directory=Path("/protected/phase4"),
            repository_root=Path("/protected/repository"),
            campaign=context.campaign,
            comparison_cohort=context.comparison_cohort,
            comparison_cell=context.comparison_cell,
        )
        with patch.object(
            Phase4CurrentAuthority,
            "snapshot",
            return_value=dict(context.phase4_binding),
        ):
            observations = _observations(context, authority)  # type: ignore[arg-type]
            untrusted = evaluate_pre_slot_admission(
                context,
                observations,
                authority=authority,
                acquisition_started_monotonic_ns=0,
                acquisition_finished_monotonic_ns=(119 * NANOSECONDS_PER_SECOND),
            )
            self.assertTrue(untrusted.admitted)
            self.assertTrue(untrusted.decision["production_authority"])
            self.assertFalse(untrusted.decision["production_observations"])

            serialized = json.loads(
                compact_canonical_json_bytes(dict(untrusted.decision))
            )
            serialized["production_observations"] = True
            forged = AdmissionResult(
                context=untrusted.context,
                observations=untrusted.observations,
                decision=MappingProxyType(serialized),
                decision_sha256=sha256_bytes(compact_canonical_json_bytes(serialized)),
                authority_snapshot=untrusted.authority_snapshot,
            )
            self.assertFalse(forged.authorized_for_production_activation(authority))

            with tempfile.TemporaryDirectory() as temporary:
                parent = Path(temporary)
                os.chmod(parent, 0o700)
                destination = parent / "forged-activation"
                with self.assertRaisesRegex(
                    AdmissionError, "serialized production admission"
                ):
                    activate_admitted_slot(
                        forged,
                        authority=authority,
                        destination=destination,
                    )
                self.assertFalse(destination.exists())

    def test_authentic_result_activates_and_persistence_reverifies_separately(
        self,
    ) -> None:
        context = _context()

        class TestWatchdog:
            def __init__(self, _lease_active) -> None:
                self.index = 0

            def start(self) -> None:
                return None

            def snapshot(self) -> dict[str, object]:
                heartbeat = self.index * NANOSECONDS_PER_SECOND
                self.index += 1
                return {
                    "healthy": True,
                    "heartbeat_monotonic_ns": heartbeat,
                    "ownership_certain": True,
                }

            def stop(self) -> None:
                return None

        monotonic_values = [0]
        for index in range(120):
            monotonic_values.extend([index * NANOSECONDS_PER_SECOND] * 3)
        monotonic_values.append(119 * NANOSECONDS_PER_SECOND)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chmod(root, 0o700)
            service = JobService(root / "jobs", runtime_environment={})
            authority = Phase4CurrentAuthority(
                directory=root / "phase4",
                repository_root=root,
                campaign=context.campaign,
                comparison_cohort=context.comparison_cohort,
                comparison_cell=context.comparison_cell,
            )
            probe = lambda: _probe(  # noqa: E731 - exact callable test seam.
                mem_available_bytes=64 * GIB,
                filesystem_free_bytes=100 * GIB,
            )
            with (
                patch.object(
                    Phase4CurrentAuthority,
                    "snapshot",
                    return_value=dict(context.phase4_binding),
                ),
                patch.object(admission_module.sys, "platform", "linux"),
                patch.object(
                    admission_module,
                    "_AdmissionOwnershipWatchdog",
                    TestWatchdog,
                ),
                patch.object(
                    admission_module,
                    "_real_sleep_until",
                    return_value=True,
                ),
                patch.object(
                    admission_module,
                    "utc_now",
                    return_value="2026-08-08T12:00:00+00:00",
                ),
                patch(
                    "tools.cuda_campaign.monitoring.resolve_trusted_nvidia_smi",
                    return_value=SimpleNamespace(path="/usr/bin/nvidia-smi"),
                ),
                patch(
                    "tools.cuda_campaign.monitoring.detect_nvidia_thermal_limit_authority",
                    return_value=SimpleNamespace(provider=None),
                ),
                patch(
                    "tools.cuda_campaign.monitoring.LinuxNvidiaHostProbe",
                    return_value=probe,
                ),
                patch(
                    "tools.cuda_campaign.monitoring.LinuxNvidiaJournalEventProvider.production",
                    return_value=SimpleNamespace(snapshot=lambda: {}),
                ),
                patch(
                    "tools.cuda_campaign.monitoring.StatvfsDiskGrowthProvider.production",
                    return_value=lambda: 0,
                ),
            ):
                with patch.object(
                    admission_module.time,
                    "monotonic_ns",
                    side_effect=monotonic_values,
                ):
                    batch = collect_production_admission_observations(
                        context,
                        authority=authority,
                        filesystem_path=root,
                        job_service=service,
                    )
                result = evaluate_pre_slot_admission(
                    context,
                    batch,
                    authority=authority,
                )
                self.assertTrue(result.admitted)
                self.assertTrue(result.authorized_for_production_activation(authority))
                activation = activate_admitted_slot(
                    result,
                    authority=authority,
                    destination=root / "activation",
                )
                self.assertTrue(activation.production_qualifying)

                replayed_decision = json.loads(
                    compact_canonical_json_bytes(dict(result.decision))
                )
                replayed_observations = tuple(
                    MappingProxyType(item)
                    for item in json.loads(
                        compact_canonical_json_bytes(
                            [dict(item) for item in result.observations]
                        )
                    )
                )
                replayed = AdmissionResult(
                    context=context,
                    observations=replayed_observations,
                    decision=MappingProxyType(replayed_decision),
                    decision_sha256=result.decision_sha256,
                    authority_snapshot=MappingProxyType(
                        json.loads(
                            compact_canonical_json_bytes(
                                dict(result.authority_snapshot)
                            )
                        )
                    ),
                )
                self.assertFalse(
                    replayed.authorized_for_production_activation(authority)
                )
                with self.assertRaisesRegex(
                    AdmissionError, "serialized production admission"
                ):
                    activate_admitted_slot(
                        replayed,
                        authority=authority,
                        destination=root / "replayed-activation",
                    )

                payloads = {
                    name: (activation.directory / name).read_bytes()
                    for name in (*ACTIVATION_FILE_NAMES, ACTIVATION_SEAL_NAME)
                }
                retained = validate_retained_activated_slot(
                    payloads,
                    planned_slot_context_record=context.record(),
                )
                self.assertFalse(hasattr(retained, "authorized_for_qualifying_harness"))
                verified = verify_activated_slot(
                    activation.directory,
                    expected_context=context,
                    authority=authority,
                )
                self.assertTrue(verified.production_qualifying)
                self.assertTrue(verified.authorized_for_qualifying_harness())

    def test_imported_verifier_internals_cannot_brand_an_activation(self) -> None:
        self.assertFalse(hasattr(admission_module, "_VERIFIED_ACTIVATION_TOKEN"))
        with self.assertRaisesRegex(TypeError, "verify_activated_slot"):
            VerifiedActivatedSlot()
        forged = object.__new__(VerifiedActivatedSlot)
        self.assertFalse(forged.production_qualifying)
        self.assertFalse(forged.authorized_for_qualifying_harness())

    def test_activation_construction_requires_normalized_real_utc(self) -> None:
        _, _, authority, result = self._admitted()
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            os.chmod(parent, 0o700)
            with self.assertRaisesRegex(AdmissionError, "calendar timestamp"):
                activate_admitted_slot(
                    result,
                    authority=authority,
                    destination=parent / "activation",
                    clock=InjectedActivationClock(1, "2026-02-30T12:05:00+00:00"),
                    opaque_id_factory=lambda _: "xrun_" + "d" * 32,
                )

    def test_retained_deep_verifier_rejects_resealed_acquisition_mismatch(
        self,
    ) -> None:
        payloads, context_record = self._retained_fixture()
        validate_retained_activated_slot(
            payloads,
            planned_slot_context_record=context_record,
        )

        observations = json.loads(payloads["admission-observations.json"])
        observations[-1]["observed_monotonic_ns"] += 1
        observations[-1]["watchdog"]["heartbeat_monotonic_ns"] += 1
        self._bind_changed_observations(payloads, observations)
        with self.assertRaisesRegex(AdmissionError, "does not reproduce"):
            validate_retained_activated_slot(
                payloads,
                planned_slot_context_record=context_record,
            )

    def test_resealed_malformed_utc_timestamps_are_rejected(self) -> None:
        payloads, context_record = self._retained_fixture()
        observations = json.loads(payloads["admission-observations.json"])
        observations[0]["wall_time_utc"] = "2026-02-30T12:00:00+00:00"
        self._bind_changed_observations(payloads, observations)
        with self.assertRaisesRegex(AdmissionError, "calendar timestamp"):
            validate_retained_activated_slot(
                payloads,
                planned_slot_context_record=context_record,
            )

        payloads, context_record = self._retained_fixture()
        activation = json.loads(payloads["activation-decision.json"])
        activation["activated_at_utc"] = "2026-08-08T12:05:00Z"
        payloads["activation-decision.json"] = canonical_json_bytes(activation)
        self._reseal_activation(payloads)
        with self.assertRaisesRegex(AdmissionError, "normalized RFC 3339"):
            validate_retained_activated_slot(
                payloads,
                planned_slot_context_record=context_record,
            )

    def test_identity_is_minted_only_after_gate_and_activation_seals_first(
        self,
    ) -> None:
        context, _, authority, result = self._admitted()
        calls: list[str] = []

        def opaque(kind: str) -> str:
            calls.append(kind)
            return "xrun_" + "d" * 32

        self.assertEqual(calls, [])
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            os.chmod(parent, 0o700)
            activated = activate_admitted_slot(
                result,
                authority=authority,
                destination=parent / "activation",
                clock=InjectedActivationClock(
                    200 * NANOSECONDS_PER_SECOND,
                    "2026-08-08T12:05:00+00:00",
                ),
                opaque_id_factory=opaque,
            )
            self.assertEqual(calls, ["xrun"])
            self.assertRegex(
                activated.execution_configuration_id, r"^exec_[0-9a-f]{20}$"
            )
            self.assertEqual(activated.experiment_run_id, "xrun_" + "d" * 32)
            self.assertFalse(activated.production_qualifying)
            self.assertTrue((activated.directory / ACTIVATION_SEAL_NAME).is_file())
            self.assertEqual(
                stat.S_IMODE(
                    (activated.directory / ACTIVATION_SEAL_NAME).stat().st_mode
                ),
                0o600,
            )
            run = json.loads(
                (activated.directory / "experiment-run-template.json").read_bytes()
            )
            self.assertEqual(run["experiment_run_id"], activated.experiment_run_id)
            self.assertEqual(run["terminal_evidence"], {"status": "pending"})

    def test_authority_mutation_fails_before_identity_mint(self) -> None:
        _, state, authority, result = self._admitted()
        state["value"]["current_host_binding_sha256"] = "e" * 64
        calls: list[str] = []

        def opaque(kind: str) -> str:
            calls.append(kind)
            return "xrun_" + "d" * 32

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            os.chmod(parent, 0o700)
            with self.assertRaisesRegex(AdmissionError, "authority changed"):
                activate_admitted_slot(
                    result,
                    authority=authority,
                    destination=parent / "activation",
                    opaque_id_factory=opaque,
                )
        self.assertEqual(calls, [])

    def test_symlink_swap_and_no_clobber_fail_closed(self) -> None:
        _, _, authority, result = self._admitted()
        clock = InjectedActivationClock(1, "2026-08-08T12:05:00+00:00")

        def opaque(_: str) -> str:
            return "xrun_" + "d" * 32

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            os.chmod(parent, 0o700)
            real = parent / "real"
            real.mkdir(mode=0o700)
            link = parent / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(AdmissionError, "parent must be private"):
                activate_admitted_slot(
                    result,
                    authority=authority,
                    destination=link / "activation",
                    clock=clock,
                    opaque_id_factory=opaque,
                )

    def test_deep_verifier_rejects_test_authority_tamper_links_and_inventory(
        self,
    ) -> None:
        context, _, authority, result = self._admitted()
        clock = InjectedActivationClock(1, "2026-08-08T12:05:00+00:00")

        def opaque(_: str) -> str:
            return "xrun_" + "d" * 32

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            os.chmod(parent, 0o700)

            intact = self._activate_test_only(parent, "intact", authority, result)
            with self.assertRaisesRegex(AdmissionError, "not production-admitted"):
                verify_activated_slot(
                    intact.directory,
                    expected_context=context,
                    authority=authority,
                )

            tampered = self._activate_test_only(parent, "tampered", authority, result)
            decision_path = tampered.directory / "admission-decision.json"
            decision_path.write_bytes(decision_path.read_bytes() + b" ")
            with self.assertRaisesRegex(AdmissionError, "canonical JSON|exact files"):
                verify_activated_slot(
                    tampered.directory,
                    expected_context=context,
                    authority=authority,
                )

            extra = self._activate_test_only(parent, "extra", authority, result)
            (extra.directory / "unexpected.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(AdmissionError, "inventory"):
                verify_activated_slot(
                    extra.directory,
                    expected_context=context,
                    authority=authority,
                )

            linked = self._activate_test_only(parent, "linked", authority, result)
            os.link(
                linked.directory / "admission-decision.json",
                parent / "external-hardlink.json",
            )
            with self.assertRaisesRegex(AdmissionError, "metadata is unsafe"):
                verify_activated_slot(
                    linked.directory,
                    expected_context=context,
                    authority=authority,
                )

            symlink = parent / "activation-link"
            symlink.symlink_to(intact.directory, target_is_directory=True)
            with self.assertRaisesRegex(AdmissionError, "private and owned"):
                verify_activated_slot(
                    symlink,
                    expected_context=context,
                    authority=authority,
                )

            destination = parent / "activation"

            def swap(path: Path) -> None:
                moved = path.with_name("moved")
                path.rename(moved)
                path.symlink_to(moved, target_is_directory=True)

            with self.assertRaisesRegex(AdmissionError, "path changed"):
                activate_admitted_slot(
                    result,
                    authority=authority,
                    destination=destination,
                    clock=clock,
                    opaque_id_factory=opaque,
                    _before_seal=swap,
                )

            existing = parent / "existing"
            existing.mkdir(mode=0o700)
            with self.assertRaisesRegex(AdmissionError, "already exists"):
                activate_admitted_slot(
                    result,
                    authority=authority,
                    destination=existing,
                    clock=clock,
                    opaque_id_factory=opaque,
                )


if __name__ == "__main__":
    unittest.main()
