from __future__ import annotations

import unittest
from pathlib import Path

from tools.cuda_campaign.admission import (
    ExecutionProposal,
    FrozenResourceBudget,
    PlannedSlotContext,
    RunProposal,
)
from tools.cuda_campaign.contracts import (
    SCHEMA_VERSIONS,
    canonical_json_bytes,
    compact_canonical_json_bytes,
    deterministic_id,
    sha256_bytes,
)
from tools.cuda_campaign.monitoring import (
    GIB,
    construct_telemetry_sample,
    validate_cooldown,
)
from tools.cuda_campaign.qualification import (
    IDLE_BASELINE_BINDING_SCHEMA,
    REQUIRED_QUALIFYING_ARTIFACT_ROLES,
    REQUIRED_QUALIFYING_AUTHORITY_ROLES,
    QualificationError,
    QualifyingRunContext,
    evaluate_passing_qualification,
    validate_passing_runtime_boundaries,
    validate_qualifying_telemetry_configuration,
)
from tools.cuda_campaign.runtime_events import RuntimeBoundary
from tools.cuda_campaign.sidecar import BackgroundTelemetrySession


DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
RUN_ID = "xrun_" + "c" * 32
WALL = "2026-08-08T12:00:00+00:00"


def qualifying_context(
    *,
    bundle_path: str = "/protected/bundle",
    state_root: str = "/protected/state",
    bundle_manifest_sha256: str = DIGEST,
    archive_sha256: str = OTHER_DIGEST,
    role: str = "anchor",
) -> QualifyingRunContext:
    campaign_identity = {
        "schema_version": SCHEMA_VERSIONS["campaign"],
        "protocol_schema_version": "aptus.cuda-campaign-protocol.v1",
        "program_key": "rtx-3050-local",
        "phase_sequence": list(range(11)),
        "host_class": "single-rtx-3050-8gib",
        "allowed_methods": ["lora", "qlora"],
        "allowed_placement": "single",
        "allowed_world_size": 1,
    }
    campaign = {
        **campaign_identity,
        "campaign_id": deterministic_id("campaign_", campaign_identity),
    }
    cell_identity = {
        "schema_version": SCHEMA_VERSIONS["comparison_cell"],
        "campaign_id": campaign["campaign_id"],
        "source_binding": {"commit": "f" * 40, "tree": "e" * 40},
        "host_binding": {"host_id": "host_" + "6" * 32},
        "environment_binding": {
            "python": "3.12",
            "nvidia_driver_version": "595.84",
        },
        "model_binding": {"revision": "d" * 40},
        "dataset_and_split_binding": {"sha256": "9" * 64},
        "method": "lora",
        "precision": "bf16",
        "quantization": None,
        "placement": "single",
        "world_size": 1,
        "sequence_length": 256,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
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
    cell = {
        **cell_identity,
        "comparison_cell_id": deterministic_id("cell_", cell_identity),
    }
    cohort_identity = {
        "schema_version": SCHEMA_VERSIONS["comparison_cohort"],
        "campaign_id": campaign["campaign_id"],
        "question": "Does the anchor repeat?",
        "held_controls": {"placement": "single"},
        "varied_dimensions": ["training_seed"],
        "member_cell_ids": [cell["comparison_cell_id"]],
        "attempt_counts": {"anchor": 5},
        "seed_schedule": {"training": [17]},
        "block_schedule": [{"block": 0}],
        "stopping_rule": {"rule": "no-replacement"},
        "promotion_rule": {"required": 5},
        "no_replacement_rule": True,
        "aggregate_rule": {"median": "type-7"},
    }
    cohort = {
        **cohort_identity,
        "comparison_cohort_id": deterministic_id("cohort_", cohort_identity),
    }
    slot_identity = {
        "schema_version": SCHEMA_VERSIONS["attempt_slot"],
        "comparison_cohort_id": cohort["comparison_cohort_id"],
        "comparison_cell_id": cell["comparison_cell_id"],
        "block": 0,
        "ordinal": 1,
        "role": role,
        "order_position": 0,
        "scheduled_seed": 17,
    }
    slot = {
        **slot_identity,
        "attempt_slot_id": deterministic_id("slot_", slot_identity),
        "slot_status": "planned-not-started",
        "execution_configuration_id": None,
        "experiment_run_id": None,
        "native_outcome": None,
        "evidence_status": "not-started",
        "reason_code": "PRIOR_STOP_RULE",
    }
    baseline = {
        "schema_version": IDLE_BASELINE_BINDING_SCHEMA,
        "phase4_source_freeze_sha256": OTHER_DIGEST,
        "phase4_source_freeze_seal_sha256": "3" * 64,
        "idle_baseline_samples_sha256": "4" * 64,
        "telemetry_configuration_sha256": "5" * 64,
        "host_binding_sha256": sha256_bytes(canonical_json_bytes(cell["host_binding"])),
        "current_host_binding_sha256": "6" * 64,
        "current_boot_id_sha256": "7" * 64,
        "journalctl_binding_sha256": "8" * 64,
        "summary": {
            "gpu_temperature_median_c": 35.0,
            "gpu_temperature_p95_c": 37.0,
            "gpu_free_vram_median_bytes": 7 * GIB,
            "gpu_power_draw_p95_w": 20.0,
        },
    }
    budget = FrozenResourceBudget(
        plan_id="plan_example",
        candidate_id="candidate_example",
        bundle_fingerprint=DIGEST,
        comparison_cell_id=cell["comparison_cell_id"],
        attempt_slot_id=slot["attempt_slot_id"],
        exact_artifact_bytes=1,
        plan_required_disk_bytes=1,
        largest_pilot_checkpoint_bytes=1,
        final_export_bytes=1,
        expected_copied_output_bytes=1,
        expected_log_bytes=1,
        expected_telemetry_bytes=1,
        plan_required_host_ram_bytes=1,
    )
    behavior = {
        "emergency_deadline_seconds": 600,
        "remaining_disk_budget_bytes": 10_000,
        "resource_budget_sha256": budget.sha256,
        "phase4_binding_sha256": sha256_bytes(compact_canonical_json_bytes(baseline)),
    }
    execution_identity = {
        "schema_version": SCHEMA_VERSIONS["execution_configuration"],
        "comparison_cell_id": slot["comparison_cell_id"],
        "exact_behavior_values": behavior,
        "split_seed": 424242,
        "training_seed": 17,
        "data_order_seed": 1000017,
        "plan_id": "plan_example",
        "candidate_id": "candidate_example",
        "bundle_fingerprint": DIGEST,
    }
    execution = {
        **execution_identity,
        "execution_configuration_id": deterministic_id("exec_", execution_identity),
        "emergency_deadline_seconds": 600,
    }
    baseline_digest = sha256_bytes(canonical_json_bytes(baseline))
    admission_decision = {"record_kind": "test-admission-decision"}
    authority_digest = "0" * 64
    run = {
        "schema_version": SCHEMA_VERSIONS["experiment_run"],
        "experiment_run_id": RUN_ID,
        "attempt_slot_id": slot["attempt_slot_id"],
        "execution_configuration_id": execution["execution_configuration_id"],
        "exact_argv": ["PENDING_MANAGED_TRAIN_SUBMISSION"],
        "working_directory": bundle_path,
        "fresh_state_root": state_root,
        "bundle_path": bundle_path,
        "output_path": str(Path(bundle_path) / "runs"),
        "run_order": {"block": 0, "position": 0},
        "observed_host_state": {
            "idle_baseline_sha256": baseline_digest,
            "admission_decision_sha256": sha256_bytes(
                compact_canonical_json_bytes(admission_decision)
            ),
            "current_authority_sha256": authority_digest,
            "resource_budget_sha256": budget.sha256,
        },
        "plan_id": execution["plan_id"],
        "candidate_id": execution["candidate_id"],
        "bundle_fingerprint": execution["bundle_fingerprint"],
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "archive_sha256": archive_sha256,
        "aptus_job_ids": [],
        "aptus_run_ids": [],
        "terminal_evidence": {"status": "pending"},
    }
    planned = PlannedSlotContext(
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
            training_seed=17,
            data_order_seed=1000017,
            emergency_deadline_seconds=600,
        ),
        run_proposal=RunProposal(
            working_directory=bundle_path,
            fresh_state_root=state_root,
            bundle_path=bundle_path,
            output_path=str(Path(bundle_path) / "runs"),
            bundle_manifest_sha256=bundle_manifest_sha256,
            archive_sha256=archive_sha256,
        ),
        phase4_binding=baseline,
        resource_budget=budget,
    )
    return QualifyingRunContext._for_test(
        dict(planned.campaign),
        dict(planned.comparison_cohort),
        dict(planned.comparison_cell),
        dict(planned.planned_attempt_slot),
        execution,
        run,
        dict(planned.phase4_binding),
    )


def telemetry_sample(slot: int) -> dict[str, object]:
    observed = slot * 1_000_000_000
    return construct_telemetry_sample(
        sequence=slot,
        experiment_run_id=RUN_ID,
        scheduled_slot=slot,
        scheduled_monotonic_ns=observed,
        observed_monotonic_ns=observed,
        wall_time_utc=WALL,
        probe_reading={
            "gpu": {
                "uuid": "GPU-private",
                "memory_used": {"value": str(GIB), "unit": "B"},
                "memory_free": {"value": str(7 * GIB), "unit": "B"},
                "memory_reserved": {"value": "0", "unit": "B"},
                "memory_total": {"value": str(8 * GIB), "unit": "B"},
                "utilization_percent": 0.0,
                "temperature_c": 35.0,
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
                "compute_processes": [],
            },
            "host": {
                "mem_available_bytes": 48 * GIB,
                "swap_used_bytes": 0,
                "swap_read_bytes": 0,
                "swap_write_bytes": 0,
                "load_1m": 0.25,
                "filesystem_free_bytes": 200 * GIB,
                "managed_process_rss_bytes": 256 * 1024**2,
                "managed_process_cpu_seconds": float(slot),
                "managed_process_read_bytes": slot,
                "managed_process_write_bytes": slot,
                "disk_growth_bytes": slot,
                "aptus_lease_active": False,
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
        },
        collector={
            "healthy": True,
            "status_code": None,
            "probe_duration_ns": 1,
        },
        watchdog={
            "healthy": True,
            "heartbeat_monotonic_ns": observed,
            "ownership_certain": True,
        },
    )


def qualifying_configuration(context: QualifyingRunContext) -> dict[str, object]:
    session = BackgroundTelemetrySession.qualifying_production(
        probe=lambda: {},
        ownership_certain=lambda: True,
        emergency_deadline_seconds=context.emergency_deadline_seconds,
        remaining_disk_budget_bytes=context.remaining_disk_budget_bytes,
        initial_thermal_limits_available=True,
        provider_name="linux-nvidia-host-probe",
        provider_version="v1",
        support_bindings={
            "cpu_temperature": "lm-sensors-v1",
            "gpu_thermal_limits": "nvidia-smi-v1",
            "hardware_events": "journalctl-v1",
            "nvidia_smi_binary": "sha256:" + "1" * 64,
            "nvme_temperature": "smartctl-v1",
            "xid_projection": "journalctl-nvrm-xid-v1",
        },
        ownership_binding="job-service-process-group-v1",
        disk_growth_binding="protected-roots-inventory-v1",
    )
    return session.configuration_record()


def passing_inputs(*, conditioning: bool = False) -> dict[str, object]:
    context = qualifying_context(role="conditioning" if conditioning else "anchor")
    job_count = 4 if conditioning else 5
    job_ids = ["job_" + f"{index:032x}" for index in range(1, job_count + 1)]
    actions = list(("dependency", "model-data", "preflight", "pilot", "train"))[
        :job_count
    ]
    levels = {
        "dependency": "dependency",
        "model-data": "model-data",
        "preflight": "measured-preflight",
        "pilot": "pilot",
    }
    run_id = "run_" + "5" * 32
    action_records = []
    for index, (job_id, action) in enumerate(zip(job_ids, actions)):
        command = (
            ["/usr/bin/python3", "validate.py", "--level", levels[action]]
            if action != "train"
            else [
                "/usr/bin/python3",
                "train.py",
                "--confirm-full-train",
                "--output-dir",
                f"runs/{run_id}",
            ]
        )
        action_records.append(
            {
                "id": job_id,
                "job_id": job_id,
                "action": action,
                "state": "completed",
                "return_code": 0,
                "run_id": run_id if action == "train" else None,
                "run_output_dir": (
                    f"/protected/bundle/runs/{run_id}" if action == "train" else None
                ),
                "bundle_dir": "/protected/bundle",
                "command": command,
                "artifact_fingerprint": DIGEST,
                "plan_id": "plan_example",
                "candidate_id": "candidate_example",
                "bundle_manifest_sha256": DIGEST,
                "monotonic_clock_binding": "linux-boot-sha256:" + "7" * 64,
                "queued_monotonic_ns": index * 100 + 1,
                "child_process_started_monotonic_ns": index * 100 + 2,
                "child_process_finished_monotonic_ns": index * 100 + 3,
                "terminal_monotonic_ns": index * 100 + 4,
            }
        )
    triples = (
        (
            ("pilot.phase-started", "pilot-phase-1", job_ids[3], 5),
            ("pilot.phase-finished", "pilot-phase-1", job_ids[3], 6),
            ("pilot.phase-started", "pilot-phase-2", job_ids[3], 7),
            ("pilot.phase-finished", "pilot-phase-2", job_ids[3], 8),
            ("training.started", "training", job_ids[4], 10),
            ("export.started", "final-export", job_ids[4], 11),
            ("export.finished", "final-export", job_ids[4], 12),
            ("training.finished", "training", job_ids[4], 13),
            ("verification.started", "parent-verification", job_ids[4], 14),
            ("verification.finished", "parent-verification", job_ids[4], 15),
        )
        if not conditioning
        else (
            ("pilot.phase-started", "pilot-phase-1", job_ids[3], 5),
            ("pilot.phase-finished", "pilot-phase-1", job_ids[3], 6),
            ("pilot.phase-started", "pilot-phase-2", job_ids[3], 7),
            ("pilot.phase-finished", "pilot-phase-2", job_ids[3], 8),
        )
    )
    boundaries = [
        RuntimeBoundary(
            experiment_run_id=RUN_ID,
            job_id=job_id,
            monotonic_ns=second * 1_000_000_000,
            wall_time_utc=WALL,
            event_type=event_type,
            phase=phase,
            action="pilot" if event_type.startswith("pilot.") else "train",
            native_outcome="passed" if event_type.endswith("finished") else None,
            reason_code="NONE",
        )
        for event_type, phase, job_id, second in triples
    ]
    ledger: list[dict[str, object]] = []
    for index, action in enumerate(actions):
        start = (1 + index * 3) * 1_000_000_000
        finish = start + 2_000_000_000
        ledger.extend(
            (
                {
                    "event_type": "command.started",
                    "monotonic_ns": start,
                    "phase": action,
                    "action": action,
                    "subject_kind": "managed-action",
                    "subject_id": action,
                },
                {
                    "event_type": "command.finished",
                    "monotonic_ns": finish,
                    "phase": action,
                    "action": action,
                    "subject_kind": "managed-action",
                    "subject_id": action,
                },
            )
        )
    ledger.extend(boundary.ledger_fields() for boundary in boundaries)
    ledger.extend(
        (
            {
                "event_type": "cooldown.started",
                "monotonic_ns": 20_000_000_000,
                "phase": "cooldown",
                "action": None,
                "subject_kind": "experiment-run",
                "subject_id": RUN_ID,
            },
            {
                "event_type": "cooldown.finished",
                "monotonic_ns": 139_000_000_000,
                "phase": "cooldown",
                "action": None,
                "subject_kind": "experiment-run",
                "subject_id": RUN_ID,
            },
        )
    )
    ledger.sort(key=lambda item: item["monotonic_ns"])
    samples = [telemetry_sample(slot) for slot in range(140)]
    cooldown = validate_cooldown(samples[20:140], context.idle_baseline_summary)
    return {
        "context": context,
        "action_records": action_records,
        "selected_artifact_roles": set(REQUIRED_QUALIFYING_ARTIFACT_ROLES)
        - ({"training-metrics", "final-export-manifest"} if conditioning else set()),
        "runtime_boundaries": boundaries,
        "telemetry_samples": samples,
        "telemetry_start_monotonic_ns": 0,
        "telemetry_stop_monotonic_ns": 139_000_000_000,
        "telemetry_configuration": qualifying_configuration(context),
        "safety_events": [],
        "cooldown": cooldown,
        "ledger_records": ledger,
    }


class QualifyingRunContextTests(unittest.TestCase):
    def test_conditioning_pass_is_complete_without_training_or_export(self) -> None:
        inputs = passing_inputs(conditioning=True)
        context = inputs["context"]

        decision = evaluate_passing_qualification(**inputs)  # type: ignore[arg-type]

        self.assertTrue(context.conditioning)  # type: ignore[union-attr]
        self.assertTrue(decision.valid)
        records = inputs["action_records"]
        slot, run = context.finalize_records(  # type: ignore[union-attr]
            records,  # type: ignore[arg-type]
            native_outcome="passed",
            evidence_status="protocol-valid",
            reason_code="NONE",
        )
        self.assertEqual(slot["role"], "conditioning")
        self.assertEqual(run["aptus_run_ids"], [])
        self.assertEqual(run["exact_argv"], records[-1]["command"])  # type: ignore[index]

    def test_action_records_are_cross_bound_to_context_paths_identity_and_argv(
        self,
    ) -> None:
        mutations = (
            (0, "bundle_dir", "/protected/other-bundle"),
            (0, "artifact_fingerprint", OTHER_DIGEST),
            (0, "plan_id", "plan_other"),
            (0, "candidate_id", "candidate_other"),
            (0, "bundle_manifest_sha256", OTHER_DIGEST),
            (1, "command", ["/tmp/python", "validate.py", "--level", "model-data"]),
            (4, "run_output_dir", "/protected/other/runs/run_bad"),
            (
                4,
                "command",
                [
                    "/usr/bin/python3",
                    "train.py",
                    "--confirm-full-train",
                    "--output-dir",
                    "runs/run_wrong",
                ],
            ),
        )
        for index, field, value in mutations:
            with self.subTest(field=field, index=index):
                inputs = passing_inputs()
                inputs["action_records"][index][field] = value
                decision = evaluate_passing_qualification(**inputs)  # type: ignore[arg-type]
                self.assertFalse(decision.valid)
                self.assertIn("MISSING_REQUIRED_EVIDENCE", decision.reason_codes)

    def test_context_binds_canonical_inputs_and_finalizes_terminal_records(
        self,
    ) -> None:
        context = qualifying_context()
        job_ids = ["job_" + f"{index:032x}" for index in range(1, 6)]
        records = passing_inputs()["action_records"]
        slot, run = context.finalize_records(
            records,
            native_outcome="passed",
            evidence_status="protocol-valid",
            reason_code="NONE",
        )
        self.assertEqual(slot["experiment_run_id"], RUN_ID)
        self.assertEqual(
            slot["execution_configuration_id"], run["execution_configuration_id"]
        )
        self.assertEqual(run["aptus_job_ids"], job_ids)
        self.assertEqual(run["aptus_run_ids"], ["run_" + "5" * 32])
        self.assertEqual(run["exact_argv"], records[-1]["command"])
        self.assertEqual(run["terminal_evidence"]["evidence_status"], "protocol-valid")
        self.assertEqual(len(context.source_bindings()), 7)

    def test_terminal_timing_allows_no_child_pair_and_rejects_other_boot(
        self,
    ) -> None:
        context = qualifying_context()
        records = passing_inputs()["action_records"]
        records[0]["child_process_started_monotonic_ns"] = None
        records[0]["child_process_finished_monotonic_ns"] = None
        _slot, run = context.finalize_records(
            records,
            native_outcome="passed",
            evidence_status="protocol-valid",
            reason_code="NONE",
        )
        first = run["terminal_evidence"]["jobs"][0]
        self.assertIsNone(first["child_runtime_duration_ns"])
        self.assertEqual(first["queue_to_terminal_duration_ns"], 3)

        records[1]["monotonic_clock_binding"] = "linux-boot-sha256:" + "9" * 64
        with self.assertRaisesRegex(QualificationError, "Phase-4 boot"):
            context.finalize_records(
                records,
                native_outcome="passed",
                evidence_status="protocol-valid",
                reason_code="NONE",
            )

    def test_idle_baseline_allows_unsupported_power_channel(self) -> None:
        context = qualifying_context()
        baseline = dict(context.idle_baseline_binding)
        summary = dict(baseline["summary"])
        summary["gpu_power_draw_p95_w"] = None
        baseline["summary"] = summary
        run = dict(context.experiment_run_template)
        observed = dict(run["observed_host_state"])
        observed["idle_baseline_sha256"] = sha256_bytes(canonical_json_bytes(baseline))
        run["observed_host_state"] = observed
        nonqualifying = QualifyingRunContext._for_test(
            dict(context.campaign),
            dict(context.comparison_cohort),
            dict(context.comparison_cell),
            dict(context.planned_attempt_slot),
            dict(context.execution_configuration),
            run,
            baseline,
        )
        self.assertIsNone(nonqualifying.idle_baseline_summary["gpu_power_draw_p95_w"])
        self.assertFalse(nonqualifying.production_qualifying)

    def test_terminal_digest_inventory_includes_every_selected_artifact(self) -> None:
        context = qualifying_context()
        records = passing_inputs()["action_records"]
        record_roles = {
            "attempt-slot-record",
            "execution-configuration-record",
            "idle-baseline-binding",
            "telemetry-configuration",
            "telemetry-summary",
            "cooldown-summary",
        }
        digests = {
            role: f"{index:064x}"
            for index, role in enumerate(
                sorted(
                    record_roles
                    | REQUIRED_QUALIFYING_ARTIFACT_ROLES
                    | REQUIRED_QUALIFYING_AUTHORITY_ROLES
                ),
                start=1,
            )
        }
        _slot, run = context.finalize_records(
            records,
            native_outcome="passed",
            evidence_status="protocol-valid",
            reason_code="NONE",
            evidence_role_sha256=digests,
        )
        self.assertEqual(run["terminal_evidence"]["evidence_role_sha256"], digests)

        missing = dict(digests)
        missing.pop("plan")
        with self.assertRaisesRegex(QualificationError, "digest roles"):
            context.finalize_records(
                records,
                native_outcome="passed",
                evidence_status="protocol-valid",
                reason_code="NONE",
                evidence_role_sha256=missing,
            )

    def test_context_rejects_mutated_static_binding_and_nonterminal_job(self) -> None:
        context = qualifying_context()
        mutated = dict(context.experiment_run_template)
        mutated["candidate_id"] = "candidate_other"
        with self.assertRaisesRegex(QualificationError, "candidate_id"):
            QualifyingRunContext._for_test(
                dict(context.campaign),
                dict(context.comparison_cohort),
                dict(context.comparison_cell),
                dict(context.planned_attempt_slot),
                dict(context.execution_configuration),
                mutated,
                dict(context.idle_baseline_binding),
            )
        with self.assertRaisesRegex(QualificationError, "state"):
            records = passing_inputs()["action_records"]
            records[-1]["state"] = "running"
            context.finalize_records(
                records,
                native_outcome="unknown",
                evidence_status="capture-invalid",
                reason_code="UNKNOWN_TERMINAL_STATE",
            )

    def test_passing_boundary_profile_is_exact_and_runtime_emitted(self) -> None:
        pilot_job = "job_" + "6" * 32
        train_job = "job_" + "7" * 32
        triples = (
            ("pilot.phase-started", "pilot-phase-1", pilot_job),
            ("pilot.phase-finished", "pilot-phase-1", pilot_job),
            ("pilot.phase-started", "pilot-phase-2", pilot_job),
            ("pilot.phase-finished", "pilot-phase-2", pilot_job),
            ("training.started", "training", train_job),
            ("export.started", "final-export", train_job),
            ("export.finished", "final-export", train_job),
            ("training.finished", "training", train_job),
            ("verification.started", "parent-verification", train_job),
            ("verification.finished", "parent-verification", train_job),
        )
        values = [
            RuntimeBoundary(
                experiment_run_id=RUN_ID,
                job_id=job_id,
                monotonic_ns=index,
                wall_time_utc=WALL,
                event_type=event_type,
                phase=phase,
                action="pilot" if event_type.startswith("pilot.") else "train",
                native_outcome="passed" if event_type.endswith("finished") else None,
                reason_code="NONE",
            )
            for index, (event_type, phase, job_id) in enumerate(triples, 1)
        ]
        validate_passing_runtime_boundaries(
            values, pilot_job_id=pilot_job, train_job_id=train_job
        )
        with self.assertRaisesRegex(QualificationError, "incomplete"):
            validate_passing_runtime_boundaries(
                values[:-1], pilot_job_id=pilot_job, train_job_id=train_job
            )

    def test_complete_passing_evidence_is_the_only_positive_decision(self) -> None:
        inputs = passing_inputs()
        context = inputs["context"]
        configuration = inputs["telemetry_configuration"]
        self.assertIsInstance(context, QualifyingRunContext)
        self.assertIsInstance(configuration, dict)
        validate_qualifying_telemetry_configuration(
            configuration,  # type: ignore[arg-type]
            context=context,  # type: ignore[arg-type]
        )
        decision = evaluate_passing_qualification(**inputs)  # type: ignore[arg-type]
        self.assertTrue(decision.valid, decision.reason_codes)
        self.assertEqual(decision.reason_codes, ())
        self.assertEqual(decision.telemetry_summary["coverage"], 1.0)
        self.assertEqual(decision.telemetry_summary["missing_scheduled_slots"], [])
        self.assertEqual(len(decision.segment_summaries), 11)

    def test_each_qualification_channel_fails_closed(self) -> None:
        mutations = []

        missing_role = passing_inputs()
        missing_role["selected_artifact_roles"] = {
            "plan",
            "bundle-manifest",
        }
        mutations.append(missing_role)

        reordered = passing_inputs()
        reordered["runtime_boundaries"] = list(
            reversed(reordered["runtime_boundaries"])  # type: ignore[arg-type]
        )
        mutations.append(reordered)

        telemetry_gap = passing_inputs()
        telemetry_gap["telemetry_samples"] = telemetry_gap["telemetry_samples"][  # type: ignore[index]
            :-2
        ]
        mutations.append(telemetry_gap)

        safety = passing_inputs()
        safety["safety_events"] = [
            {
                "level": "stop",
                "monotonic_ns": 1,
                "reason_code": "CUDA_XID",
            }
        ]
        mutations.append(safety)

        invalid_cooldown = passing_inputs()
        invalid_cooldown["cooldown"] = invalid_cooldown["cooldown"].__class__(  # type: ignore[union-attr]
            False, ("THERMAL_WARNING_SUSTAINED",)
        )
        mutations.append(invalid_cooldown)

        nonqualifying = passing_inputs()
        configuration = dict(nonqualifying["telemetry_configuration"])  # type: ignore[arg-type]
        configuration["profile"] = {
            "id": "custom-nonqualifying",
            "qualifying": False,
            "reason_code": "CUSTOM_OR_UNBOUND_TELEMETRY_PROFILE",
        }
        nonqualifying["telemetry_configuration"] = configuration
        mutations.append(nonqualifying)

        for inputs in mutations:
            with self.subTest(mutation=mutations.index(inputs)):
                decision = evaluate_passing_qualification(  # type: ignore[arg-type]
                    **inputs
                )
                self.assertFalse(decision.valid)
                self.assertTrue(decision.reason_codes)


if __name__ == "__main__":
    unittest.main()
