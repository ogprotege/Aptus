from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from tools.cuda_campaign import phase4 as phase4_module
from tools.cuda_campaign.contracts import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    compact_canonical_json_bytes,
    deterministic_id,
    sha256_bytes,
)
from tools.cuda_campaign.phase4 import (
    PHASE4_SOURCE_FREEZE_NAME,
    PHASE4_SOURCE_FREEZE_SEAL_NAME,
    Phase4SourceFreezeError,
    _create_phase4_source_freeze_for_test,
    _test_phase4_boundary,
    _validate_retained_phase4_source_freeze_for_test,
    _verify_phase4_source_freeze_for_test,
    validate_retained_phase4_source_freeze,
)
from tools.cuda_campaign.sidecar import BackgroundTelemetrySession
from tests.tools.test_cuda_campaign_qualification import (
    qualifying_context,
    telemetry_sample,
)


def private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def host_observation() -> dict[str, object]:
    return {
        "gpu_index": 0,
        "gpu_memory_total_bytes": 8 * 1024**3,
        "gpu_name": "NVIDIA GeForce RTX 3050",
        "gpu_thermal_limits": None,
        "gpu_thermal_limits_status": "unsupported",
        "gpu_thermal_limits_support_binding": (
            "unsupported:trusted-nvidia-help-query:" + "7" * 64
        ),
        "gpu_uuid_sha256": sha256_bytes(b"GPU-private"),
        "host_memory_total_bytes": 62 * 1024**3,
        "kernel_release": "6.8.0-test",
        "logical_cpu_count": 16,
        "machine_id_sha256": "8" * 64,
        "nvidia_driver_version": "595.84",
        "nvidia_smi_binding_sha256": "1" * 64,
    }


def phase4_records() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    context = qualifying_context()
    campaign = dict(context.campaign)
    cell_identity = dict(context.comparison_cell)
    cell_identity.pop("comparison_cell_id")
    cell_identity["host_binding"] = {
        "host_id": "host_" + "6" * 32,
        **host_observation(),
    }
    cell = {
        **cell_identity,
        "comparison_cell_id": deterministic_id("cell_", cell_identity),
    }
    cohort_identity = dict(context.comparison_cohort)
    cohort_identity.pop("comparison_cohort_id")
    cohort_identity["member_cell_ids"] = [cell["comparison_cell_id"]]
    cohort = {
        **cohort_identity,
        "comparison_cohort_id": deterministic_id("cohort_", cohort_identity),
    }
    return campaign, cohort, cell


def phase4_configuration() -> dict[str, object]:
    context = qualifying_context()
    session = BackgroundTelemetrySession.qualifying_production(
        probe=lambda: {},
        ownership_certain=lambda: True,
        emergency_deadline_seconds=context.emergency_deadline_seconds,
        remaining_disk_budget_bytes=context.remaining_disk_budget_bytes,
        initial_thermal_limits_available=False,
        provider_name="linux-nvidia-host-probe",
        provider_version="aptus-cuda-campaign-v1",
        support_bindings={
            "cpu_temperature": "unsupported:reviewed-not-configured",
            "gpu_thermal_limits": host_observation()[
                "gpu_thermal_limits_support_binding"
            ],
            "hardware_events": (
                "journalctl-current-boot-cursor-v1:"
                + "boot-sha256:"
                + "2" * 64
                + ":journalctl-sha256:"
                + "3" * 64
            ),
            "nvidia_smi_binary": "sha256:" + "1" * 64,
            "nvme_temperature": "unsupported:reviewed-not-configured",
            "xid_projection": (
                "journalctl-nvrm-xid-v1:"
                + "boot-sha256:"
                + "2" * 64
                + ":journalctl-sha256:"
                + "3" * 64
            ),
        },
        ownership_binding="factory-owned-job-service-process-group-v1",
        disk_growth_binding="factory-owned-statvfs-baseline-v1",
    )
    return session.configuration_record()


def baseline_path(
    root: Path,
    *,
    mutate: Callable[[list[dict[str, object]]], None] | None = None,
) -> Path:
    rows = [telemetry_sample(slot) for slot in range(600)]
    if mutate is not None:
        mutate(rows)
    path = root / "baseline.jsonl"
    path.write_bytes(canonical_jsonl_bytes(rows))
    path.chmod(0o600)
    return path


class Phase4SourceFreezeTests(unittest.TestCase):
    def test_exact_separate_seal_recomputes_600_sample_idle_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = private_directory(Path(temporary) / "root")
            repository = private_directory(root / "repository")
            campaign, cohort, cell = phase4_records()
            boundary = _test_phase4_boundary(
                source_binding=cell["source_binding"],
                host_observation=host_observation(),
            )
            result = _create_phase4_source_freeze_for_test(
                directory=root / "freeze",
                repository_root=repository,
                campaign=campaign,
                comparison_cohort=cohort,
                comparison_cell=cell,
                telemetry_configuration=phase4_configuration(),
                telemetry_samples_path=baseline_path(root),
                _trusted_boundary=boundary,
            )

            self.assertNotIn("record_kind", result.source_freeze)
            self.assertEqual(
                result.source_freeze["idle_baseline_summary"]["sample_count"], 600
            )
            self.assertEqual(
                result.source_freeze["idle_baseline_summary"][
                    "gpu_compute_process_nonempty_sample_count"
                ],
                0,
            )
            repeated = _verify_phase4_source_freeze_for_test(
                directory=result.directory,
                repository_root=repository,
                campaign=campaign,
                comparison_cohort=cohort,
                comparison_cell=cell,
                _trusted_boundary=boundary,
            )
            self.assertEqual(repeated.baseline_binding, result.baseline_binding)
            self.assertEqual(
                result.source_freeze["producer"]["name"],
                "aptus-cuda-campaign-phase4-test-fixture",
            )
            with self.assertRaisesRegex(Phase4SourceFreezeError, "misbound"):
                validate_retained_phase4_source_freeze(
                    source_freeze_bytes=(
                        result.directory / PHASE4_SOURCE_FREEZE_NAME
                    ).read_bytes(),
                    idle_samples_bytes=(
                        result.directory / "idle-baseline-samples.jsonl"
                    ).read_bytes(),
                    seal_bytes=(
                        result.directory / PHASE4_SOURCE_FREEZE_SEAL_NAME
                    ).read_bytes(),
                    campaign=campaign,
                    comparison_cohort=cohort,
                    comparison_cell=cell,
                )
            retained_test = _validate_retained_phase4_source_freeze_for_test(
                source_freeze_bytes=(
                    result.directory / PHASE4_SOURCE_FREEZE_NAME
                ).read_bytes(),
                idle_samples_bytes=(
                    result.directory / "idle-baseline-samples.jsonl"
                ).read_bytes(),
                seal_bytes=(
                    result.directory / PHASE4_SOURCE_FREEZE_SEAL_NAME
                ).read_bytes(),
                campaign=campaign,
                comparison_cohort=cohort,
                comparison_cell=cell,
            )
            self.assertEqual(retained_test.source_freeze, result.source_freeze)

    def test_imported_test_factory_can_never_mint_production_authority(self) -> None:
        self.assertFalse(hasattr(phase4_module, "_TEST_BOUNDARY_TOKEN"))
        with tempfile.TemporaryDirectory() as temporary:
            root = private_directory(Path(temporary) / "root")
            repository = private_directory(root / "repository")
            campaign, cohort, cell = phase4_records()
            boundary = _test_phase4_boundary(
                source_binding=cell["source_binding"],
                host_observation=host_observation(),
            )
            result = _create_phase4_source_freeze_for_test(
                directory=root / "freeze",
                repository_root=repository,
                campaign=campaign,
                comparison_cohort=cohort,
                comparison_cell=cell,
                telemetry_configuration=phase4_configuration(),
                telemetry_samples_path=baseline_path(root),
                _trusted_boundary=boundary,
            )
            self.assertEqual(
                result.source_freeze["producer"],
                {
                    "name": "aptus-cuda-campaign-phase4-test-fixture",
                    "version": "v1-nonproduction",
                },
            )
            with self.assertRaisesRegex(Phase4SourceFreezeError, "seal"):
                phase4_module.verify_phase4_source_freeze_artifact(
                    result.directory,
                    repository_root=repository,
                    campaign=campaign,
                    comparison_cohort=cohort,
                    comparison_cell=cell,
                )

    def test_declared_host_or_source_cannot_differ_from_trusted_boundary(self) -> None:
        for mutation in ("host", "source"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = private_directory(Path(temporary) / "root")
                repository = private_directory(root / "repository")
                campaign, cohort, cell = phase4_records()
                boundary_host = host_observation()
                boundary_source = dict(cell["source_binding"])
                if mutation == "host":
                    boundary_host["gpu_name"] = "NVIDIA H100 80GB HBM3"
                else:
                    boundary_source["commit"] = "0" * 40
                boundary = _test_phase4_boundary(
                    source_binding=boundary_source,
                    host_observation=boundary_host,
                )
                with self.assertRaises(Phase4SourceFreezeError):
                    _create_phase4_source_freeze_for_test(
                        directory=root / "freeze",
                        repository_root=repository,
                        campaign=campaign,
                        comparison_cohort=cohort,
                        comparison_cell=cell,
                        telemetry_configuration=phase4_configuration(),
                        telemetry_samples_path=baseline_path(root),
                        _trusted_boundary=boundary,
                    )

    def test_preexisting_xid_or_compute_process_blocks_idle_freeze(self) -> None:
        def xid(rows: list[dict[str, object]]) -> None:
            rows[0]["gpu"]["xid_errors"] = [79]  # type: ignore[index]

        def process(rows: list[dict[str, object]]) -> None:
            rows[0]["gpu"]["compute_processes"] = [  # type: ignore[index]
                {"pid": 7, "used_memory_bytes": 1024, "managed": True}
            ]

        for name, mutation in (("xid", xid), ("process", process)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = private_directory(Path(temporary) / "root")
                repository = private_directory(root / "repository")
                campaign, cohort, cell = phase4_records()
                boundary = _test_phase4_boundary(
                    source_binding=cell["source_binding"],
                    host_observation=host_observation(),
                )
                with self.assertRaises(Phase4SourceFreezeError):
                    _create_phase4_source_freeze_for_test(
                        directory=root / "freeze",
                        repository_root=repository,
                        campaign=campaign,
                        comparison_cohort=cohort,
                        comparison_cell=cell,
                        telemetry_configuration=phase4_configuration(),
                        telemetry_samples_path=baseline_path(root, mutate=mutation),
                        _trusted_boundary=boundary,
                    )

    def test_source_sample_symlink_and_hardlink_are_rejected(self) -> None:
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = private_directory(Path(temporary) / "root")
                repository = private_directory(root / "repository")
                campaign, cohort, cell = phase4_records()
                original = baseline_path(root)
                linked = root / "linked.jsonl"
                if kind == "symlink":
                    linked.symlink_to(original)
                else:
                    os.link(original, linked)
                boundary = _test_phase4_boundary(
                    source_binding=cell["source_binding"],
                    host_observation=host_observation(),
                )
                with self.assertRaises(Phase4SourceFreezeError):
                    _create_phase4_source_freeze_for_test(
                        directory=root / "freeze",
                        repository_root=repository,
                        campaign=campaign,
                        comparison_cohort=cohort,
                        comparison_cell=cell,
                        telemetry_configuration=phase4_configuration(),
                        telemetry_samples_path=linked,
                        _trusted_boundary=boundary,
                    )

    def test_whole_directory_replacement_during_verification_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = private_directory(Path(temporary) / "root")
            repository = private_directory(root / "repository")
            campaign, cohort, cell = phase4_records()
            boundary = _test_phase4_boundary(
                source_binding=cell["source_binding"],
                host_observation=host_observation(),
            )
            first = _create_phase4_source_freeze_for_test(
                directory=root / "freeze",
                repository_root=repository,
                campaign=campaign,
                comparison_cohort=cohort,
                comparison_cell=cell,
                telemetry_configuration=phase4_configuration(),
                telemetry_samples_path=baseline_path(root),
                _trusted_boundary=boundary,
            )
            second_baseline = root / "second-baseline.jsonl"
            second_baseline.write_bytes((root / "baseline.jsonl").read_bytes())
            second_baseline.chmod(0o600)
            second = _create_phase4_source_freeze_for_test(
                directory=root / "replacement",
                repository_root=repository,
                campaign=campaign,
                comparison_cohort=cohort,
                comparison_cell=cell,
                telemetry_configuration=phase4_configuration(),
                telemetry_samples_path=second_baseline,
                _trusted_boundary=boundary,
            )
            held = root / "held"
            real_open = os.open
            swapped = False

            def swap_then_open(
                path: object,
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                nonlocal swapped
                if (
                    path == PHASE4_SOURCE_FREEZE_NAME
                    and dir_fd is not None
                    and not swapped
                ):
                    swapped = True
                    first.directory.rename(held)
                    second.directory.rename(first.directory)
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with patch(
                "tools.cuda_campaign.phase4.os.open", side_effect=swap_then_open
            ):
                with self.assertRaisesRegex(
                    Phase4SourceFreezeError, "inventory changed|root changed"
                ):
                    _verify_phase4_source_freeze_for_test(
                        directory=first.directory,
                        repository_root=repository,
                        campaign=campaign,
                        comparison_cohort=cohort,
                        comparison_cell=cell,
                        _trusted_boundary=boundary,
                    )

    def test_resealed_summary_forgery_is_recomputed_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = private_directory(Path(temporary) / "root")
            repository = private_directory(root / "repository")
            campaign, cohort, cell = phase4_records()
            boundary = _test_phase4_boundary(
                source_binding=cell["source_binding"],
                host_observation=host_observation(),
            )
            result = _create_phase4_source_freeze_for_test(
                directory=root / "freeze",
                repository_root=repository,
                campaign=campaign,
                comparison_cohort=cohort,
                comparison_cell=cell,
                telemetry_configuration=phase4_configuration(),
                telemetry_samples_path=baseline_path(root),
                _trusted_boundary=boundary,
            )
            record_path = result.directory / PHASE4_SOURCE_FREEZE_NAME
            seal_path = result.directory / PHASE4_SOURCE_FREEZE_SEAL_NAME
            record = json.loads(record_path.read_bytes())
            record["idle_baseline_summary"]["gpu_temperature_median_c"] = 99
            record_bytes = canonical_json_bytes(record)
            record_path.write_bytes(record_bytes)
            record_path.chmod(0o600)
            seal = json.loads(seal_path.read_bytes())
            seal["source_freeze_sha256"] = sha256_bytes(record_bytes)
            seal["source_freeze_size_bytes"] = len(record_bytes)
            unsigned = dict(record["telemetry_configuration"])
            unsigned.pop("configuration_sha256")
            self.assertEqual(
                record["telemetry_configuration"]["configuration_sha256"],
                sha256_bytes(compact_canonical_json_bytes(unsigned)),
            )
            seal_path.write_bytes(canonical_json_bytes(seal))
            seal_path.chmod(0o600)

            with self.assertRaises(Phase4SourceFreezeError):
                validate_retained_phase4_source_freeze(
                    source_freeze_bytes=record_path.read_bytes(),
                    idle_samples_bytes=(
                        result.directory / "idle-baseline-samples.jsonl"
                    ).read_bytes(),
                    seal_bytes=seal_path.read_bytes(),
                    campaign=campaign,
                    comparison_cohort=cohort,
                    comparison_cell=cell,
                )

    def test_resealed_malformed_phase4_timestamps_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = private_directory(Path(temporary) / "root")
            repository = private_directory(root / "repository")
            campaign, cohort, cell = phase4_records()
            boundary = _test_phase4_boundary(
                source_binding=cell["source_binding"],
                host_observation=host_observation(),
            )
            with (
                patch(
                    "tools.cuda_campaign.phase4.utc_now",
                    return_value="2026-08-08T12:00:00Z",
                ),
                self.assertRaisesRegex(Phase4SourceFreezeError, "normalized RFC 3339"),
            ):
                _create_phase4_source_freeze_for_test(
                    directory=root / "invalid-clock",
                    repository_root=repository,
                    campaign=campaign,
                    comparison_cohort=cohort,
                    comparison_cell=cell,
                    telemetry_configuration=phase4_configuration(),
                    telemetry_samples_path=baseline_path(root),
                    _trusted_boundary=boundary,
                )
            for index, (target, value) in enumerate(
                (
                    ("record", "2026-02-30T12:00:00+00:00"),
                    ("seal", "2026-08-08T12:00:00Z"),
                )
            ):
                result = _create_phase4_source_freeze_for_test(
                    directory=root / f"freeze-{index}",
                    repository_root=repository,
                    campaign=campaign,
                    comparison_cohort=cohort,
                    comparison_cell=cell,
                    telemetry_configuration=phase4_configuration(),
                    telemetry_samples_path=baseline_path(root),
                    _trusted_boundary=boundary,
                )
                record_path = result.directory / PHASE4_SOURCE_FREEZE_NAME
                seal_path = result.directory / PHASE4_SOURCE_FREEZE_SEAL_NAME
                if target == "record":
                    record = json.loads(record_path.read_bytes())
                    record["created_at_utc"] = value
                    record_bytes = canonical_json_bytes(record)
                    record_path.write_bytes(record_bytes)
                    record_path.chmod(0o600)
                    seal = json.loads(seal_path.read_bytes())
                    seal["source_freeze_sha256"] = sha256_bytes(record_bytes)
                    seal["source_freeze_size_bytes"] = len(record_bytes)
                else:
                    seal = json.loads(seal_path.read_bytes())
                    seal["sealed_at_utc"] = value
                seal_path.write_bytes(canonical_json_bytes(seal))
                seal_path.chmod(0o600)

                with (
                    self.subTest(target=target),
                    self.assertRaisesRegex(
                        Phase4SourceFreezeError,
                        "calendar timestamp|normalized RFC 3339",
                    ),
                ):
                    _validate_retained_phase4_source_freeze_for_test(
                        source_freeze_bytes=record_path.read_bytes(),
                        idle_samples_bytes=(
                            result.directory / "idle-baseline-samples.jsonl"
                        ).read_bytes(),
                        seal_bytes=seal_path.read_bytes(),
                        campaign=campaign,
                        comparison_cohort=cohort,
                        comparison_cell=cell,
                    )


if __name__ == "__main__":
    unittest.main()
