from __future__ import annotations

import json
import os
import stat
import threading
import tempfile
import unittest
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.cuda_campaign import monitoring as monitoring_module
from tools.cuda_campaign.monitoring import (
    GIB,
    MIB,
    NANOSECONDS_PER_SECOND,
    SAMPLE_INTERVAL_SECONDS,
    FixedRateSampler,
    LinuxNvidiaHostProbe,
    LinuxNvidiaJournalEventProvider,
    ManagedProcessGroup,
    ProbeCommandResult,
    ProbeFailure,
    SafetyLimits,
    SafetyStateMachine,
    StatvfsDiskGrowthProvider,
    TelemetryValidationError,
    TrustedExecutable,
    construct_telemetry_sample,
    detect_nvidia_thermal_limit_authority,
    estimate_gpu_energy,
    exact_bytes,
    maximum_gap_seconds,
    summarize_telemetry,
    telemetry_coverage,
    type7_quantile,
    validate_cooldown,
    validate_idle_baseline,
    validate_telemetry_sample,
    resolve_trusted_nvidia_smi,
)


RUN_ID = "xrun_" + "a" * 32
REPOSITORY = Path(__file__).resolve().parents[2]


def _probe_reading(
    *,
    free_vram_bytes: int = 7 * GIB,
    temperature_c: float = 40.0,
    utilization_percent: float = 0.0,
    power_draw_w: float = 20.0,
    mem_available_bytes: int = 48 * GIB,
    filesystem_free_bytes: int = 200 * GIB,
    swap_read_bytes: int = 0,
    swap_write_bytes: int = 0,
    compute_processes: list[dict[str, object]] | None = None,
    throttle_reasons: list[str] | None = None,
    xid_errors: list[int] | None = None,
    reset_detected: bool = False,
    device_lost: bool = False,
    hardware_error: bool = False,
    aptus_lease_active: bool = False,
) -> dict[str, object]:
    total = 8 * GIB
    used = total - free_vram_bytes
    return {
        "gpu": {
            "uuid": "GPU-protected-test-uuid",
            "memory_used": {"value": str(used), "unit": "B"},
            "memory_free": {"value": str(free_vram_bytes), "unit": "B"},
            "memory_reserved": {"value": "0", "unit": "B"},
            "memory_total": {"value": str(total), "unit": "B"},
            "utilization_percent": utilization_percent,
            "temperature_c": temperature_c,
            "power_draw_w": power_draw_w,
            "power_limit_w": 130.0,
            "graphics_clock_mhz": 210.0,
            "memory_clock_mhz": 405.0,
            "performance_state": "P8",
            "throttle_reasons": throttle_reasons or [],
            "throttle_state": "Active" if throttle_reasons else "0x0000000000000000",
            "xid_errors": xid_errors or [],
            "reset_detected": reset_detected,
            "device_lost": device_lost,
            "hardware_error": hardware_error,
            "compute_processes": compute_processes or [],
        },
        "host": {
            "mem_available_bytes": mem_available_bytes,
            "swap_used_bytes": 0,
            "swap_read_bytes": swap_read_bytes,
            "swap_write_bytes": swap_write_bytes,
            "load_1m": 0.25,
            "filesystem_free_bytes": filesystem_free_bytes,
            "managed_process_rss_bytes": 256 * MIB,
            "managed_process_cpu_seconds": 1.5,
            "managed_process_read_bytes": 1024,
            "managed_process_write_bytes": 2048,
            "disk_growth_bytes": 4096,
            "aptus_lease_active": aptus_lease_active,
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


def _sample(
    slot: int,
    *,
    sequence: int | None = None,
    observed_monotonic_ns: int | None = None,
    heartbeat_age_ns: int = 0,
    collector_healthy: bool = True,
    watchdog_healthy: bool = True,
    ownership_certain: bool = True,
    **reading_options: object,
) -> dict[str, object]:
    scheduled = slot * NANOSECONDS_PER_SECOND
    observed = scheduled if observed_monotonic_ns is None else observed_monotonic_ns
    return construct_telemetry_sample(
        sequence=slot if sequence is None else sequence,
        experiment_run_id=RUN_ID,
        scheduled_slot=slot,
        scheduled_monotonic_ns=scheduled,
        observed_monotonic_ns=observed,
        wall_time_utc="2026-08-08T12:00:00+00:00",
        probe_reading=_probe_reading(**reading_options),
        collector={
            "healthy": collector_healthy,
            "status_code": None if collector_healthy else "COLLECTOR_FAILED",
            "probe_duration_ns": 1000,
        },
        watchdog={
            "healthy": watchdog_healthy,
            "heartbeat_monotonic_ns": observed - heartbeat_age_ns,
            "ownership_certain": ownership_certain,
        },
    )


class FakeClock:
    def __init__(self) -> None:
        self._now = 0
        self._lock = threading.Lock()

    def monotonic_ns(self) -> int:
        with self._lock:
            return self._now

    def sleep(self, seconds: float) -> None:
        self.advance(round(seconds * NANOSECONDS_PER_SECOND))

    def advance(self, nanoseconds: int) -> None:
        with self._lock:
            self._now += nanoseconds


class TelemetryConstructionTests(unittest.TestCase):
    def test_exact_decimal_conversion_and_memory_integrity(self) -> None:
        self.assertEqual(exact_bytes("1.5", "MiB"), 1_572_864)
        self.assertEqual(exact_bytes(2, "GiB"), 2 * GIB)
        with self.assertRaisesRegex(TelemetryValidationError, "convert exactly"):
            exact_bytes("0.1", "B")
        with self.assertRaisesRegex(TelemetryValidationError, "unsupported"):
            exact_bytes("1", "mystery")

        reading = _probe_reading()
        reading["gpu"]["memory_free"] = {"value": "3", "unit": "B"}
        with self.assertRaisesRegex(TelemetryValidationError, "do not reconcile"):
            construct_telemetry_sample(
                sequence=0,
                experiment_run_id=RUN_ID,
                scheduled_slot=0,
                scheduled_monotonic_ns=0,
                observed_monotonic_ns=0,
                wall_time_utc="2026-08-08T12:00:00+00:00",
                probe_reading=reading,
                collector={
                    "healthy": True,
                    "status_code": None,
                    "probe_duration_ns": 0,
                },
                watchdog={
                    "healthy": True,
                    "heartbeat_monotonic_ns": 0,
                    "ownership_certain": True,
                },
            )

    def test_memory_integrity_retains_reserved_and_bounds_display_rounding(self) -> None:
        reading = _probe_reading()
        reading["gpu"].update(
            {
                "memory_used": {"value": "156", "unit": "MiB"},
                "memory_free": {"value": "7684", "unit": "MiB"},
                "memory_reserved": {"value": "353", "unit": "MiB"},
                "memory_total": {"value": "8192", "unit": "MiB"},
            }
        )
        sample = construct_telemetry_sample(
            sequence=0,
            experiment_run_id=RUN_ID,
            scheduled_slot=0,
            scheduled_monotonic_ns=0,
            observed_monotonic_ns=0,
            wall_time_utc="2026-08-08T12:00:00+00:00",
            probe_reading=reading,
            collector={
                "healthy": True,
                "status_code": None,
                "probe_duration_ns": 0,
            },
            watchdog={
                "healthy": True,
                "heartbeat_monotonic_ns": 0,
                "ownership_certain": True,
            },
        )
        self.assertEqual(sample["gpu"]["memory"]["reserved"]["source_value"], "353")

        reading["gpu"]["memory_total"] = {"value": "8190", "unit": "MiB"}
        with self.assertRaisesRegex(TelemetryValidationError, "do not reconcile"):
            construct_telemetry_sample(
                sequence=0,
                experiment_run_id=RUN_ID,
                scheduled_slot=0,
                scheduled_monotonic_ns=0,
                observed_monotonic_ns=0,
                wall_time_utc="2026-08-08T12:00:00+00:00",
                probe_reading=reading,
                collector={
                    "healthy": True,
                    "status_code": None,
                    "probe_duration_ns": 0,
                },
                watchdog={
                    "healthy": True,
                    "heartbeat_monotonic_ns": 0,
                    "ownership_certain": True,
                },
            )

    def test_sample_construction_is_strict_at_every_layer(self) -> None:
        reading = _probe_reading()
        del reading["gpu"]["temperature_c"]
        with self.assertRaisesRegex(TelemetryValidationError, "missing temperature_c"):
            construct_telemetry_sample(
                sequence=0,
                experiment_run_id=RUN_ID,
                scheduled_slot=0,
                scheduled_monotonic_ns=0,
                observed_monotonic_ns=0,
                wall_time_utc="2026-08-08T12:00:00+00:00",
                probe_reading=reading,
                collector={
                    "healthy": True,
                    "status_code": None,
                    "probe_duration_ns": 0,
                },
                watchdog={
                    "healthy": True,
                    "heartbeat_monotonic_ns": 0,
                    "ownership_certain": True,
                },
            )

        sample = _sample(0)
        sample["private_extra"] = "/home/operator/secret"
        with self.assertRaisesRegex(TelemetryValidationError, "unexpected field"):
            validate_telemetry_sample(sample)

    def test_normalized_memory_tampering_is_rejected(self) -> None:
        sample = deepcopy(_sample(0))
        sample["gpu"]["memory"]["free"]["bytes"] += 1
        with self.assertRaises(TelemetryValidationError):
            validate_telemetry_sample(sample)


class FixedRateSamplerTests(unittest.TestCase):
    def test_elapsed_slots_are_skipped_without_catch_up(self) -> None:
        clock = FakeClock()
        calls = 0

        def probe() -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 1:
                clock.advance(2_200_000_000)
            return _probe_reading()

        sampler = FixedRateSampler(
            probe,
            lambda: {
                "healthy": True,
                "heartbeat_monotonic_ns": clock.monotonic_ns(),
                "ownership_certain": True,
            },
            probe_timeout_seconds=0.1,
            monotonic_ns=clock.monotonic_ns,
            sleep=clock.sleep,
            wall_time=lambda: "2026-08-08T12:00:00+00:00",
        )

        result = sampler.collect(
            experiment_run_id=RUN_ID,
            start_monotonic_ns=0,
            stop_monotonic_ns=4 * NANOSECONDS_PER_SECOND,
        )

        self.assertTrue(result.collector_healthy)
        self.assertEqual(result.expected_sample_count, 5)
        self.assertEqual([item["scheduled_slot"] for item in result.samples], [0, 3, 4])
        self.assertEqual(result.missed_slots, (1, 2))
        self.assertEqual(result.coverage, 0.6)
        self.assertEqual(calls, 3)

    def test_probe_failure_is_bounded_and_does_not_expose_exception(self) -> None:
        def failing_probe() -> dict[str, object]:
            raise RuntimeError("token=private-value /home/operator/model")

        sampler = FixedRateSampler(
            failing_probe,
            lambda: {
                "healthy": True,
                "heartbeat_monotonic_ns": 0,
                "ownership_certain": True,
            },
            probe_timeout_seconds=0.05,
            monotonic_ns=lambda: 0,
            sleep=lambda _seconds: None,
            wall_time=lambda: "2026-08-08T12:00:00+00:00",
        )
        result = sampler.collect(
            experiment_run_id=RUN_ID,
            start_monotonic_ns=0,
            stop_monotonic_ns=0,
        )

        self.assertFalse(result.collector_healthy)
        self.assertEqual(result.failure_code, "PROBE_FAILED")
        self.assertNotIn("private-value", repr(result))
        self.assertEqual(result.missed_slots, (0,))

    def test_probe_timeout_is_a_stable_failure_code(self) -> None:
        gate = threading.Event()
        sampler = FixedRateSampler(
            lambda: gate.wait(1),
            lambda: {
                "healthy": True,
                "heartbeat_monotonic_ns": 0,
                "ownership_certain": True,
            },
            probe_timeout_seconds=0.01,
            monotonic_ns=lambda: 0,
            sleep=lambda _seconds: None,
        )
        result = sampler.collect(
            experiment_run_id=RUN_ID,
            start_monotonic_ns=0,
            stop_monotonic_ns=0,
        )
        self.assertEqual(result.failure_code, "PROBE_TIMEOUT")


class LinuxNvidiaJournalEventProviderTests(unittest.TestCase):
    def test_imported_provider_internals_cannot_mint_production_authority(
        self,
    ) -> None:
        self.assertFalse(hasattr(monitoring_module, "_PRODUCTION_PROVIDER_TOKEN"))
        boot_id = "a" * 32
        row = json.dumps(
            {
                "_BOOT_ID": boot_id,
                "_TRANSPORT": "kernel",
                "MESSAGE": "Linux version test",
            }
        )
        provider = LinuxNvidiaJournalEventProvider._for_test(
            journalctl_path="/usr/bin/journalctl",
            boot_id=boot_id,
            command_runner=lambda _command, _timeout: ProbeCommandResult(
                0, row + "\n-- cursor: s=test\n"
            ),
        )
        self.assertFalse(provider.production_authorized)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authentic_disk = StatvfsDiskGrowthProvider.production(root)
            self.assertTrue(authentic_disk.production_authorized)
            authentic_disk._statvfs = lambda _path: os.statvfs(root)
            self.assertFalse(authentic_disk.production_authorized)
            forged_disk = object.__new__(StatvfsDiskGrowthProvider)
            forged_disk._initialize(root, statvfs=os.statvfs)
            self.assertFalse(forged_disk.production_authorized)

    def test_current_boot_cursor_projects_xid_and_hardware_events(self) -> None:
        boot_id = "a" * 32
        commands: list[tuple[str, ...]] = []

        def runner(command: Sequence[str], _timeout: float) -> ProbeCommandResult:
            exact = tuple(command)
            commands.append(exact)
            if len(commands) == 1:
                baseline = json.dumps(
                    {
                        "_BOOT_ID": boot_id,
                        "_TRANSPORT": "kernel",
                        "MESSAGE": "Linux version test",
                    },
                    separators=(",", ":"),
                )
                return ProbeCommandResult(0, baseline + "\n-- cursor: s=baseline\n")
            rows = (
                json.dumps(
                    {
                        "_BOOT_ID": boot_id,
                        "_TRANSPORT": "kernel",
                        "MESSAGE": (
                            "NVRM: Xid (PCI:0000:01:00): 79, "
                            "GPU has fallen off the bus."
                        ),
                    },
                    separators=(",", ":"),
                )
                + "\n-- cursor: s=next\n"
            )
            return ProbeCommandResult(0, rows)

        provider = LinuxNvidiaJournalEventProvider._for_test(
            journalctl_path="/usr/bin/journalctl",
            boot_id=boot_id,
            command_runner=runner,
        )

        self.assertEqual(
            provider.snapshot(),
            {
                "xid_errors": [79],
                "reset_detected": True,
                "device_lost": True,
                "hardware_error": True,
            },
        )
        self.assertIn("--boot=" + boot_id, commands[0])
        self.assertIn("--grep=(NVRM|nvidia|nouveau)", commands[0])
        self.assertIn("--after-cursor=s=baseline", commands[1])

    def test_empty_incremental_journal_query_accepts_cursor_only_status_one(
        self,
    ) -> None:
        boot_id = "a" * 32
        calls = 0

        def runner(_command: Sequence[str], _timeout: float) -> ProbeCommandResult:
            nonlocal calls
            calls += 1
            if calls == 1:
                baseline = json.dumps(
                    {
                        "_BOOT_ID": boot_id,
                        "_TRANSPORT": "kernel",
                        "MESSAGE": "Linux version test",
                    },
                    separators=(",", ":"),
                )
                return ProbeCommandResult(0, baseline + "\n-- cursor: s=baseline\n")
            return ProbeCommandResult(1, "-- cursor: s=baseline\n")

        provider = LinuxNvidiaJournalEventProvider._for_test(
            journalctl_path="/usr/bin/journalctl",
            boot_id=boot_id,
            command_runner=runner,
        )

        self.assertEqual(
            provider.snapshot(),
            {
                "xid_errors": [],
                "reset_detected": False,
                "device_lost": False,
                "hardware_error": False,
            },
        )

    def test_preconstruction_xid_remains_sticky_and_blocks_baseline(self) -> None:
        boot_id = "a" * 32
        calls = 0

        def runner(_command: Sequence[str], _timeout: float) -> ProbeCommandResult:
            nonlocal calls
            calls += 1
            message = (
                "NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus."
                if calls == 1
                else "nvidia: no new event"
            )
            row = json.dumps(
                {
                    "_BOOT_ID": boot_id,
                    "_TRANSPORT": "kernel",
                    "MESSAGE": message,
                },
                separators=(",", ":"),
            )
            return ProbeCommandResult(0, row + f"\n-- cursor: s={calls}\n")

        provider = LinuxNvidiaJournalEventProvider._for_test(
            journalctl_path="/usr/bin/journalctl",
            boot_id=boot_id,
            command_runner=runner,
        )

        self.assertEqual(provider.snapshot()["xid_errors"], [79])
        self.assertTrue(provider.snapshot()["device_lost"])

    def test_direct_construction_and_unbounded_or_wrong_boot_output_fail_closed(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            LinuxNvidiaJournalEventProvider(
                journalctl_path="/usr/bin/journalctl",
                boot_id="a" * 32,
            )
        with self.assertRaisesRegex(ProbeFailure, "JOURNAL_KERNEL_ACCESS_UNPROVEN"):
            LinuxNvidiaJournalEventProvider._for_test(
                journalctl_path="/usr/bin/journalctl",
                boot_id="a" * 32,
                command_runner=lambda _command, _timeout: ProbeCommandResult(
                    0, "-- cursor: s=baseline\n"
                ),
            )

        calls = 0

        def runner(_command: Sequence[str], _timeout: float) -> ProbeCommandResult:
            nonlocal calls
            calls += 1
            if calls == 1:
                baseline = json.dumps(
                    {
                        "_BOOT_ID": "a" * 32,
                        "_TRANSPORT": "kernel",
                        "MESSAGE": "Linux version test",
                    }
                )
                return ProbeCommandResult(0, baseline + "\n-- cursor: s=baseline\n")
            row = json.dumps(
                {
                    "_BOOT_ID": "b" * 32,
                    "_TRANSPORT": "kernel",
                    "MESSAGE": "NVRM: Xid (PCI:x): 48",
                }
            )
            return ProbeCommandResult(0, row + "\n-- cursor: s=next\n")

        provider = LinuxNvidiaJournalEventProvider._for_test(
            journalctl_path="/usr/bin/journalctl",
            boot_id="a" * 32,
            command_runner=runner,
        )
        with self.assertRaisesRegex(ProbeFailure, "JOURNAL_OUTPUT_INVALID"):
            provider.snapshot()


class TrustedNvidiaExecutableTests(unittest.TestCase):
    def test_thermal_limit_support_or_absence_is_trusted_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "nvidia-smi"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            metadata = executable.stat()
            trusted = TrustedExecutable(
                path=str(executable),
                device=metadata.st_dev,
                inode=metadata.st_ino,
                mode=metadata.st_mode,
                size=metadata.st_size,
                modified_ns=metadata.st_mtime_ns,
                binding_sha256="a" * 64,
            )

            def supported(
                command: Sequence[str], _timeout: float
            ) -> ProbeCommandResult:
                if "--help-query-gpu" in command:
                    return ProbeCommandResult(
                        0,
                        "temperature.gpu.max temperature.gpu.slowdown "
                        "temperature.gpu.shutdown temperature.gpu.target",
                    )
                return ProbeCommandResult(0, "92, 95, 100, 83\n")

            authority = detect_nvidia_thermal_limit_authority(
                trusted, gpu_index=0, command_runner=supported
            )
            self.assertEqual(authority.status, "supported")
            self.assertIsNotNone(authority.provider)
            self.assertEqual(
                authority.limits["target_temperature_c"],  # type: ignore[index]
                83.0,
            )

            unsupported = detect_nvidia_thermal_limit_authority(
                trusted,
                gpu_index=0,
                command_runner=lambda _command, _timeout: ProbeCommandResult(
                    0, "utilization.gpu memory.total"
                ),
            )
            self.assertEqual(unsupported.status, "unsupported")
            self.assertIsNone(unsupported.provider)
            self.assertRegex(
                unsupported.support_binding,
                r"^unsupported:trusted-nvidia-help-query:[0-9a-f]{64}$",
            )

            with self.assertRaisesRegex(
                ProbeFailure, "GPU_THERMAL_LIMIT_DISCOVERY_PARTIAL"
            ):
                detect_nvidia_thermal_limit_authority(
                    trusted,
                    gpu_index=0,
                    command_runner=lambda _command, _timeout: ProbeCommandResult(
                        0, "temperature.gpu.max"
                    ),
                )

    def test_rejects_missing_and_group_or_world_writable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "fake-nvidia-smi"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o777)
            with self.assertRaisesRegex(ProbeFailure, "NVIDIA_SMI_UNTRUSTED"):
                resolve_trusted_nvidia_smi(str(executable))
            with self.assertRaisesRegex(ProbeFailure, "NVIDIA_SMI_UNAVAILABLE"):
                resolve_trusted_nvidia_smi(str(executable.with_name("missing")))

    def test_linux_rejects_user_owned_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "fake-nvidia-smi"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o700)
            real_metadata = executable.stat()
            user_owned_metadata = SimpleNamespace(
                st_dev=real_metadata.st_dev,
                st_ino=real_metadata.st_ino,
                st_mode=stat.S_IFREG | 0o700,
                st_size=real_metadata.st_size,
                st_mtime_ns=real_metadata.st_mtime_ns,
                st_uid=max(1, os.getuid()),
            )
            with (
                patch("tools.cuda_campaign.monitoring.sys.platform", "linux"),
                patch.object(Path, "stat", return_value=user_owned_metadata),
                self.assertRaisesRegex(ProbeFailure, "NVIDIA_SMI_UNTRUSTED"),
            ):
                resolve_trusted_nvidia_smi(str(executable))

    def test_pinned_identity_detects_subsequent_file_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "nvidia-smi"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o700)
            with patch("tools.cuda_campaign.monitoring.sys.platform", "darwin"):
                trusted = resolve_trusted_nvidia_smi(str(executable))
            executable.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ProbeFailure, "NVIDIA_SMI_IDENTITY_CHANGED"):
                trusted.verify()


class LinuxNvidiaHostProbeTests(unittest.TestCase):
    GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def _proc_fixture(self, root: Path) -> None:
        (root / "meminfo").write_text(
            "MemAvailable: 50331648 kB\nSwapTotal: 2097152 kB\nSwapFree: 1048576 kB\n",
            encoding="utf-8",
        )
        (root / "vmstat").write_text("pswpin 10\npswpout 20\n", encoding="utf-8")
        (root / "loadavg").write_text("0.25 0.50 0.75 1/100 123\n", encoding="utf-8")
        process = root / "123"
        process.mkdir()
        (process / "statm").write_text("100 50 0 0 0 0 0\n", encoding="utf-8")
        after_name = [
            "S",
            "1",
            "1",
            "1",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "100",
            "50",
            "0",
            "0",
        ]
        (process / "stat").write_text(
            "123 (python worker) " + " ".join(after_name) + "\n",
            encoding="utf-8",
        )
        (process / "io").write_text(
            "read_bytes: 4096\nwrite_bytes: 8192\n",
            encoding="utf-8",
        )

    def _write_process(
        self,
        root: Path,
        *,
        pid: int,
        process_group_id: int,
        start_ticks: int,
        resident_pages: int,
        user_ticks: int,
        system_ticks: int,
        read_bytes: int,
        write_bytes: int,
    ) -> None:
        process = root / str(pid)
        process.mkdir(exist_ok=True)
        (process / "statm").write_text(
            f"100 {resident_pages} 0 0 0 0 0\n", encoding="utf-8"
        )
        after_name = [
            "S",
            "1",
            str(process_group_id),
            str(process_group_id),
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            str(user_ticks),
            str(system_ticks),
            "0",
            "0",
            "0",
            "0",
            "1",
            "0",
            str(start_ticks),
        ]
        (process / "stat").write_text(
            f"{pid} (python nested worker) " + " ".join(after_name) + "\n",
            encoding="utf-8",
        )
        (process / "io").write_text(
            f"read_bytes: {read_bytes}\nwrite_bytes: {write_bytes}\n",
            encoding="utf-8",
        )

    def _runner(
        self, commands: list[tuple[tuple[str, ...], float]], *, power: str = "20.50 W"
    ):
        gpu_row = ", ".join(
            (
                self.GPU_UUID,
                "1024 MiB",
                "7168 MiB",
                "0 MiB",
                "8192 MiB",
                "0 %",
                "40 C",
                power,
                "130.00 W",
                "210 MHz",
                "405 MHz",
                "P8",
                "0x0000000000000000",
            )
        )
        process_rows = f"123, {self.GPU_UUID}, 256 MiB\n999, {self.GPU_UUID}, 64 MiB\n"

        def run(command, timeout):
            commands.append((tuple(command), timeout))
            if any(str(item).startswith("--query-gpu=") for item in command):
                return ProbeCommandResult(0, gpu_row + "\n")
            return ProbeCommandResult(0, process_rows)

        return run

    def test_production_probe_captures_exact_gpu_proc_and_process_channels(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            self._proc_fixture(proc)
            commands: list[tuple[tuple[str, ...], float]] = []
            probe = LinuxNvidiaHostProbe(
                filesystem_path=root,
                managed_pids=lambda: {123},
                kernel_events=lambda: {
                    "xid_errors": [],
                    "reset_detected": False,
                    "device_lost": False,
                    "hardware_error": False,
                },
                lease_active=lambda: False,
                disk_growth_bytes=lambda: 4096,
                nvidia_smi_path="/usr/bin/nvidia-smi",
                command_runner=self._runner(commands),
                proc_root=proc,
                cpu_temperature=lambda: 45.5,
                nvme_temperature=lambda: None,
                page_size_bytes=4096,
                clock_ticks_per_second=100,
            )

            reading = probe()
            sample = construct_telemetry_sample(
                sequence=0,
                experiment_run_id=RUN_ID,
                scheduled_slot=0,
                scheduled_monotonic_ns=0,
                observed_monotonic_ns=0,
                wall_time_utc="2026-08-08T12:00:00+00:00",
                probe_reading=reading,
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

        self.assertEqual(sample["gpu"]["memory"]["free"]["source_unit"], "MiB")
        self.assertEqual(sample["gpu"]["memory"]["free"]["bytes"], 7 * GIB)
        processes = sample["gpu"]["compute_processes"]
        self.assertTrue(processes[0]["managed"])
        self.assertFalse(processes[1]["managed"])
        self.assertEqual(sample["host"]["managed_process_rss_bytes"], 50 * 4096)
        self.assertEqual(sample["host"]["managed_process_cpu_seconds"], 1.5)
        self.assertEqual(sample["host"]["swap_read_bytes"], 10 * 4096)
        self.assertEqual(sample["host"]["cpu_temperature"]["value"], 45.5)
        self.assertEqual(
            sample["host"]["nvme_temperature"],
            {
                "status": "unsupported",
                "value": None,
                "reason_code": "UNAVAILABLE_AT_FREEZE",
            },
        )
        self.assertEqual(len(commands), 2)
        self.assertTrue(all(timeout == 0.25 for _command, timeout in commands))
        arguments = {command[2] for command, _timeout in commands}
        self.assertTrue(
            any("clocks_event_reasons.active" in argument for argument in arguments)
        )
        self.assertIn(
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", arguments
        )

    def test_independent_probe_channels_run_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            self._proc_fixture(proc)
            rendezvous = threading.Barrier(5, timeout=1.0)
            commands: list[tuple[tuple[str, ...], float]] = []
            base_runner = self._runner(commands)

            def runner(command, timeout):
                rendezvous.wait()
                return base_runner(command, timeout)

            def kernel_events():
                rendezvous.wait()
                return {
                    "xid_errors": [],
                    "reset_detected": False,
                    "device_lost": False,
                    "hardware_error": False,
                }

            def thermal_limits():
                rendezvous.wait()
                return {
                    "maximum_operating_temperature_c": 92.0,
                    "slowdown_temperature_c": 94.0,
                    "shutdown_temperature_c": 97.0,
                    "target_temperature_c": 83.0,
                }

            def disk_growth():
                rendezvous.wait()
                return 4096

            probe = LinuxNvidiaHostProbe(
                filesystem_path=root,
                managed_pids=lambda: {123},
                kernel_events=kernel_events,
                lease_active=lambda: False,
                disk_growth_bytes=disk_growth,
                nvidia_smi_path="/usr/bin/nvidia-smi",
                command_runner=runner,
                proc_root=proc,
                gpu_thermal_limits=thermal_limits,
                page_size_bytes=4096,
                clock_ticks_per_second=100,
            )

            reading = probe()

        self.assertEqual(reading["gpu"]["temperature_c"], 40.0)
        self.assertEqual(reading["host"]["disk_growth_bytes"], 4096)
        self.assertEqual(len(commands), 2)

    def test_process_group_descendant_is_managed_and_unrelated_group_is_not(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            self._proc_fixture(proc)
            self._write_process(
                proc,
                pid=123,
                process_group_id=123,
                start_ticks=777,
                resident_pages=50,
                user_ticks=100,
                system_ticks=50,
                read_bytes=4096,
                write_bytes=8192,
            )
            self._write_process(
                proc,
                pid=456,
                process_group_id=123,
                start_ticks=778,
                resident_pages=25,
                user_ticks=30,
                system_ticks=20,
                read_bytes=1024,
                write_bytes=2048,
            )
            self._write_process(
                proc,
                pid=999,
                process_group_id=999,
                start_ticks=900,
                resident_pages=10,
                user_ticks=10,
                system_ticks=5,
                read_bytes=512,
                write_bytes=256,
            )
            gpu_row = ", ".join(
                (
                    self.GPU_UUID,
                    "1024 MiB",
                    "7168 MiB",
                    "0 MiB",
                    "8192 MiB",
                    "0 %",
                    "40 C",
                    "20.50 W",
                    "130.00 W",
                    "210 MHz",
                    "405 MHz",
                    "P8",
                    "0x0000000000000000",
                )
            )

            def runner(command, _timeout):
                if any(str(item).startswith("--query-gpu=") for item in command):
                    return ProbeCommandResult(0, gpu_row + "\n")
                return ProbeCommandResult(
                    0,
                    f"456, {self.GPU_UUID}, 128 MiB\n999, {self.GPU_UUID}, 64 MiB\n",
                )

            probe = LinuxNvidiaHostProbe(
                filesystem_path=root,
                managed_pids=lambda: {123},
                managed_process_groups=lambda: (
                    ManagedProcessGroup(123, 123, "linux-start-ticks:777"),
                ),
                xid_errors=lambda: [],
                hardware_events=lambda: {
                    "reset_detected": False,
                    "device_lost": False,
                    "hardware_error": False,
                },
                lease_active=lambda: True,
                disk_growth_bytes=lambda: 0,
                nvidia_smi_path="/usr/bin/nvidia-smi",
                command_runner=runner,
                proc_root=proc,
                page_size_bytes=4096,
                clock_ticks_per_second=100,
            )

            reading = probe()

        self.assertEqual(
            reading["gpu"]["compute_processes"],
            [
                {
                    "pid": 456,
                    "used_memory": {"value": "128", "unit": "MiB"},
                    "managed": True,
                },
                {
                    "pid": 999,
                    "used_memory": {"value": "64", "unit": "MiB"},
                    "managed": False,
                },
            ],
        )
        self.assertEqual(reading["host"]["managed_process_rss_bytes"], (50 + 25) * 4096)
        self.assertEqual(reading["host"]["managed_process_cpu_seconds"], 2.0)
        self.assertEqual(reading["host"]["managed_process_read_bytes"], 5120)
        self.assertEqual(reading["host"]["managed_process_write_bytes"], 10240)

    def test_process_group_discovery_ignores_unrelated_proc_churn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            self._proc_fixture(proc)
            self._write_process(
                proc,
                pid=123,
                process_group_id=123,
                start_ticks=777,
                resident_pages=50,
                user_ticks=100,
                system_ticks=50,
                read_bytes=4096,
                write_bytes=8192,
            )
            malformed = proc / "777"
            malformed.mkdir()
            malformed_stat = malformed / "stat"
            malformed_stat.write_text("departing process", encoding="utf-8")
            probe = LinuxNvidiaHostProbe(
                filesystem_path=root,
                managed_pids=lambda: {123},
                managed_process_groups=lambda: (
                    ManagedProcessGroup(123, 123, "linux-start-ticks:777"),
                ),
                xid_errors=lambda: [],
                hardware_events=lambda: {
                    "reset_detected": False,
                    "device_lost": False,
                    "hardware_error": False,
                },
                lease_active=lambda: True,
                disk_growth_bytes=lambda: 0,
                nvidia_smi_path="/usr/bin/nvidia-smi",
                command_runner=self._runner([]),
                proc_root=proc,
                page_size_bytes=4096,
                clock_ticks_per_second=100,
            )

            original_read_text = Path.read_text

            def read_text(path: Path, *args: object, **kwargs: object) -> str:
                value = original_read_text(path, *args, **kwargs)
                if path == malformed_stat:
                    malformed_stat.unlink()
                return value

            with patch.object(Path, "read_text", new=read_text):
                reading = probe()
            malformed_stat.write_text("persistently malformed", encoding="utf-8")
            persistent_reading = probe()

            def unavailable_read_text(
                path: Path, *args: object, **kwargs: object
            ) -> str:
                if path == malformed_stat:
                    raise PermissionError
                return original_read_text(path, *args, **kwargs)

            with patch.object(Path, "read_text", new=unavailable_read_text):
                unavailable_reading = probe()

        self.assertEqual(reading["host"]["managed_process_rss_bytes"], 50 * 4096)
        self.assertEqual(
            persistent_reading["host"]["managed_process_rss_bytes"], 50 * 4096
        )
        self.assertEqual(
            unavailable_reading["host"]["managed_process_rss_bytes"], 50 * 4096
        )

    def test_persistently_malformed_group_leader_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            self._proc_fixture(proc)
            probe = LinuxNvidiaHostProbe(
                filesystem_path=root,
                managed_pids=lambda: {123},
                managed_process_groups=lambda: (
                    ManagedProcessGroup(123, 123, "linux-start-ticks:777"),
                ),
                xid_errors=lambda: [],
                hardware_events=lambda: {
                    "reset_detected": False,
                    "device_lost": False,
                    "hardware_error": False,
                },
                lease_active=lambda: True,
                disk_growth_bytes=lambda: 0,
                nvidia_smi_path="/usr/bin/nvidia-smi",
                command_runner=self._runner([]),
                proc_root=proc,
                page_size_bytes=4096,
                clock_ticks_per_second=100,
            )
            (proc / "123" / "stat").write_text("malformed leader", encoding="utf-8")

            with self.assertRaisesRegex(
                ProbeFailure, "PROC_PROCESS_IDENTITY_INVALID"
            ):
                probe._managed_pid_set()

    def test_reused_process_group_leader_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            self._proc_fixture(proc)
            self._write_process(
                proc,
                pid=123,
                process_group_id=123,
                start_ticks=778,
                resident_pages=50,
                user_ticks=100,
                system_ticks=50,
                read_bytes=4096,
                write_bytes=8192,
            )
            probe = LinuxNvidiaHostProbe(
                filesystem_path=root,
                managed_pids=lambda: {123},
                managed_process_groups=lambda: (
                    ManagedProcessGroup(123, 123, "linux-start-ticks:777"),
                ),
                xid_errors=lambda: [],
                hardware_events=lambda: {
                    "reset_detected": False,
                    "device_lost": False,
                    "hardware_error": False,
                },
                lease_active=lambda: True,
                disk_growth_bytes=lambda: 0,
                nvidia_smi_path="/usr/bin/nvidia-smi",
                command_runner=self._runner([]),
                proc_root=proc,
                page_size_bytes=4096,
                clock_ticks_per_second=100,
            )

            with self.assertRaisesRegex(
                ProbeFailure, "MANAGED_PROCESS_GROUP_IDENTITY_LOST"
            ):
                probe()

    def test_process_group_exit_during_snapshot_is_not_identity_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            self._proc_fixture(proc)
            self._write_process(
                proc,
                pid=123,
                process_group_id=123,
                start_ticks=777,
                resident_pages=50,
                user_ticks=100,
                system_ticks=50,
                read_bytes=4096,
                write_bytes=8192,
            )
            probe = LinuxNvidiaHostProbe(
                filesystem_path=root,
                managed_pids=lambda: {123},
                managed_process_groups=lambda: (
                    ManagedProcessGroup(123, 123, "linux-start-ticks:777"),
                ),
                xid_errors=lambda: [],
                hardware_events=lambda: {
                    "reset_detected": False,
                    "device_lost": False,
                    "hardware_error": False,
                },
                lease_active=lambda: True,
                disk_growth_bytes=lambda: 0,
                nvidia_smi_path="/usr/bin/nvidia-smi",
                command_runner=self._runner([]),
                proc_root=proc,
                page_size_bytes=4096,
                clock_ticks_per_second=100,
            )
            leader_stat = proc / "123" / "stat"
            original_read_text = Path.read_text
            leader_reads = 0

            def read_text(path: Path, *args: object, **kwargs: object) -> str:
                nonlocal leader_reads
                value = original_read_text(path, *args, **kwargs)
                if path == leader_stat:
                    leader_reads += 1
                    if leader_reads == 2:
                        leader_stat.unlink()
                return value

            with patch.object(Path, "read_text", new=read_text):
                managed = probe._managed_pid_set()

        self.assertEqual(managed, set())

    def test_managed_process_exit_during_totals_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            self._proc_fixture(proc)
            self._write_process(
                proc,
                pid=123,
                process_group_id=123,
                start_ticks=777,
                resident_pages=50,
                user_ticks=100,
                system_ticks=50,
                read_bytes=4096,
                write_bytes=8192,
            )
            leader_io = proc / "123" / "io"
            probe = LinuxNvidiaHostProbe(
                filesystem_path=root,
                managed_pids=lambda: {123},
                managed_process_groups=lambda: (
                    ManagedProcessGroup(123, 123, "linux-start-ticks:777"),
                ),
                xid_errors=lambda: [],
                hardware_events=lambda: {
                    "reset_detected": False,
                    "device_lost": False,
                    "hardware_error": False,
                },
                lease_active=lambda: True,
                disk_growth_bytes=lambda: 0,
                nvidia_smi_path="/usr/bin/nvidia-smi",
                command_runner=self._runner([]),
                proc_root=proc,
                page_size_bytes=4096,
                clock_ticks_per_second=100,
            )
            original_read_text = Path.read_text

            def read_text(path: Path, *args: object, **kwargs: object) -> str:
                if path == leader_io:
                    raise FileNotFoundError
                return original_read_text(path, *args, **kwargs)

            with patch.object(Path, "read_text", new=read_text):
                reading = probe()

        self.assertEqual(reading["host"]["managed_process_rss_bytes"], 0)
        self.assertEqual(reading["host"]["managed_process_cpu_seconds"], 0.0)

    def test_terminal_managed_process_is_excluded_before_channel_validation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            self._proc_fixture(proc)
            self._write_process(
                proc,
                pid=123,
                process_group_id=123,
                start_ticks=777,
                resident_pages=50,
                user_ticks=100,
                system_ticks=50,
                read_bytes=4096,
                write_bytes=8192,
            )
            leader = proc / "123"
            stat_text = (leader / "stat").read_text(encoding="utf-8")
            (leader / "stat").write_text(
                stat_text.replace(
                    "(python nested worker) S ", "(python nested worker) Z "
                ),
                encoding="utf-8",
            )
            (leader / "statm").write_text("", encoding="utf-8")
            (leader / "io").write_text("", encoding="utf-8")
            probe = LinuxNvidiaHostProbe(
                filesystem_path=root,
                managed_pids=lambda: {123},
                managed_process_groups=lambda: (
                    ManagedProcessGroup(123, 123, "linux-start-ticks:777"),
                ),
                xid_errors=lambda: [],
                hardware_events=lambda: {
                    "reset_detected": False,
                    "device_lost": False,
                    "hardware_error": False,
                },
                lease_active=lambda: True,
                disk_growth_bytes=lambda: 0,
                nvidia_smi_path="/usr/bin/nvidia-smi",
                command_runner=self._runner([]),
                proc_root=proc,
                page_size_bytes=4096,
                clock_ticks_per_second=100,
            )

            reading = probe()

        self.assertEqual(reading["host"]["managed_process_rss_bytes"], 0)
        self.assertEqual(reading["host"]["managed_process_cpu_seconds"], 0.0)

    def test_transient_managed_process_permission_race_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            self._proc_fixture(proc)
            self._write_process(
                proc,
                pid=123,
                process_group_id=123,
                start_ticks=777,
                resident_pages=50,
                user_ticks=100,
                system_ticks=50,
                read_bytes=4096,
                write_bytes=8192,
            )
            leader_io = proc / "123" / "io"
            probe = LinuxNvidiaHostProbe(
                filesystem_path=root,
                managed_pids=lambda: {123},
                managed_process_groups=lambda: (
                    ManagedProcessGroup(123, 123, "linux-start-ticks:777"),
                ),
                xid_errors=lambda: [],
                hardware_events=lambda: {
                    "reset_detected": False,
                    "device_lost": False,
                    "hardware_error": False,
                },
                lease_active=lambda: True,
                disk_growth_bytes=lambda: 0,
                nvidia_smi_path="/usr/bin/nvidia-smi",
                command_runner=self._runner([]),
                proc_root=proc,
                page_size_bytes=4096,
                clock_ticks_per_second=100,
            )
            original_read_text = Path.read_text
            io_reads = 0

            def read_text(path: Path, *args: object, **kwargs: object) -> str:
                nonlocal io_reads
                if path == leader_io:
                    io_reads += 1
                    if io_reads == 1:
                        raise PermissionError
                return original_read_text(path, *args, **kwargs)

            with patch.object(Path, "read_text", new=read_text):
                reading = probe()

            def denied_read_text(
                path: Path, *args: object, **kwargs: object
            ) -> str:
                if path == leader_io:
                    raise PermissionError
                return original_read_text(path, *args, **kwargs)

            with (
                patch.object(Path, "read_text", new=denied_read_text),
                self.assertRaisesRegex(
                    ProbeFailure, "PROC_PROCESS_CHANNEL_INVALID"
                ),
            ):
                probe()

        self.assertEqual(reading["host"]["managed_process_rss_bytes"], 50 * 4096)
        self.assertEqual(io_reads, 2)

    def test_required_sensor_and_runner_failures_are_stable_and_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            self._proc_fixture(proc)
            commands: list[tuple[tuple[str, ...], float]] = []
            probe = LinuxNvidiaHostProbe(
                filesystem_path=root,
                managed_pids=lambda: {123},
                xid_errors=lambda: [],
                hardware_events=lambda: {
                    "reset_detected": False,
                    "device_lost": False,
                    "hardware_error": False,
                },
                lease_active=lambda: False,
                disk_growth_bytes=lambda: 0,
                nvidia_smi_path="/usr/bin/nvidia-smi",
                command_runner=self._runner(commands, power="N/A"),
                proc_root=proc,
                page_size_bytes=4096,
                clock_ticks_per_second=100,
            )
            with self.assertRaises(ProbeFailure) as unsupported:
                probe()
            self.assertEqual(
                unsupported.exception.code, "NVIDIA_POWER_FIELD_UNSUPPORTED"
            )

            def leaking_runner(_command, _timeout):
                raise RuntimeError("/home/operator token=private-value")

            failing = LinuxNvidiaHostProbe(
                filesystem_path=root,
                managed_pids=lambda: {123},
                xid_errors=lambda: [],
                hardware_events=lambda: {
                    "reset_detected": False,
                    "device_lost": False,
                    "hardware_error": False,
                },
                lease_active=lambda: False,
                disk_growth_bytes=lambda: 0,
                nvidia_smi_path="/usr/bin/nvidia-smi",
                command_runner=leaking_runner,
                proc_root=proc,
                page_size_bytes=4096,
                clock_ticks_per_second=100,
            )
            with self.assertRaises(ProbeFailure) as failed:
                failing()
            self.assertEqual(failed.exception.code, "NVIDIA_SMI_EXECUTION_FAILED")
            self.assertNotIn("private-value", str(failed.exception))

    def test_supported_gpu_thermal_limits_cannot_disappear_or_change(self) -> None:
        readings: list[dict[str, float | None] | None] = [
            {
                "maximum_operating_temperature_c": 92.0,
                "slowdown_temperature_c": 94.0,
                "shutdown_temperature_c": 97.0,
                "target_temperature_c": 83.0,
            },
            None,
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            self._proc_fixture(proc)
            commands: list[tuple[tuple[str, ...], float]] = []
            probe = LinuxNvidiaHostProbe(
                filesystem_path=root,
                managed_pids=lambda: {123},
                xid_errors=lambda: [],
                hardware_events=lambda: {
                    "reset_detected": False,
                    "device_lost": False,
                    "hardware_error": False,
                },
                lease_active=lambda: False,
                disk_growth_bytes=lambda: 0,
                nvidia_smi_path="/usr/bin/nvidia-smi",
                command_runner=self._runner(commands),
                proc_root=proc,
                gpu_thermal_limits=lambda: readings.pop(0),
                page_size_bytes=4096,
                clock_ticks_per_second=100,
            )
            probe()
            with self.assertRaises(ProbeFailure) as raised:
                probe()
        self.assertEqual(raised.exception.code, "GPU_THERMAL_LIMIT_DISAPPEARED")


class TelemetrySummaryTests(unittest.TestCase):
    def test_coverage_gap_type7_and_energy_follow_frozen_math(self) -> None:
        samples = [
            _sample(0, power_draw_w=10.0),
            _sample(1, power_draw_w=20.0),
            _sample(2, power_draw_w=30.0),
        ]
        stop = 2 * NANOSECONDS_PER_SECOND

        self.assertEqual(telemetry_coverage(samples, 0, stop), 1.0)
        self.assertEqual(maximum_gap_seconds(samples, 0, stop), 1.0)
        self.assertEqual(type7_quantile([0, 10, 20, 30], 0.95), 28.5)
        energy = estimate_gpu_energy(samples, 0, stop)
        self.assertIsNotNone(energy)
        self.assertEqual(energy["estimated_gpu_energy_joules"], 40.0)
        self.assertEqual(energy["integrated_covered_duration_ns"], stop)

        summary = summarize_telemetry(samples, 0, stop)
        self.assertEqual(summary["coverage"], 1.0)
        self.assertEqual(summary["channels"]["gpu_power_draw_w"]["p95"], 29.0)
        self.assertEqual(
            summary["channels"]["managed_process_cpu_seconds"]["coverage"], 1.0
        )
        self.assertEqual(
            summary["channels"]["cpu_temperature_c"]["status"], "unsupported"
        )

    def test_scalar_summary_preserves_large_integer_extrema_exactly(self) -> None:
        values = [2**60, 2**60 + 1, 2**60 + 2]
        summary = summarize_telemetry(
            [
                _sample(index, filesystem_free_bytes=value)
                for index, value in enumerate(values)
            ],
            0,
            2 * NANOSECONDS_PER_SECOND,
        )
        filesystem = summary["channels"]["filesystem_free_bytes"]
        self.assertEqual(filesystem["maximum"], 2**60 + 2)
        self.assertEqual(filesystem["minimum"], 2**60)

    def test_missing_slot_is_visible_and_blocks_energy_estimate(self) -> None:
        samples = [_sample(0), _sample(2, sequence=1)]
        stop = 2 * NANOSECONDS_PER_SECOND
        self.assertEqual(telemetry_coverage(samples, 0, stop), 2 / 3)
        self.assertEqual(maximum_gap_seconds(samples, 0, stop), 2.0)
        self.assertIsNone(estimate_gpu_energy(samples, 0, stop))

    def test_duplicate_slot_is_rejected(self) -> None:
        duplicate = deepcopy(_sample(0))
        duplicate["sequence"] = 1
        with self.assertRaisesRegex(TelemetryValidationError, "strictly increasing"):
            telemetry_coverage([_sample(0), duplicate], 0, NANOSECONDS_PER_SECOND)

    def test_window_rejects_multiple_experiment_runs(self) -> None:
        second = deepcopy(_sample(1))
        second["experiment_run_id"] = "xrun_" + "b" * 32
        with self.assertRaisesRegex(TelemetryValidationError, "multiple experiment"):
            telemetry_coverage([_sample(0), second], 0, NANOSECONDS_PER_SECOND)

    def test_window_rejects_noncontiguous_or_reordered_sequences(self) -> None:
        noncontiguous = _sample(1, sequence=2)
        with self.assertRaisesRegex(TelemetryValidationError, "ordered and contiguous"):
            telemetry_coverage([_sample(0), noncontiguous], 0, NANOSECONDS_PER_SECOND)
        with self.assertRaisesRegex(TelemetryValidationError, "ordered and contiguous"):
            telemetry_coverage(
                [_sample(1, sequence=1), _sample(0, sequence=0)],
                0,
                NANOSECONDS_PER_SECOND,
            )

    def test_window_rejects_nonincreasing_slots(self) -> None:
        repeated = _sample(0, sequence=1)
        with self.assertRaisesRegex(TelemetryValidationError, "strictly increasing"):
            telemetry_coverage([_sample(0), repeated], 0, NANOSECONDS_PER_SECOND)

    def test_window_rejects_scheduled_time_not_bound_to_window_slot(self) -> None:
        shifted = deepcopy(_sample(1))
        shifted["scheduled_monotonic_ns"] -= 1
        with self.assertRaisesRegex(TelemetryValidationError, "window slot"):
            telemetry_coverage([_sample(0), shifted], 0, NANOSECONDS_PER_SECOND)


def _limits(*, deadline: float = 10_000, budget: int = 0) -> SafetyLimits:
    return SafetyLimits.frozen_phase1(
        emergency_deadline_seconds=deadline,
        remaining_disk_budget_bytes=budget,
    )


class SafetyStateMachineTests(unittest.TestCase):
    def test_frozen_limits_match_the_machine_protocol(self) -> None:
        protocol = json.loads(
            (REPOSITORY / "docs/reference/cuda-campaign-protocol.v1.json").read_text(
                encoding="utf-8"
            )
        )
        safety = protocol["safety_contract"]
        telemetry = protocol["telemetry_contract"]
        limits = SafetyLimits.frozen_phase1(
            emergency_deadline_seconds=123,
            remaining_disk_budget_bytes=456,
        )
        thermal = safety["gpu_thermal"]
        vram = safety["vram"]
        host_memory = safety["host_memory"]
        disk = safety["disk"]
        swap = safety["swap_io"]
        gap = telemetry["gap_policy"]
        watchdog = safety["telemetry_and_watchdog"]
        expected = {
            "thermal_warning_c": thermal["warning_temperature_c"],
            "thermal_warning_seconds": thermal["warning_sustained_seconds"],
            "thermal_stop_c": thermal["hard_stop_temperature_c"],
            "thermal_stop_seconds": thermal["hard_stop_sustained_seconds"],
            "thermal_immediate_c": thermal["hard_stop_once_temperature_c"],
            "vram_warning_bytes": vram["warning_free_bytes_below"],
            "vram_warning_seconds": vram["warning_sustained_seconds"],
            "vram_stop_bytes": vram["hard_stop_free_bytes_below"],
            "vram_stop_seconds": vram["hard_stop_sustained_seconds"],
            "ram_warning_bytes": host_memory["warning_mem_available_bytes_below"],
            "ram_warning_seconds": host_memory["warning_sustained_seconds"],
            "ram_stop_bytes": host_memory["hard_stop_mem_available_bytes_below"],
            "ram_stop_seconds": host_memory["hard_stop_sustained_seconds"],
            "disk_warning_bytes": disk["warning_free_bytes_below"],
            "disk_warning_seconds": disk["warning_sustained_seconds"],
            "disk_stop_bytes": disk["hard_stop_free_bytes_below"],
            "swap_warning_bytes_per_second": swap["warning_bytes_per_second"],
            "swap_warning_seconds": swap["warning_sustained_seconds"],
            "swap_stop_bytes_per_second": swap["hard_stop_bytes_per_second"],
            "swap_stop_seconds": swap["hard_stop_sustained_seconds"],
            "swap_secondary_stop_bytes_per_second": swap[
                "secondary_hard_stop_bytes_per_second"
            ],
            "swap_secondary_stop_seconds": swap[
                "secondary_hard_stop_sustained_seconds"
            ],
            "qualifying_gap_seconds": gap["maximum_qualifying_gap_seconds"],
            "gap_warning_seconds": gap["warning_seconds_greater_than"],
            "gap_stop_seconds": gap["hard_stop_seconds_greater_than"],
            "heartbeat_warning_seconds": watchdog["heartbeat_warning_missing_seconds"],
            "heartbeat_stop_seconds": watchdog["heartbeat_cancel_by_seconds"],
        }
        for field_name, expected_value in expected.items():
            with self.subTest(field_name=field_name):
                self.assertEqual(getattr(limits, field_name), expected_value)
        self.assertEqual(SAMPLE_INTERVAL_SECONDS, telemetry["sample_interval_seconds"])

        fallback = SafetyLimits.frozen_phase1(
            emergency_deadline_seconds=123,
            remaining_disk_budget_bytes=456,
            initial_thermal_limits_available=False,
        )
        frozen_fallback = thermal["fallback_when_initial_limits_unavailable"]
        self.assertEqual(
            fallback.thermal_warning_c,
            frozen_fallback["warning_temperature_c"],
        )
        self.assertEqual(
            fallback.thermal_stop_c,
            frozen_fallback["stop_temperature_c"],
        )
        self.assertEqual(
            fallback.thermal_immediate_c,
            frozen_fallback["immediate_stop_temperature_c"],
        )

    def test_thermal_threshold_boundaries_are_sustained_and_exact(self) -> None:
        machine = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
        for second in range(5):
            self.assertEqual(
                machine.observe_sample(_sample(second, temperature_c=84)), ()
            )
        events = machine.observe_sample(_sample(5, temperature_c=84))
        self.assertEqual(events[0].reason_code, "THERMAL_STOP_SUSTAINED")
        self.assertEqual(machine.observe_sample(_sample(6, temperature_c=40)), ())

        immediate = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
        event = immediate.observe_sample(_sample(0, temperature_c=89))[0]
        self.assertEqual(event.reason_code, "THERMAL_STOP_IMMEDIATE")

        fallback = SafetyLimits.frozen_phase1(
            emergency_deadline_seconds=100,
            remaining_disk_budget_bytes=0,
            initial_thermal_limits_available=False,
        )
        self.assertEqual(fallback.thermal_warning_c, 75)
        self.assertEqual(fallback.thermal_stop_c, 82)
        self.assertEqual(fallback.thermal_immediate_c, 85)

    def test_vram_ram_disk_and_budget_stops(self) -> None:
        vram = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
        for second in range(6):
            events = vram.observe_sample(_sample(second, free_vram_bytes=2 * GIB - 1))
        self.assertEqual(events[0].reason_code, "FREE_VRAM_FLOOR")

        ram = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
        for second in range(6):
            events = ram.observe_sample(
                _sample(second, mem_available_bytes=8 * GIB - 1)
            )
        self.assertEqual(events[0].reason_code, "HOST_RAM_FLOOR")

        disk = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
        event = disk.observe_sample(_sample(0, filesystem_free_bytes=32 * GIB - 1))[0]
        self.assertEqual(event.reason_code, "DISK_FLOOR")

        budget = SafetyStateMachine(
            _limits(budget=10 * GIB), run_started_monotonic_ns=0
        )
        event = budget.observe_sample(_sample(0, filesystem_free_bytes=40 * GIB))[0]
        self.assertEqual(event.reason_code, "DISK_BUDGET_INSUFFICIENT")

    def test_all_sustained_warning_thresholds_activate_at_the_frozen_boundary(
        self,
    ) -> None:
        cases = (
            ({"temperature_c": 78}, 30, "THERMAL_WARNING_SUSTAINED"),
            ({"free_vram_bytes": 2 * GIB + 1}, 10, "FREE_VRAM_WARNING"),
            ({"mem_available_bytes": 8 * GIB + 1}, 30, "HOST_RAM_WARNING"),
            ({"filesystem_free_bytes": 40 * GIB}, 30, "DISK_WARNING"),
        )
        for options, duration, expected in cases:
            with self.subTest(expected=expected):
                machine = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
                observed: list[str] = []
                for second in range(duration + 1):
                    observed.extend(
                        event.reason_code
                        for event in machine.observe_sample(_sample(second, **options))
                    )
                self.assertIn(expected, observed)

    def test_swap_warning_and_both_stop_windows_use_at_least_boundary(self) -> None:
        warning = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
        observed_codes: list[str] = []
        for second in range(11):
            events = warning.observe_sample(
                _sample(second, swap_read_bytes=second * 16 * MIB)
            )
            observed_codes.extend(event.reason_code for event in events)
        self.assertIn("SWAP_RATE_WARNING", observed_codes)

        fast = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
        for second in range(11):
            events = fast.observe_sample(
                _sample(second, swap_read_bytes=second * 64 * MIB)
            )
        self.assertEqual(events[0].reason_code, "SWAP_RATE_LIMIT")

        burst = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
        burst_codes: list[str] = []
        for second in range(11):
            swapped = 0 if second == 0 else 64 * MIB
            burst_codes.extend(
                event.reason_code
                for event in burst.observe_sample(
                    _sample(second, swap_read_bytes=swapped)
                )
            )
        self.assertNotIn("SWAP_RATE_LIMIT", burst_codes)
        self.assertNotIn("SWAP_RATE_WARNING", burst_codes)

        sustained = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
        for second in range(61):
            events = sustained.observe_sample(
                _sample(second, swap_read_bytes=second * 16 * MIB)
            )
        self.assertEqual(events[0].reason_code, "SWAP_RATE_LIMIT")

    def test_gap_heartbeat_deadline_and_health_fail_closed(self) -> None:
        exact_gap = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
        exact_gap.observe_sample(_sample(0))
        events = exact_gap.observe_sample(
            _sample(5, observed_monotonic_ns=5 * NANOSECONDS_PER_SECOND)
        )
        self.assertEqual(events[0].reason_code, "TELEMETRY_QUALIFYING_GAP")

        warning_gap = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
        warning_gap.observe_sample(_sample(0))
        events = warning_gap.observe_sample(
            _sample(
                3,
                observed_monotonic_ns=3 * NANOSECONDS_PER_SECOND + 1,
            )
        )
        self.assertEqual(
            [(event.level, event.reason_code) for event in events],
            [
                ("capture-invalid", "TELEMETRY_QUALIFYING_GAP"),
                ("warning", "TELEMETRY_QUALIFYING_GAP"),
            ],
        )

        hard_gap = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
        hard_gap.observe_sample(_sample(0))
        event = hard_gap.observe_sample(
            _sample(
                5,
                observed_monotonic_ns=5 * NANOSECONDS_PER_SECOND + 1,
            )
        )[0]
        self.assertEqual(event.reason_code, "TELEMETRY_HARD_GAP")

        heartbeat = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
        event = heartbeat.observe_sample(
            _sample(5, heartbeat_age_ns=5 * NANOSECONDS_PER_SECOND)
        )[0]
        self.assertEqual(event.reason_code, "WATCHDOG_HEARTBEAT_LOST")

        deadline = SafetyStateMachine(_limits(deadline=5), run_started_monotonic_ns=0)
        event = deadline.observe_sample(_sample(5))[0]
        self.assertEqual(event.reason_code, "EMERGENCY_DEADLINE_EXCEEDED")

        collector = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
        event = collector.observe_health(
            now_monotonic_ns=0,
            collector_alive=False,
            watchdog_heartbeat_monotonic_ns=0,
            watchdog_alive=True,
            ownership_certain=True,
        )[0]
        self.assertEqual(event.reason_code, "TELEMETRY_COLLECTOR_FAILURE")

        ownership = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
        event = ownership.observe_sample(_sample(0, ownership_certain=False))[0]
        self.assertEqual(event.reason_code, "OWNERSHIP_UNCERTAIN")

    def test_immediate_hardware_and_foreign_process_signals(self) -> None:
        cases = (
            ({"xid_errors": [31]}, "CUDA_XID"),
            ({"reset_detected": True}, "CUDA_DEVICE_RESET"),
            ({"device_lost": True}, "CUDA_DEVICE_LOST"),
            ({"hardware_error": True}, "HARDWARE_ERROR"),
            ({"throttle_reasons": ["HW_THERMAL_SLOWDOWN"]}, "THERMAL_THROTTLE"),
            (
                {
                    "compute_processes": [
                        {
                            "pid": 123,
                            "used_memory": {"value": "1", "unit": "MiB"},
                            "managed": False,
                        }
                    ]
                },
                "UNRELATED_GPU_ACTIVITY",
            ),
        )
        for options, expected in cases:
            with self.subTest(expected=expected):
                machine = SafetyStateMachine(_limits(), run_started_monotonic_ns=0)
                event = machine.observe_sample(_sample(0, **options))[0]
                self.assertEqual(event.reason_code, expected)


class IdleAndCooldownTests(unittest.TestCase):
    def test_idle_baseline_and_cooldown_accept_a_complete_safe_window(self) -> None:
        baseline_samples = [_sample(slot) for slot in range(4)]
        baseline = validate_idle_baseline(baseline_samples, required_samples=4)
        self.assertTrue(baseline.valid)
        self.assertEqual(baseline.summary["gpu_temperature_median_c"], 40)

        cooldown_samples = [_sample(slot) for slot in range(120)]
        cooldown = validate_cooldown(cooldown_samples, baseline.summary)
        self.assertTrue(cooldown.valid, cooldown.reason_codes)
        self.assertEqual(cooldown.summary["zero_utilization_sample_count"], 120)
        self.assertLess(cooldown.summary["gpu_temperature_slope_c_per_minute"], 0.1)

    def test_cooldown_rejects_thermal_utilization_and_health_boundaries(self) -> None:
        baseline = validate_idle_baseline(
            [_sample(slot) for slot in range(4)], required_samples=4
        )
        samples = [
            _sample(
                slot,
                temperature_c=46 if slot == 119 else 40,
                utilization_percent=1 if slot < 11 else 0,
                collector_healthy=slot != 118,
            )
            for slot in range(120)
        ]
        result = validate_cooldown(samples, baseline.summary)
        self.assertFalse(result.valid)
        self.assertIn("THERMAL_WARNING_SUSTAINED", result.reason_codes)
        self.assertIn("UNRELATED_GPU_ACTIVITY", result.reason_codes)
        self.assertIn("TELEMETRY_COLLECTOR_FAILURE", result.reason_codes)

    def test_idle_window_rejects_missing_or_noncontiguous_samples(self) -> None:
        missing = validate_idle_baseline([_sample(0)], required_samples=2)
        self.assertEqual(missing.reason_codes, ("MISSING_REQUIRED_EVIDENCE",))

        second = _sample(2)
        second["sequence"] = 1
        noncontiguous = validate_idle_baseline([_sample(0), second], required_samples=2)
        self.assertIn("TELEMETRY_QUALIFYING_GAP", noncontiguous.reason_codes)


if __name__ == "__main__":
    unittest.main()
