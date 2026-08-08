from __future__ import annotations

import hashlib
import json
import threading
import time
import unittest
from pathlib import Path

from tools.cuda_campaign.harness import TelemetryCapture
from tools.cuda_campaign.monitoring import (
    GIB,
    LinuxNvidiaHostProbe,
    MIB,
    ProbeFailure,
    SafetyEvent,
    SafetyLimits,
)
from tools.cuda_campaign.sidecar import (
    BackgroundTelemetrySession,
    TelemetrySidecarError,
)


RUN_ID = "xrun_" + "d" * 32


def _probe_reading(*, temperature_c: float = 40.0) -> dict[str, object]:
    total = 8 * GIB
    free = 7 * GIB
    return {
        "gpu": {
            "uuid": "GPU-protected-sidecar-test",
            "memory_used": {"value": str(total - free), "unit": "B"},
            "memory_free": {"value": str(free), "unit": "B"},
            "memory_total": {"value": str(total), "unit": "B"},
            "utilization_percent": 25.0,
            "temperature_c": temperature_c,
            "power_draw_w": 35.0,
            "power_limit_w": 130.0,
            "graphics_clock_mhz": 900.0,
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
            "managed_process_rss_bytes": 256 * MIB,
            "managed_process_cpu_seconds": 1.5,
            "managed_process_read_bytes": 1024,
            "managed_process_write_bytes": 2048,
            "disk_growth_bytes": 4096,
            "aptus_lease_active": True,
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


class AcceleratedClock:
    def __init__(self, *, scale: float = 100.0) -> None:
        self._scale = scale
        self._origin = time.monotonic_ns()
        self._offset_ns = 0
        self._lock = threading.Lock()

    def monotonic_ns(self) -> int:
        elapsed_ns = time.monotonic_ns() - self._origin
        with self._lock:
            return int(elapsed_ns * self._scale) + self._offset_ns

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds / self._scale)

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._offset_ns += int(seconds * 1_000_000_000)


def _limits(*, deadline_seconds: float = 10_000.0) -> SafetyLimits:
    return SafetyLimits.frozen_phase1(
        emergency_deadline_seconds=deadline_seconds,
        remaining_disk_budget_bytes=0,
    )


def _session(
    clock: AcceleratedClock,
    *,
    probe: object | None = None,
    ownership_certain: object | None = None,
    watchdog_tick: object | None = None,
    safety_limits: SafetyLimits | None = None,
    join_timeout_seconds: float = 0.2,
) -> BackgroundTelemetrySession:
    return BackgroundTelemetrySession(
        probe=probe or _probe_reading,
        safety_limits=safety_limits or _limits(),
        ownership_certain=ownership_certain or (lambda: True),
        readiness_timeout_seconds=0.5,
        join_timeout_seconds=join_timeout_seconds,
        monotonic_ns=clock.monotonic_ns,
        wall_time=lambda: "2026-08-08T12:00:00+00:00",
        sleep=clock.sleep,
        watchdog_tick=watchdog_tick,
    )


def _qualifying_session(
    clock: AcceleratedClock,
    *,
    initial_thermal_limits_available: bool = True,
) -> BackgroundTelemetrySession:
    return BackgroundTelemetrySession.qualifying_production(
        probe=_probe_reading,
        ownership_certain=lambda: True,
        emergency_deadline_seconds=10_000.0,
        remaining_disk_budget_bytes=5 * GIB,
        initial_thermal_limits_available=initial_thermal_limits_available,
        provider_name="linux-nvidia-host-probe",
        provider_version="aptus-monitoring-1+nvidia-595.84",
        support_bindings={
            "cpu_temperature": "unsupported:NOT_CONFIGURED",
            "gpu_thermal_limits": "host-profile-binding:test",
            "hardware_events": "journal-cursor-binding:test",
            "nvidia_smi_binary": "sha256:" + "1" * 64,
            "nvme_temperature": "unsupported:NOT_CONFIGURED",
            "xid_projection": "journal-cursor-binding:test",
        },
        ownership_binding="job-service-process-group-binding:test",
        disk_growth_binding="capture-root-budget-binding:test",
        readiness_timeout_seconds=0.5,
        join_timeout_seconds=0.2,
        monotonic_ns=clock.monotonic_ns,
        wall_time=lambda: "2026-08-08T12:00:00+00:00",
        sleep=clock.sleep,
    )


def _wait_for_signal(
    session: BackgroundTelemetrySession,
    reason_code: str,
    *,
    timeout_seconds: float = 0.5,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        signal = session.safety_signal()
        if signal is not None and signal.reason_code == reason_code:
            return
        time.sleep(0.001)
    observed = session.safety_signal()
    raise AssertionError(f"expected {reason_code}, observed {observed!r}")


class BackgroundTelemetrySessionTests(unittest.TestCase):
    def test_exact_injected_probe_and_readable_authority_cannot_forge_runtime_auth(
        self,
    ) -> None:
        probe = LinuxNvidiaHostProbe(
            filesystem_path=Path.cwd(),
            managed_pids=lambda: (),
            managed_process_groups=lambda: (),
            kernel_events=lambda: {
                "xid_errors": [],
                "reset_detected": False,
                "device_lost": False,
                "hardware_error": False,
            },
            lease_active=lambda: False,
            disk_growth_bytes=lambda: 0,
        )
        session = BackgroundTelemetrySession.qualifying_production(
            probe=probe,
            ownership_certain=lambda: True,
            emergency_deadline_seconds=60.0,
            remaining_disk_budget_bytes=GIB,
            initial_thermal_limits_available=False,
            provider_name="linux-nvidia-host-probe",
            provider_version="aptus-cuda-campaign-v1",
            support_bindings={
                "cpu_temperature": "unsupported:reviewed-not-configured",
                "gpu_thermal_limits": "unsupported:test",
                "hardware_events": "journal:test",
                "nvidia_smi_binary": "sha256:" + "1" * 64,
                "nvme_temperature": "unsupported:reviewed-not-configured",
                "xid_projection": "journal:test",
            },
            ownership_binding="factory-owned-job-service-process-group-v1",
            disk_growth_binding="factory-owned-statvfs-baseline-v1",
        )
        readable_authority = object()
        session._qualifying_profile = True
        session._qualifying_authority = readable_authority
        self.assertFalse(session.qualifying_profile)
        self.assertFalse(session._authorized_for_harness(readable_authority))

    def test_direct_constructor_is_explicitly_nonqualifying(self) -> None:
        clock = AcceleratedClock()
        session = _session(clock)

        configuration = session.configuration_record()

        self.assertFalse(session.qualifying_profile)
        self.assertEqual(
            configuration["profile"],
            {
                "id": "custom-nonqualifying-test-only",
                "qualifying": False,
                "reason_code": "CUSTOM_OR_UNBOUND_TELEMETRY_PROFILE",
            },
        )
        self.assertEqual(
            configuration["thermal_policy"],
            {"initial_limits_available": None, "mode": "custom-unbound"},
        )

    def test_injected_frozen_factory_is_runtime_nonqualifying(self) -> None:
        clock = AcceleratedClock()
        session = _qualifying_session(
            clock,
            initial_thermal_limits_available=False,
        )
        configuration = session.configuration_record()
        digest = configuration.pop("configuration_sha256")
        canonical = json.dumps(
            configuration,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

        self.assertFalse(session.qualifying_profile)
        self.assertEqual(digest, hashlib.sha256(canonical).hexdigest())
        self.assertEqual(
            configuration["profile"],
            {
                "id": "phase1-frozen-qualifying",
                "qualifying": True,
                "reason_code": None,
            },
        )
        self.assertEqual(
            configuration["thermal_policy"],
            {
                "initial_limits_available": False,
                "mode": "frozen-conservative-fallback",
            },
        )
        self.assertEqual(
            configuration["safety_limits"]["emergency_deadline_seconds"],
            10_000.0,
        )
        self.assertEqual(
            configuration["safety_limits"]["remaining_disk_budget_bytes"],
            5 * GIB,
        )
        self.assertEqual(configuration["safety_limits"]["thermal_warning_c"], 75.0)
        self.assertEqual(configuration["safety_limits"]["thermal_stop_c"], 82.0)
        self.assertEqual(configuration["safety_limits"]["thermal_immediate_c"], 85.0)
        self.assertEqual(
            list(configuration["provenance"]["support_bindings"]),
            [
                "cpu_temperature",
                "gpu_thermal_limits",
                "hardware_events",
                "nvidia_smi_binary",
                "nvme_temperature",
                "xid_projection",
            ],
        )

    def test_qualifying_factory_rejects_missing_or_ambiguous_bindings(self) -> None:
        clock = AcceleratedClock()
        base: dict[str, object] = {
            "probe": _probe_reading,
            "ownership_certain": lambda: True,
            "emergency_deadline_seconds": 60.0,
            "remaining_disk_budget_bytes": 0,
            "initial_thermal_limits_available": True,
            "provider_name": "linux-nvidia-host-probe",
            "provider_version": "1",
            "support_bindings": {
                "cpu_temperature": "unsupported:NOT_CONFIGURED",
                "gpu_thermal_limits": "host-profile:test",
                "hardware_events": "journal:test",
                "nvidia_smi_binary": "sha256:" + "1" * 64,
                "nvme_temperature": "unsupported:NOT_CONFIGURED",
                "xid_projection": "journal:test",
            },
            "ownership_binding": "ownership:test",
            "disk_growth_binding": "disk-growth:test",
            "monotonic_ns": clock.monotonic_ns,
            "sleep": clock.sleep,
        }
        invalid = (
            ("provider_name", ""),
            ("provider_version", " version "),
            ("support_bindings", {}),
            ("support_bindings", {"Xid Projection": "journal:test"}),
            ("support_bindings", {"xid_projection": "journal:test"}),
            ("ownership_binding", ""),
            ("disk_growth_binding", "\n"),
            ("initial_thermal_limits_available", 1),
        )

        for name, value in invalid:
            with self.subTest(name=name, value=value):
                arguments = dict(base)
                arguments[name] = value
                with self.assertRaises(ValueError):
                    BackgroundTelemetrySession.qualifying_production(**arguments)

    def test_ready_stop_returns_healthy_capture_and_joins_threads(self) -> None:
        clock = AcceleratedClock()
        session = _session(clock)
        started_ns = clock.monotonic_ns()

        session.start(experiment_run_id=RUN_ID, start_monotonic_ns=started_ns)
        capture = session.stop(stop_monotonic_ns=clock.monotonic_ns())

        self.assertIsInstance(capture, TelemetryCapture)
        self.assertTrue(capture.healthy)
        self.assertIsNone(capture.failure_code)
        self.assertGreaterEqual(len(capture.samples), 1)
        self.assertEqual(capture.samples[0]["sequence"], 0)
        self.assertEqual(capture.samples[0]["experiment_run_id"], RUN_ID)
        self.assertEqual(capture.configuration, session.configuration_record())
        self.assertEqual(capture.safety_events, ())
        self.assertFalse(session.collector_alive)
        self.assertFalse(session.watchdog_alive)

    def test_capture_retains_every_safety_event_in_deterministic_order(self) -> None:
        clock = AcceleratedClock()
        session = _session(clock)
        session.start(
            experiment_run_id=RUN_ID,
            start_monotonic_ns=clock.monotonic_ns(),
        )
        observed_ns = clock.monotonic_ns()
        session._record_events(
            (
                SafetyEvent("stop", "THERMAL_STOP_IMMEDIATE", observed_ns + 2),
                SafetyEvent("warning", "DISK_WARNING", observed_ns + 2),
                SafetyEvent(
                    "capture-invalid",
                    "TELEMETRY_QUALIFYING_GAP",
                    observed_ns + 1,
                ),
            )
        )

        capture = session.stop(stop_monotonic_ns=clock.monotonic_ns())

        self.assertEqual(capture.failure_code, "TELEMETRY_QUALIFYING_GAP")
        self.assertEqual(
            capture.safety_events,
            (
                {
                    "level": "capture-invalid",
                    "monotonic_ns": observed_ns + 1,
                    "reason_code": "TELEMETRY_QUALIFYING_GAP",
                },
                {
                    "level": "warning",
                    "monotonic_ns": observed_ns + 2,
                    "reason_code": "DISK_WARNING",
                },
                {
                    "level": "stop",
                    "monotonic_ns": observed_ns + 2,
                    "reason_code": "THERMAL_STOP_IMMEDIATE",
                },
            ),
        )

    def test_snapshot_is_detached_and_does_not_stop_collection(self) -> None:
        clock = AcceleratedClock()
        session = _qualifying_session(clock)
        session.start(
            experiment_run_id=RUN_ID,
            start_monotonic_ns=clock.monotonic_ns(),
        )

        first = session.snapshot()
        first.samples[0]["gpu"]["uuid"] = "GPU-mutated-caller-copy"
        first.configuration["profile"]["qualifying"] = False
        second = session.snapshot()

        self.assertEqual(second.samples[0]["gpu"]["uuid"], "GPU-protected-sidecar-test")
        self.assertTrue(second.configuration["profile"]["qualifying"])
        self.assertTrue(session.collector_alive)
        session.stop(stop_monotonic_ns=clock.monotonic_ns())

    def test_rejects_non_exact_experiment_run_id(self) -> None:
        clock = AcceleratedClock()
        session = _session(clock)

        with self.assertRaisesRegex(TelemetrySidecarError, "EXPERIMENT_RUN_ID_INVALID"):
            session.start(
                experiment_run_id="xrun_" + "D" * 32,
                start_monotonic_ns=clock.monotonic_ns(),
            )

    def test_probe_death_sets_collector_signal_and_unhealthy_capture(self) -> None:
        clock = AcceleratedClock()
        calls = 0

        def probe() -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls > 1:
                raise ProbeFailure("PROBE_CHANNEL_DIED")
            return _probe_reading()

        session = _session(clock, probe=probe)
        session.start(
            experiment_run_id=RUN_ID,
            start_monotonic_ns=clock.monotonic_ns(),
        )

        _wait_for_signal(session, "TELEMETRY_COLLECTOR_FAILURE")
        capture = session.stop(stop_monotonic_ns=clock.monotonic_ns())
        self.assertFalse(capture.healthy)
        self.assertEqual(capture.failure_code, "PROBE_CHANNEL_DIED")

    def test_thermal_limit_disappearance_preserves_its_exact_stop_reason(self) -> None:
        clock = AcceleratedClock()
        calls = 0

        def probe() -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls > 1:
                raise ProbeFailure("GPU_THERMAL_LIMIT_DISAPPEARED")
            return _probe_reading()

        session = _session(clock, probe=probe)
        session.start(
            experiment_run_id=RUN_ID,
            start_monotonic_ns=clock.monotonic_ns(),
        )

        _wait_for_signal(session, "THERMAL_LIMIT_DISAPPEARED")
        capture = session.stop(stop_monotonic_ns=clock.monotonic_ns())
        self.assertFalse(capture.healthy)
        self.assertEqual(capture.failure_code, "GPU_THERMAL_LIMIT_DISAPPEARED")

    def test_watchdog_loss_sets_signal_and_unhealthy_capture(self) -> None:
        clock = AcceleratedClock()
        armed = threading.Event()

        def watchdog_tick() -> None:
            if armed.is_set():
                raise RuntimeError("private provider detail")

        session = _session(clock, watchdog_tick=watchdog_tick)
        session.start(
            experiment_run_id=RUN_ID,
            start_monotonic_ns=clock.monotonic_ns(),
        )
        armed.set()

        _wait_for_signal(session, "WATCHDOG_HEARTBEAT_LOST")
        capture = session.stop(stop_monotonic_ns=clock.monotonic_ns())
        self.assertFalse(capture.healthy)
        self.assertEqual(capture.failure_code, "WATCHDOG_THREAD_FAILED")

    def test_ownership_loss_sets_fail_closed_signal(self) -> None:
        clock = AcceleratedClock()
        ownership_lost = threading.Event()
        session = _session(
            clock,
            ownership_certain=lambda: not ownership_lost.is_set(),
        )
        session.start(
            experiment_run_id=RUN_ID,
            start_monotonic_ns=clock.monotonic_ns(),
        )
        ownership_lost.set()

        _wait_for_signal(session, "OWNERSHIP_UNCERTAIN")
        capture = session.stop(stop_monotonic_ns=clock.monotonic_ns())
        self.assertTrue(capture.healthy)

    def test_safety_threshold_is_exposed_without_poisoning_capture(self) -> None:
        clock = AcceleratedClock()
        session = _session(clock, probe=lambda: _probe_reading(temperature_c=90.0))
        session.start(
            experiment_run_id=RUN_ID,
            start_monotonic_ns=clock.monotonic_ns(),
        )

        _wait_for_signal(session, "THERMAL_STOP_IMMEDIATE")
        capture = session.stop(stop_monotonic_ns=clock.monotonic_ns())
        self.assertTrue(capture.healthy)

    def test_deadline_is_detected_by_independent_watchdog(self) -> None:
        clock = AcceleratedClock()
        session = _session(clock, safety_limits=_limits(deadline_seconds=0.5))
        session.start(
            experiment_run_id=RUN_ID,
            start_monotonic_ns=clock.monotonic_ns(),
        )

        _wait_for_signal(session, "EMERGENCY_DEADLINE_EXCEEDED")
        capture = session.stop(stop_monotonic_ns=clock.monotonic_ns())
        self.assertTrue(capture.healthy)

    def test_late_probe_skips_slots_instead_of_catching_up(self) -> None:
        clock = AcceleratedClock()
        calls = 0

        def probe() -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 1:
                clock.advance(2.2)
            return _probe_reading()

        session = _session(clock, probe=probe)
        session.start(
            experiment_run_id=RUN_ID,
            start_monotonic_ns=clock.monotonic_ns(),
        )
        deadline = time.monotonic() + 0.2
        while calls < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        capture = session.stop(stop_monotonic_ns=clock.monotonic_ns())

        slots = [sample["scheduled_slot"] for sample in capture.samples]
        self.assertGreaterEqual(len(slots), 2)
        self.assertEqual(slots[:2], [0, 3])
        self.assertEqual(slots, sorted(set(slots)))
        self.assertFalse(capture.healthy)
        self.assertEqual(capture.failure_code, "TELEMETRY_QUALIFYING_GAP")

    def test_invalid_later_sample_is_an_unhealthy_stable_failure(self) -> None:
        clock = AcceleratedClock()
        calls = 0

        def probe() -> dict[str, object]:
            nonlocal calls
            calls += 1
            reading = _probe_reading()
            if calls > 1:
                reading["unexpected"] = True
            return reading

        session = _session(clock, probe=probe)
        session.start(
            experiment_run_id=RUN_ID,
            start_monotonic_ns=clock.monotonic_ns(),
        )

        _wait_for_signal(session, "TELEMETRY_COLLECTOR_FAILURE")
        capture = session.stop(stop_monotonic_ns=clock.monotonic_ns())
        self.assertFalse(capture.healthy)
        self.assertEqual(capture.failure_code, "TELEMETRY_SAMPLE_INVALID")

    def test_lingering_collector_marks_capture_unhealthy(self) -> None:
        clock = AcceleratedClock()
        entered = threading.Event()
        release = threading.Event()
        calls = 0

        def probe() -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls > 1:
                entered.set()
                release.wait(1.0)
            return _probe_reading()

        session = _session(clock, probe=probe, join_timeout_seconds=0.01)
        session.start(
            experiment_run_id=RUN_ID,
            start_monotonic_ns=clock.monotonic_ns(),
        )
        self.assertTrue(entered.wait(0.2))
        capture = session.stop(stop_monotonic_ns=clock.monotonic_ns())
        release.set()

        self.assertFalse(capture.healthy)
        self.assertEqual(capture.failure_code, "COLLECTOR_THREAD_LINGERING")


if __name__ == "__main__":
    unittest.main()
