import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from aptus.api import (
    ApiContext,
    JobRequest,
    ProfileRequest,
    _resolve_static_dir,
    create_app,
)
from aptus.domain import Backend, ValidationReport, ValidationState
from aptus.execution import ActiveJobError, JobPrerequisiteError
from aptus.profiling import build_hardware_spec
from aptus.runtime_env import RuntimeInterpreter

from tests.aptus.helpers import make_plan

try:
    from fastapi.testclient import TestClient
except ImportError:  # The base package intentionally keeps the server optional.
    TestClient = None


class ApiContractTests(unittest.TestCase):
    def test_request_models_reject_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            ProfileRequest(dataset_path="data.jsonl", unknown=True)
        with self.assertRaises(ValidationError):
            JobRequest(
                bundle_dir="bundle",
                action="train",
                confirm_full_train=True,
                resume_from="checkpoint-1",
            )

    def test_plan_store_survives_context_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root)
            first = ApiContext(root / "state")
            first.save_plan(plan)
            restored = ApiContext(root / "state").load_plan(plan.plan_id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.plan_id, plan.plan_id)
        self.assertEqual(
            restored.recommended.candidate_id, plan.recommended.candidate_id
        )

    def test_invalid_plan_id_does_not_escape_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = ApiContext(Path(temporary) / "state")
            self.assertIsNone(context.load_plan("../../plan_secret"))

    def test_runtime_configuration_is_private_and_survives_context_restart(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            interpreter_path = str((root / "mlx-python").resolve())
            probe = RuntimeInterpreter(
                path=interpreter_path,
                source="configured:APTUS_MLX_PYTHON",
                python_version="3.12.9",
                runtimes={"mlx-lm": {"available": True}},
            )
            context = ApiContext(root / "state")
            with patch(
                "aptus.api.validate_runtime_configuration",
                return_value=probe,
            ):
                result = context.configure_runtime("mlx-lm", Path(interpreter_path))
            configuration_path = context.runtime_config_path
            configuration_mode = stat.S_IMODE(configuration_path.stat().st_mode)
            restarted = ApiContext(root / "state")

        self.assertEqual(result["interpreter_path"], interpreter_path)
        self.assertEqual(configuration_mode, 0o600)
        self.assertEqual(restarted.runtime_paths["mlx-lm"], interpreter_path)
        self.assertEqual(
            restarted.jobs.runtime_environment["APTUS_MLX_PYTHON"],
            interpreter_path,
        )

    def test_server_extra_failure_is_explicit_when_fastapi_absent(self) -> None:
        try:
            import fastapi  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(RuntimeError, "server"):
                create_app()

    def test_invalid_explicit_workbench_directory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "must contain index.html"):
                _resolve_static_dir(Path(temporary))

    def test_packaged_workbench_is_discoverable(self) -> None:
        workbench = _resolve_static_dir(None)
        self.assertIsNotNone(workbench)
        self.assertTrue((workbench / "index.html").is_file())

    @unittest.skipIf(
        TestClient is None,
        "Install the server and test extras for endpoint integration tests.",
    )
    def test_desktop_session_token_is_required_and_constant_time_checked(self) -> None:
        token = "desktop-session-token-that-is-long-enough"
        with tempfile.TemporaryDirectory() as temporary:
            client = TestClient(
                create_app(
                    state_dir=Path(temporary) / "state",
                    session_token=token,
                )
            )
            try:
                public_health = client.get("/api/v1/health")
                rejected = client.get("/api/v1/bootstrap")
                client.cookies.set("aptus_desktop_session", token)
                accepted = client.get("/api/v1/bootstrap")
            finally:
                client.close()
        self.assertEqual(public_health.status_code, 200)
        self.assertEqual(rejected.status_code, 403)
        self.assertEqual(rejected.json()["error"], "desktop_session_required")
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.json()["version"], "0.2.0")
        self.assertEqual(accepted.headers["x-content-type-options"], "nosniff")
        self.assertIn("default-src 'self'", accepted.headers["content-security-policy"])
        self.assertEqual(rejected.headers["x-frame-options"], "DENY")

    @unittest.skipIf(
        TestClient is None,
        "Install the server and test extras for endpoint integration tests.",
    )
    def test_server_session_partitions_public_static_from_protected_api(self) -> None:
        token = "server-session-token-that-is-long-enough"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            static = root / "web"
            static.mkdir()
            (static / "index.html").write_text("Aptus static", encoding="utf-8")
            client = TestClient(
                create_app(
                    state_dir=root / "state",
                    static_dir=static,
                    session_token=token,
                )
            )
            try:
                static_response = client.get("/")
                platform_response = client.get("/api/v1/platform")
                runtimes_response = client.get("/api/v1/runtimes")
                mutation_response = client.post(
                    "/api/v1/jobs",
                    json={"bundle_dir": "/tmp/bundle", "action": "pilot"},
                )
                bearer_response = client.get(
                    "/api/v1/bootstrap",
                    headers={"Authorization": f"Bearer {token}"},
                )
                exchange_response = client.get(
                    f"/?aptus_session_token={token}",
                    follow_redirects=False,
                )
                cookie_response = client.get("/api/v1/bootstrap")
            finally:
                client.close()

        self.assertEqual(static_response.status_code, 200)
        self.assertIn("Aptus static", static_response.text)
        for response in (platform_response, runtimes_response, mutation_response):
            self.assertEqual(response.status_code, 403)
            self.assertEqual(response.json()["error"], "desktop_session_required")
        self.assertEqual(bearer_response.status_code, 200)
        self.assertEqual(exchange_response.status_code, 303)
        self.assertEqual(exchange_response.headers["location"], "/")
        self.assertIn("HttpOnly", exchange_response.headers["set-cookie"])
        self.assertIn("SameSite=strict", exchange_response.headers["set-cookie"])
        self.assertEqual(cookie_response.status_code, 200)

    def test_desktop_session_token_rejects_short_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 32"):
            create_app(session_token="too-short")

    @unittest.skipIf(
        TestClient is None,
        "Install the server and test extras for endpoint integration tests.",
    )
    def test_desktop_server_enforces_the_no_execution_boundary(self) -> None:
        token = "desktop-session-token-that-is-long-enough"
        with tempfile.TemporaryDirectory() as temporary:
            client = TestClient(
                create_app(
                    state_dir=Path(temporary) / "state",
                    session_token=token,
                    execution_enabled=False,
                )
            )
            client.cookies.set("aptus_desktop_session", token)
            try:
                job_response = client.post(
                    "/api/v1/jobs",
                    json={"bundle_dir": "/tmp/untrusted-bundle", "action": "train"},
                )
                validation_response = client.post(
                    "/api/v1/validate",
                    json={
                        "bundle_dir": "/tmp/untrusted-bundle",
                        "level": "dependency",
                        "run": True,
                    },
                )
            finally:
                client.close()

        self.assertEqual(job_response.status_code, 403)
        self.assertEqual(
            job_response.json()["error"],
            "desktop_execution_disabled",
        )
        self.assertEqual(validation_response.status_code, 403)
        self.assertEqual(
            validation_response.json()["error"],
            "desktop_execution_disabled",
        )


@unittest.skipIf(
    TestClient is None,
    "Install the server and test extras for endpoint integration tests.",
)
class ApiEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "data.jsonl"
        self.dataset.write_text(
            '{"prompt":"Question?","completion":"Answer."}\n', encoding="utf-8"
        )
        static = self.root / "web"
        static.mkdir()
        (static / "index.html").write_text(
            "<html><body>Aptus workbench</body></html>", encoding="utf-8"
        )
        self.client = TestClient(
            create_app(state_dir=self.root / "state", static_dir=static)
        )

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def plan_payload(self) -> dict[str, object]:
        return {
            "model": {
                "model_id": "example/model-1b",
                "revision": "a" * 40,
                "family": "llama",
                "parameters_b": 1,
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "layers": 24,
                "context_length": 4096,
                "license_name": "apache-2.0",
                "training_allowed": True,
            },
            "hardware": {
                "discovery": "manual",
                "backend": "cuda",
                "gpu_count": 1,
                "vram_gib": 24,
                "free_vram_gib": 22,
                "supports_bf16": True,
                "supports_8bit": True,
                "supports_4bit": True,
                "host_ram_gib": 64,
                "host_ram_free_gib": 56,
                "reserve_gib": 2,
                "disk_free_gib": 500,
            },
            "target": {
                "objective": "memory",
                "sequence_length": 128,
                "effective_batch_size": 8,
                "max_epochs": 1,
                "task": "sft",
                "evaluation_fraction": 0.1,
                "packing": False,
                "checkpoint_steps": 10,
            },
            "dataset_path": str(self.dataset),
            "sample_limit": 64,
        }

    def test_health_spa_plan_compile_and_static_validation(self) -> None:
        self.assertEqual(self.client.get("/api/v1/health").json()["status"], "ok")
        self.assertIn("Aptus workbench", self.client.get("/").text)
        capabilities = self.client.get("/api/v1/bootstrap").json()["capabilities"]
        self.assertEqual(capabilities["backends"], ["cuda", "mps"])
        self.assertEqual(capabilities["supported_execution_backends"], ["cuda", "mps"])
        self.assertEqual(
            capabilities["supported_execution_backend"],
            "mps" if sys.platform == "darwin" else "cuda",
        )
        defaults = self.client.get("/api/v1/bootstrap").json()["defaults"]
        self.assertEqual(
            defaults["backend"], capabilities["supported_execution_backend"]
        )
        self.assertEqual(
            defaults["training_runtime"],
            "mlx-lm" if sys.platform == "darwin" else "transformers-peft-cuda",
        )
        self.assertEqual(
            defaults["reserve_gib"], 8.0 if sys.platform == "darwin" else 2.0
        )
        self.assertEqual(
            capabilities["training_runtimes"],
            ["transformers-peft-cuda", "mlx-lm"],
        )
        self.assertEqual(
            set(capabilities["known_backends"]), {"cuda", "rocm", "mps", "cpu"}
        )
        self.assertEqual(
            set(capabilities["methods"]), {"full", "lora", "int8-lora", "qlora"}
        )
        method_catalog = {
            item["method_id"]: item for item in capabilities["method_catalog"]
        }
        self.assertEqual(method_catalog["lora"]["lifecycle"], "gated-executable")
        self.assertTrue(method_catalog["lora"]["selectable"])
        self.assertEqual(method_catalog["dora"]["lifecycle"], "experimental")
        self.assertFalse(method_catalog["dora"]["selectable"])
        self.assertEqual(method_catalog["bitfit"]["lifecycle"], "experimental")
        self.assertEqual(method_catalog["loreft"]["lifecycle"], "research-only")

        planned = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(planned.status_code, 200, planned.text)
        plan_id = planned.json()["plan_id"]

        bundle = self.root / "bundle"
        compiled = self.client.post(
            "/api/v1/compile",
            json={"plan_id": plan_id, "output_dir": str(bundle)},
        )
        self.assertEqual(compiled.status_code, 200, compiled.text)
        self.assertTrue(compiled.json()["archive_path"].endswith("bundle.zip"))

        validated = self.client.post(
            "/api/v1/validate",
            json={"bundle_dir": str(bundle), "level": "static", "run": False},
        )
        self.assertEqual(validated.status_code, 200, validated.text)
        self.assertEqual(validated.json()["state"], "static-pass")

        restored = self.client.get("/api/v1/bootstrap")
        self.assertEqual(restored.status_code, 200, restored.text)
        self.assertEqual(restored.json()["plan"]["plan_id"], plan_id)
        self.assertEqual(restored.json()["bundle"]["bundle_dir"], str(bundle.resolve()))
        self.assertEqual(restored.json()["bundle"]["report"]["state"], "static-pass")

        runtime = self.client.post(
            "/api/v1/validate",
            json={"bundle_dir": str(bundle), "level": "pilot", "run": True},
        )
        self.assertEqual(runtime.status_code, 409)
        self.assertEqual(runtime.json()["suggested_action"], "pilot")

        conflict = self.client.post(
            "/api/v1/compile",
            json={"plan_id": plan_id, "output_dir": str(bundle)},
        )
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.json()["error"], "path_conflict")

    def test_expected_missing_path_and_no_fit_errors_are_typed(self) -> None:
        missing = self.client.post(
            "/api/v1/profile",
            json={"dataset_path": str(self.root / "missing.jsonl")},
        )
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"], "path_not_found")

        payload = self.plan_payload()
        payload["model"] = {
            **payload["model"],
            "parameters_b": 70,
            "hidden_size": 8192,
            "intermediate_size": 28672,
            "layers": 80,
        }
        payload["hardware"] = {
            **payload["hardware"],
            "vram_gib": 4,
            "free_vram_gib": 4,
            "reserve_gib": 1,
            "host_ram_gib": 512,
            "host_ram_free_gib": 512,
            "disk_free_gib": 2000,
        }
        no_fit = self.client.post("/api/v1/plan", json=payload)
        self.assertEqual(no_fit.status_code, 422, no_fit.text)
        self.assertEqual(no_fit.json()["error"], "no_feasible_plan")
        self.assertEqual(len(no_fit.json()["candidates"]), 12)

    def test_hardware_probe_runtime_failure_returns_manual_fallback(self) -> None:
        with patch(
            "aptus.api.probe_local_hardware",
            side_effect=RuntimeError("driver unavailable"),
        ):
            response = self.client.get("/api/v1/hardware")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unavailable")
        self.assertTrue(response.json()["manual_facts_supported"])

    def test_darwin_local_scan_enforces_unified_memory_reserve(self) -> None:
        payload = self.plan_payload()
        payload["hardware"] = {
            **payload["hardware"],
            "discovery": "local-scan",
            "reserve_gib": 2,
        }
        apple_hardware = build_hardware_spec(
            backend=Backend.MPS,
            gpu_count=1,
            vram_gib=64,
            supports_bf16=True,
            supports_4bit=False,
            supports_8bit=False,
            host_ram_gib=64,
            host_ram_free_gib=48,
            reserve_gib=8,
            disk_free_gib=500,
        )
        with (
            patch("aptus.api.sys.platform", "darwin"),
            patch(
                "aptus.api.probe_local_hardware",
                return_value=apple_hardware,
            ) as probe,
        ):
            response = self.client.post("/api/v1/plan", json=payload)

        self.assertEqual(response.status_code, 200, response.text)
        probe.assert_called_once_with(reserve_gib=8.0)
        self.assertEqual(
            response.json()["hardware"]["reserve_per_device_bytes"],
            8 * 1024**3,
        )

    def test_untrusted_host_header_is_rejected(self) -> None:
        response = self.client.get(
            "/api/v1/health", headers={"host": "attacker.example"}
        )
        self.assertEqual(response.status_code, 400, response.text)

    def test_job_submission_conflict_is_typed_as_http_409(self) -> None:
        with patch.object(
            self.client.app.state.aptus.jobs,
            "submit",
            side_effect=ActiveJobError("one local GPU job is already active"),
        ):
            response = self.client.post(
                "/api/v1/jobs",
                json={"bundle_dir": str(self.root / "bundle"), "action": "pilot"},
            )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"], "active_job_conflict")

    def test_job_prerequisite_failure_is_typed_as_http_409(self) -> None:
        with patch.object(
            self.client.app.state.aptus.jobs,
            "submit",
            side_effect=JobPrerequisiteError(
                action="pilot",
                required_state="measured-preflight-pass",
                current_state="model-data-pass",
                reason="insufficient_state",
            ),
        ):
            response = self.client.post(
                "/api/v1/jobs",
                json={"bundle_dir": str(self.root / "bundle"), "action": "pilot"},
            )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"], "job_prerequisite_not_met")
        self.assertEqual(response.json()["action"], "pilot")
        self.assertEqual(response.json()["required_state"], "measured-preflight-pass")
        self.assertEqual(response.json()["current_state"], "model-data-pass")
        self.assertEqual(response.json()["reason"], "insufficient_state")

    def test_validate_response_defers_deep_pilot_authorization_to_submit(self) -> None:
        report = ValidationReport(
            state=ValidationState.PILOT_PASS,
            findings=(),
            checked_files=(),
            artifact_fingerprint="a" * 64,
            validation_level="pilot",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            static = root / "web"
            static.mkdir()
            (static / "index.html").write_text("ok", encoding="utf-8")
            with patch("aptus.validation.validate_bundle", return_value=report):
                app = create_app(state_dir=root / "state", static_dir=static)
            with patch.object(
                app.state.aptus.jobs,
                "pilot_authorization",
            ) as authorization:
                with TestClient(app) as client:
                    response = client.post(
                        "/api/v1/validate",
                        json={
                            "bundle_dir": str(root / "bundle"),
                            "level": "static",
                            "run": False,
                        },
                    )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["authorization_current"])
        self.assertIsNone(response.json()["prelaunch_capacity_check"])
        self.assertIn(
            "performed atomically when full training is submitted",
            response.json()["authorization_error"],
        )
        authorization.assert_not_called()


@unittest.skipIf(
    TestClient is None,
    "Install the server and test extras for endpoint integration tests.",
)
class AppleRuntimeApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.client = TestClient(
            create_app(state_dir=Path(self.temporary.name) / "state")
        )

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def test_platform_and_runtime_inventory_are_explicit_endpoints(self) -> None:
        platform_profile = types.SimpleNamespace(
            to_dict=lambda: {
                "chip_name": "Apple M5 Pro",
                "metal_gpu_core_count": 20,
                "unified_memory_bytes": 64,
            }
        )
        with (
            patch("aptus.api.probe_apple_platform", return_value=platform_profile),
            patch(
                "aptus.api.runtime_inventory",
                return_value={
                    "schema_version": "aptus.runtime-inventory.v1",
                    "available": {"mlx-lm": ["/managed/python"]},
                },
            ),
        ):
            self.client.app.state.aptus.runtime_paths = {"mlx-lm": "/managed/python"}
            platform_response = self.client.get("/api/v1/platform")
            runtime_response = self.client.get("/api/v1/runtimes")

        self.assertEqual(platform_response.status_code, 200)
        self.assertEqual(
            platform_response.json()["platform"]["chip_name"], "Apple M5 Pro"
        )
        self.assertEqual(
            platform_response.json()["platform"]["metal_gpu_core_count"], 20
        )
        self.assertEqual(
            runtime_response.json()["available"]["mlx-lm"], ["/managed/python"]
        )
        self.assertEqual(
            runtime_response.json()["selected"],
            {"mlx-lm": "/managed/python"},
        )

    def test_inference_models_reject_non_loopback_endpoints(self) -> None:
        response = self.client.post(
            "/api/v1/inference/models",
            json={
                "service": "lm-studio",
                "endpoint": "http://example.com:1234",
            },
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"]["code"], "non_loopback_endpoint")

    def test_runtime_configuration_returns_the_canonical_interpreter(self) -> None:
        configured = {
            "status": "ok",
            "runtime_id": "mlx-lm",
            "interpreter_path": "/managed/python",
            "interpreter": {"path": "/managed/python"},
            "persisted": True,
        }
        with patch.object(
            self.client.app.state.aptus,
            "configure_runtime",
            return_value=configured,
        ) as configure:
            response = self.client.post(
                "/api/v1/runtimes/configure",
                json={
                    "runtime_id": "mlx-lm",
                    "interpreter_path": "/selected/python",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["interpreter_path"], "/managed/python")
        configure.assert_called_once_with("mlx-lm", Path("/selected/python"))

    def test_inference_generation_uses_the_selected_local_adapter(self) -> None:
        client = types.SimpleNamespace(
            generate=lambda **values: {
                "status": "ok",
                "service": "omlx",
                "content": values["messages"][0]["content"],
            }
        )
        with patch("aptus.api.OMLXClient", return_value=client) as constructor:
            response = self.client.post(
                "/api/v1/inference/generate",
                json={
                    "service": "omlx",
                    "model": "local/model",
                    "messages": [{"role": "user", "content": "test prompt"}],
                    "max_tokens": 32,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["content"], "test prompt")
        constructor.assert_called_once_with(endpoint=None, timeout=5.0)


if __name__ == "__main__":
    unittest.main()
