import tempfile
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
from aptus.domain import ValidationReport, ValidationState
from aptus.execution import ActiveJobError, JobPrerequisiteError

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
        self.assertEqual(capabilities["backends"], ["cuda"])
        self.assertEqual(capabilities["supported_execution_backends"], ["cuda"])
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


if __name__ == "__main__":
    unittest.main()
