import shutil
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
from aptus.api_contracts import ModelCompatibilityResponse, ModelInspectionResponse
from aptus.catalog import reviewed_qwen3_moe_quantization_layout
from aptus.domain import Backend, ValidationReport, ValidationState, to_primitive
from aptus.execution import ActiveJobError, JobPrerequisiteError
from aptus.local_store import atomic_write_json
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
                project_id="project_" + "a" * 32,
                expected_project_revision_id="revision_" + "b" * 32,
                action="train",
                confirm_full_train=True,
                resume_from="checkpoint-1",
            )

    def test_inspection_response_allows_incomplete_moe_evidence(self) -> None:
        response = ModelInspectionResponse.model_validate(
            {
                "status": "ok",
                "model_id": "provider/incomplete-moe",
                "requested_revision": "main",
                "resolved_revision": "a" * 40,
                "facts": {
                    "model_type": "unknown_moe",
                    "architecture": "UnknownMoeForCausalLM",
                    "moe": {
                        "expert_count": 64,
                        "experts_per_token": None,
                        "expert_intermediate_size": 512,
                        "decoder_sparse_step": 1,
                        "mlp_only_layers": None,
                        "shared_expert_intermediate_size": None,
                    },
                },
                "compatibility": {
                    "status": "unsupported",
                    "family": "unknown_moe",
                    "supported_runtime": None,
                    "supported_methods": [],
                    "distribution": None,
                    "evidence_requirement": "implementation-required",
                    "adapter_scope": None,
                    "reason": "The provider topology is incomplete.",
                },
            }
        )

        self.assertIsNone(response.facts.moe.experts_per_token)

    def test_model_compatibility_contract_accepts_only_coherent_variants(self) -> None:
        variants = (
            {
                "status": "conditional",
                "family": "qwen3_moe",
                "supported_runtime": "mlx-lm",
                "supported_methods": ["qlora"],
                "distribution": "single",
                "evidence_requirement": "pilot-required",
                "adapter_scope": "attention-only",
                "reason": "A measured pilot remains required.",
            },
            {
                "status": "recognized",
                "family": "llama",
                "supported_runtime": None,
                "supported_methods": [],
                "distribution": None,
                "evidence_requirement": "pilot-required",
                "adapter_scope": None,
                "reason": "The planner decides the executable path.",
            },
            {
                "status": "unsupported",
                "family": None,
                "supported_runtime": None,
                "supported_methods": [],
                "distribution": None,
                "evidence_requirement": "implementation-required",
                "adapter_scope": None,
                "reason": "No reviewed policy matches this model.",
            },
        )

        for payload in variants:
            with self.subTest(status=payload["status"]):
                validated = ModelCompatibilityResponse.model_validate(payload)
                self.assertEqual(validated.model_dump(), payload)

    def test_model_compatibility_contract_rejects_contradictory_evidence(
        self,
    ) -> None:
        conditional = {
            "status": "conditional",
            "family": "qwen3_moe",
            "supported_runtime": "mlx-lm",
            "supported_methods": ["qlora"],
            "distribution": "single",
            "evidence_requirement": "pilot-required",
            "adapter_scope": "attention-only",
            "reason": "A measured pilot remains required.",
        }
        recognized = {
            "status": "recognized",
            "family": "llama",
            "supported_runtime": None,
            "supported_methods": [],
            "distribution": None,
            "evidence_requirement": "pilot-required",
            "adapter_scope": None,
            "reason": "The planner decides the executable path.",
        }
        unsupported = {
            "status": "unsupported",
            "family": None,
            "supported_runtime": None,
            "supported_methods": [],
            "distribution": None,
            "evidence_requirement": "implementation-required",
            "adapter_scope": None,
            "reason": "No reviewed policy matches this model.",
        }
        invalid_variants = {
            "null runtime": {**conditional, "supported_runtime": None},
            "empty runtime": {**conditional, "supported_runtime": ""},
            "empty methods": {**conditional, "supported_methods": []},
            "empty method name": {**conditional, "supported_methods": [""]},
            "null distribution": {**conditional, "distribution": None},
            "empty distribution": {**conditional, "distribution": ""},
            "implementation evidence": {
                **conditional,
                "evidence_requirement": "implementation-required",
            },
            "missing adapter scope": {
                key: value
                for key, value in conditional.items()
                if key != "adapter_scope"
            },
            "empty adapter scope": {**conditional, "adapter_scope": ""},
            "empty family": {**conditional, "family": ""},
            "empty reason": {**conditional, "reason": ""},
            "unknown field": {**conditional, "unreviewed": True},
            "recognized runtime claim": {
                **recognized,
                "supported_runtime": "mlx-lm",
            },
            "recognized method claim": {
                **recognized,
                "supported_methods": ["lora"],
            },
            "recognized distribution claim": {
                **recognized,
                "distribution": "single",
            },
            "recognized adapter claim": {
                **recognized,
                "adapter_scope": "attention-only",
            },
            "recognized implementation evidence": {
                **recognized,
                "evidence_requirement": "implementation-required",
            },
            "unsupported runtime claim": {
                **unsupported,
                "supported_runtime": "mlx-lm",
            },
            "unsupported method claim": {
                **unsupported,
                "supported_methods": ["qlora"],
            },
            "unsupported distribution claim": {
                **unsupported,
                "distribution": "single",
            },
            "unsupported adapter claim": {
                **unsupported,
                "adapter_scope": "attention-only",
            },
            "unsupported pilot evidence": {
                **unsupported,
                "evidence_requirement": "pilot-required",
            },
        }

        for case, payload in invalid_variants.items():
            with self.subTest(case=case):
                with self.assertRaises(ValidationError):
                    ModelCompatibilityResponse.model_validate(payload)

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
                    json={
                        "bundle_dir": "/tmp/untrusted-bundle",
                        "project_id": "project_" + "a" * 32,
                        "expected_project_revision_id": "revision_" + "b" * 32,
                        "action": "train",
                    },
                )
                validation_response = client.post(
                    "/api/v1/validate",
                    json={
                        "bundle_dir": "/tmp/untrusted-bundle",
                        "project_id": "project_" + "a" * 32,
                        "expected_project_revision_id": "revision_" + "b" * 32,
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

    def owned_bundle_request(self, bundle: Path) -> dict[str, str]:
        planned = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(planned.status_code, 200, planned.text)
        plan = planned.json()
        compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": plan["plan_id"],
                "output_dir": str(bundle),
                "project_id": plan["project_id"],
                "expected_project_revision_id": plan["project_revision_id"],
            },
        )
        self.assertEqual(compiled.status_code, 200, compiled.text)
        return {
            "project_id": plan["project_id"],
            "expected_project_revision_id": compiled.json()["project_revision_id"],
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
        planned_payload = planned.json()
        plan_id = planned_payload["plan_id"]
        project_id = planned_payload["project_id"]
        project_revision_id = planned_payload["project_revision_id"]

        bundle = self.root / "bundle"
        compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": plan_id,
                "output_dir": str(bundle),
                "project_id": project_id,
                "expected_project_revision_id": project_revision_id,
            },
        )
        self.assertEqual(compiled.status_code, 200, compiled.text)
        self.assertTrue(compiled.json()["archive_path"].endswith("bundle.zip"))
        compiled_revision_id = compiled.json()["project_revision_id"]

        validated = self.client.post(
            "/api/v1/validate",
            json={
                "bundle_dir": str(bundle),
                "project_id": project_id,
                "expected_project_revision_id": compiled_revision_id,
                "level": "static",
                "run": False,
            },
        )
        self.assertEqual(validated.status_code, 200, validated.text)
        self.assertEqual(validated.json()["state"], "static-pass")
        validated_revision_id = validated.json()["project_revision_id"]

        restored = self.client.get("/api/v1/bootstrap")
        self.assertEqual(restored.status_code, 200, restored.text)
        self.assertEqual(restored.json()["plan"]["plan_id"], plan_id)
        self.assertEqual(restored.json()["bundle"]["bundle_dir"], str(bundle.resolve()))
        self.assertEqual(restored.json()["bundle"]["report"]["state"], "static-pass")

        runtime = self.client.post(
            "/api/v1/validate",
            json={
                "bundle_dir": str(bundle),
                "project_id": project_id,
                "expected_project_revision_id": validated_revision_id,
                "level": "pilot",
                "run": True,
            },
        )
        self.assertEqual(runtime.status_code, 409)
        self.assertEqual(runtime.json()["suggested_action"], "pilot")

        conflict = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": plan_id,
                "output_dir": str(bundle),
                "project_id": project_id,
                "expected_project_revision_id": validated_revision_id,
            },
        )
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.json()["error"], "path_conflict")

    def test_exact_qwen3_moe_plan_preserves_topology_and_derived_facts(self) -> None:
        payload = self.plan_payload()
        payload["model"] = {
            **payload["model"],
            "model_id": "Qwen/Qwen3-30B-A3B-MLX-4bit",
            "family": "qwen3_moe",
            "parameters_b": 30.5,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "layers": 48,
            "context_length": 40960,
            "model_type": "qwen3_moe",
            "architecture": "Qwen3MoeForCausalLM",
            "quantization_bits": 4,
            "quantization_layout": to_primitive(
                reviewed_qwen3_moe_quantization_layout(48)
            ),
            "moe": {
                "expert_count": 128,
                "experts_per_token": 8,
                "expert_intermediate_size": 768,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
            },
        }
        payload["hardware"] = {
            **payload["hardware"],
            "backend": "mps",
            "gpu_count": 1,
            "vram_gib": 64,
            "free_vram_gib": 48,
            "supports_bf16": False,
            "supports_8bit": False,
            "supports_4bit": False,
            "host_ram_gib": 64,
            "host_ram_free_gib": 48,
            "reserve_gib": 8,
        }
        payload["target"] = {
            **payload["target"],
            "method_preference": "qlora",
            "training_runtime": "mlx-lm",
        }

        response = self.client.post("/api/v1/plan", json=payload)

        self.assertEqual(response.status_code, 200, response.text)
        plan = response.json()
        self.assertEqual(plan["schema_version"], "aptus.training-plan.v3")
        self.assertEqual(plan["model"]["model_type"], "qwen3_moe")
        self.assertEqual(
            len(plan["model"]["quantization_layout"]["module_overrides"]), 48
        )
        self.assertEqual(plan["model"]["moe"]["expert_count"], 128)
        self.assertEqual(plan["model"]["sparse_layer_count"], 48)
        self.assertLess(plan["model"]["active_parameters"], plan["model"]["parameters"])
        self.assertEqual(plan["recommended"]["method"], "qlora")
        self.assertEqual(
            plan["recommended"]["runtime_contract"]["training_runtime"],
            "mlx-lm",
        )

    def test_moe_integer_facts_reject_boolean_coercion(self) -> None:
        payload = self.plan_payload()
        payload["model"] = {
            **payload["model"],
            "quantization_bits": True,
            "quantization_layout": {
                "default_bits": True,
                "default_group_size": 64,
                "module_overrides": [],
            },
            "moe": {
                "expert_count": True,
                "experts_per_token": 1,
                "expert_intermediate_size": 768,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
            },
        }

        response = self.client.post("/api/v1/plan", json=payload)

        self.assertEqual(response.status_code, 422, response.text)
        locations = {tuple(item["loc"]) for item in response.json()["details"]}
        self.assertIn(("body", "model", "quantization_bits"), locations)
        self.assertIn(
            ("body", "model", "quantization_layout", "default_bits"), locations
        )
        self.assertIn(("body", "model", "moe", "expert_count"), locations)

    def test_model_inspection_contract_exposes_exact_moe_compatibility(self) -> None:
        inspection = {
            "status": "ok",
            "model_id": "Qwen/Qwen3-30B-A3B-MLX-4bit",
            "requested_revision": "main",
            "resolved_revision": "d" * 40,
            "facts": {
                "architecture": "Qwen3MoeForCausalLM",
                "architectures": ["Qwen3MoeForCausalLM"],
                "model_type": "qwen3_moe",
                "family": "qwen3_moe",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "layers": 48,
                "context_length": 40960,
                "attention_heads": 32,
                "key_value_heads": 4,
                "vocab_size": 151936,
                "quantization_bits": 4,
                "quantization_layout": to_primitive(
                    reviewed_qwen3_moe_quantization_layout(48)
                ),
                "moe": {
                    "expert_count": 128,
                    "experts_per_token": 8,
                    "expert_intermediate_size": 768,
                    "decoder_sparse_step": 1,
                    "mlp_only_layers": [],
                    "shared_expert_intermediate_size": None,
                },
                "license_name": "apache-2.0",
                "parameters": None,
                "training_allowed": None,
            },
            "provenance": {},
            "warnings": [],
            "compatibility": {
                "status": "conditional",
                "family": "qwen3_moe",
                "supported_runtime": "mlx-lm",
                "supported_methods": ["qlora"],
                "distribution": "single",
                "evidence_requirement": "pilot-required",
                "adapter_scope": "attention-only",
                "reason": "Exact model-data and pilot evidence are required.",
            },
            "explicit_user_facts_required": ["parameters", "training_allowed"],
        }
        static = self.root / "inspection-web"
        static.mkdir()
        (static / "index.html").write_text("Aptus", encoding="utf-8")
        with patch(
            "aptus.inspection.inspect_huggingface_model",
            return_value=inspection,
        ):
            client = TestClient(
                create_app(state_dir=self.root / "inspection-state", static_dir=static)
            )
        try:
            response = client.post(
                "/api/v1/models/inspect",
                json={
                    "model_id": "Qwen/Qwen3-30B-A3B-MLX-4bit",
                    "revision": "main",
                },
            )
        finally:
            client.close()

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["facts"]["moe"]["experts_per_token"], 8)
        self.assertEqual(
            len(result["facts"]["quantization_layout"]["module_overrides"]), 48
        )
        self.assertEqual(result["compatibility"]["status"], "conditional")
        self.assertEqual(result["compatibility"]["adapter_scope"], "attention-only")

    def test_model_inspection_response_rejects_malformed_compatibility(self) -> None:
        inspection = {
            "status": "ok",
            "model_id": "provider/malformed",
            "requested_revision": "main",
            "compatibility": {
                "status": "conditional",
                "family": "qwen3_moe",
                "supported_runtime": None,
                "supported_methods": ["qlora"],
                "distribution": "single",
                "evidence_requirement": "pilot-required",
                "adapter_scope": "attention-only",
                "reason": "The producer omitted the executable runtime.",
            },
        }
        with patch(
            "aptus.inspection.inspect_huggingface_model",
            return_value=inspection,
        ):
            client = TestClient(
                create_app(
                    state_dir=self.root / "malformed-inspection-state",
                    static_dir=self.root / "web",
                ),
                raise_server_exceptions=False,
            )
        try:
            response = client.post(
                "/api/v1/models/inspect",
                json={"model_id": "provider/malformed", "revision": "main"},
            )
        finally:
            client.close()

        self.assertEqual(response.status_code, 500, response.text)

    def test_mutating_workflow_requests_require_exact_project_identity(self) -> None:
        for path, payload in (
            ("/api/v1/compile", {"plan_id": "plan_test", "output_dir": "bundle"}),
            (
                "/api/v1/validate",
                {"bundle_dir": "bundle", "level": "static", "run": False},
            ),
            ("/api/v1/jobs", {"bundle_dir": "bundle", "action": "preflight"}),
        ):
            with self.subTest(path=path):
                response = self.client.post(path, json=payload)
                self.assertEqual(response.status_code, 422, response.text)

    def test_compile_rejects_stale_or_cross_project_plan_identity(self) -> None:
        first = self.client.post("/api/v1/plan", json=self.plan_payload()).json()
        second_payload = self.plan_payload()
        second_payload["model"] = {
            **second_payload["model"],
            "revision": "b" * 40,
        }
        second = self.client.post("/api/v1/plan", json=second_payload).json()
        cross_project = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": first["plan_id"],
                "output_dir": str(self.root / "cross-project"),
                "project_id": second["project_id"],
                "expected_project_revision_id": second["project_revision_id"],
            },
        )
        self.assertEqual(cross_project.status_code, 409, cross_project.text)
        self.assertEqual(cross_project.json()["error"], "project_plan_mismatch")

        payload = self.plan_payload()
        payload["project_id"] = first["project_id"]
        latest = self.client.post("/api/v1/plan", json=payload)
        self.assertEqual(latest.status_code, 200, latest.text)
        stale = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": first["plan_id"],
                "output_dir": str(self.root / "stale"),
                "project_id": first["project_id"],
                "expected_project_revision_id": first["project_revision_id"],
            },
        )
        self.assertEqual(stale.status_code, 409, stale.text)
        self.assertEqual(stale.json()["error"], "project_revision_conflict")

    def test_compile_race_never_publishes_uncommitted_legacy_bundle_pointer(
        self,
    ) -> None:
        planned = self.client.post("/api/v1/plan", json=self.plan_payload()).json()
        projects = self.client.app.state.aptus.projects
        create_revision = projects.create_revision

        def race_project_revision(
            project_id: str, **changes: object
        ) -> dict[str, object]:
            create_revision(
                project_id,
                reason="concurrent-update",
                base_revision_id=planned["project_revision_id"],
                expected_latest_revision_id=planned["project_revision_id"],
            )
            return create_revision(project_id, **changes)

        with patch.object(
            projects,
            "create_revision",
            side_effect=race_project_revision,
        ):
            response = self.client.post(
                "/api/v1/compile",
                json={
                    "plan_id": planned["plan_id"],
                    "output_dir": str(self.root / "uncommitted-bundle"),
                    "project_id": planned["project_id"],
                    "expected_project_revision_id": planned["project_revision_id"],
                },
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"], "project_revision_conflict")
        self.assertFalse(self.client.app.state.aptus.current_bundle_path.exists())

    def test_named_project_history_and_recovery_are_explicit_contracts(self) -> None:
        created = self.client.post(
            "/api/v1/projects", json={"name": "Parish corpus adapter"}
        )
        self.assertEqual(created.status_code, 201, created.text)
        project_id = created.json()["project_id"]

        payload = self.plan_payload()
        payload["project_id"] = project_id
        payload["project_name"] = "Parish corpus adapter"
        planned = self.client.post("/api/v1/plan", json=payload)
        self.assertEqual(planned.status_code, 200, planned.text)
        self.assertEqual(planned.json()["project_id"], project_id)
        revision_id = planned.json()["project_revision_id"]

        history = self.client.get(f"/api/v1/projects/{project_id}/revisions")
        self.assertEqual(history.status_code, 200, history.text)
        self.assertEqual(history.json()[0]["reason"], "plan-created")
        revision = self.client.get(
            f"/api/v1/projects/{project_id}/revisions/{revision_id}"
        )
        self.assertEqual(revision.status_code, 200, revision.text)
        self.assertFalse(revision.json()["training_authorization"]["current"])

        recovered = self.client.post(
            f"/api/v1/projects/{project_id}/recover",
            json={"revision_id": revision_id},
        )
        self.assertEqual(recovered.status_code, 200, recovered.text)
        self.assertFalse(recovered.json()["training_authorization_current"])
        self.assertNotEqual(recovered.json()["revision"]["revision_id"], revision_id)

        bootstrap = self.client.get("/api/v1/bootstrap")
        self.assertEqual(bootstrap.status_code, 200, bootstrap.text)
        self.assertEqual(bootstrap.json()["project"]["project_id"], project_id)
        self.assertEqual(bootstrap.json()["project_history"][0]["ordinal"], 2)

    def test_legacy_plan_is_preserved_and_requires_explicit_replan(self) -> None:
        context = self.client.app.state.aptus
        plan_id = "plan_" + "c" * 20
        legacy_plan = {
            "schema_version": "aptus.training-plan.v2",
            "plan_id": plan_id,
            "recommended": {"candidate_id": "candidate_legacy"},
        }
        atomic_write_json(
            context.plans_dir / f"{plan_id}.json", legacy_plan, mode=0o600
        )
        project = context.projects.create("Legacy saved plan")
        revision = context.projects.create_revision(
            project["project_id"],
            reason="legacy-plan-imported",
            plan_id=plan_id,
            plan_snapshot=legacy_plan,
            selected_candidate_id="candidate_legacy",
        )
        saved_plan_path = context.plans_dir / f"{plan_id}.json"
        before = saved_plan_path.read_bytes()

        bootstrap = self.client.get("/api/v1/bootstrap")
        loaded = self.client.get(f"/api/v1/plans/{plan_id}")
        compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": plan_id,
                "output_dir": str(self.root / "legacy-output"),
                "project_id": project["project_id"],
                "expected_project_revision_id": revision["revision_id"],
            },
        )
        recovered = self.client.post(
            f"/api/v1/projects/{project['project_id']}/recover",
            json={"revision_id": revision["revision_id"]},
        )

        self.assertEqual(bootstrap.status_code, 200, bootstrap.text)
        self.assertIsNone(bootstrap.json().get("plan"))
        self.assertEqual(
            bootstrap.json()["replan_required"],
            {
                "status": "replan_required",
                "plan_id": plan_id,
                "found_schema": "aptus.training-plan.v2",
                "required_schema": "aptus.training-plan.v3",
                "source": "project-revision",
                "project_id": project["project_id"],
                "project_revision_id": revision["revision_id"],
                "message": (
                    "This saved plan predates the current executable contract. "
                    "Create a new plan from its preserved facts before compiling "
                    "or recovering it."
                ),
            },
        )
        for response in (loaded, compiled, recovered):
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(response.json()["error"], "replan_required")
            self.assertEqual(
                response.json()["required_schema"], "aptus.training-plan.v3"
            )
        self.assertEqual(saved_plan_path.read_bytes(), before)
        self.assertFalse((self.root / "legacy-output").exists())
        self.assertEqual(
            context.projects.get(project["project_id"])["revision_count"], 1
        )

    def test_plan_only_recovery_ignores_newer_legacy_bundle_pointer(self) -> None:
        planned = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(planned.status_code, 200, planned.text)
        plan = planned.json()
        bundle = self.root / "newer-global-bundle"
        compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": plan["plan_id"],
                "output_dir": str(bundle),
                "project_id": plan["project_id"],
                "expected_project_revision_id": plan["project_revision_id"],
            },
        )
        self.assertEqual(compiled.status_code, 200, compiled.text)
        recovered = self.client.post(
            f"/api/v1/projects/{plan['project_id']}/recover",
            json={"revision_id": plan["project_revision_id"]},
        )
        self.assertEqual(recovered.status_code, 200, recovered.text)

        bootstrap = self.client.get("/api/v1/bootstrap")

        self.assertEqual(bootstrap.status_code, 200, bootstrap.text)
        self.assertEqual(bootstrap.json()["plan"]["plan_id"], plan["plan_id"])
        self.assertEqual(bootstrap.json()["plan"]["project_id"], plan["project_id"])
        self.assertEqual(
            bootstrap.json()["plan"]["project_revision_id"],
            recovered.json()["revision"]["revision_id"],
        )
        self.assertIsNone(bootstrap.json().get("bundle"))

    def test_bootstrap_rejects_valid_bundle_replaced_by_another_identity(self) -> None:
        first = self.client.post("/api/v1/plan", json=self.plan_payload()).json()
        first_bundle = self.root / "first-bundle"
        first_compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": first["plan_id"],
                "output_dir": str(first_bundle),
                "project_id": first["project_id"],
                "expected_project_revision_id": first["project_revision_id"],
            },
        ).json()

        second_payload = self.plan_payload()
        second_payload["model"] = {
            **second_payload["model"],
            "revision": "b" * 40,
        }
        second = self.client.post("/api/v1/plan", json=second_payload).json()
        second_bundle = self.root / "second-bundle"
        second_compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": second["plan_id"],
                "output_dir": str(second_bundle),
                "project_id": second["project_id"],
                "expected_project_revision_id": second["project_revision_id"],
            },
        )
        self.assertEqual(second_compiled.status_code, 200, second_compiled.text)
        recovered = self.client.post(
            f"/api/v1/projects/{first['project_id']}/recover",
            json={"revision_id": first_compiled["project_revision_id"]},
        )
        self.assertEqual(recovered.status_code, 200, recovered.text)

        shutil.rmtree(first_bundle)
        shutil.copytree(second_bundle, first_bundle)
        bootstrap = self.client.get("/api/v1/bootstrap")

        self.assertEqual(bootstrap.status_code, 200, bootstrap.text)
        self.assertEqual(bootstrap.json()["plan"]["plan_id"], first["plan_id"])
        self.assertIsNone(bootstrap.json().get("bundle"))

    def test_restored_project_identity_supports_the_next_validate_and_compile(
        self,
    ) -> None:
        planned = self.client.post("/api/v1/plan", json=self.plan_payload()).json()
        first_bundle = self.root / "restored-first"
        compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": planned["plan_id"],
                "output_dir": str(first_bundle),
                "project_id": planned["project_id"],
                "expected_project_revision_id": planned["project_revision_id"],
            },
        )
        self.assertEqual(compiled.status_code, 200, compiled.text)

        restored = self.client.get("/api/v1/bootstrap").json()
        self.assertEqual(restored["plan"]["project_id"], planned["project_id"])
        self.assertEqual(
            restored["plan"]["project_revision_id"],
            restored["bundle"]["project_revision_id"],
        )
        validated = self.client.post(
            "/api/v1/validate",
            json={
                "bundle_dir": restored["bundle"]["bundle_dir"],
                "project_id": restored["bundle"]["project_id"],
                "expected_project_revision_id": restored["bundle"][
                    "project_revision_id"
                ],
                "level": "static",
                "run": False,
            },
        )
        self.assertEqual(validated.status_code, 200, validated.text)

        latest = self.client.get("/api/v1/bootstrap").json()["plan"]
        second_compile = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": latest["plan_id"],
                "output_dir": str(self.root / "restored-second"),
                "project_id": latest["project_id"],
                "expected_project_revision_id": latest["project_revision_id"],
            },
        )
        self.assertEqual(second_compile.status_code, 200, second_compile.text)

    def test_identical_plan_ids_remain_bound_to_distinct_projects(self) -> None:
        first_project = self.client.post(
            "/api/v1/projects", json={"name": "First named project"}
        ).json()
        second_project = self.client.post(
            "/api/v1/projects", json={"name": "Second named project"}
        ).json()
        first_payload = {
            **self.plan_payload(),
            "project_id": first_project["project_id"],
            "project_name": "First named project",
        }
        second_payload = {
            **self.plan_payload(),
            "project_id": second_project["project_id"],
            "project_name": "Second named project",
        }
        first_plan = self.client.post("/api/v1/plan", json=first_payload).json()
        second_plan = self.client.post("/api/v1/plan", json=second_payload).json()
        self.assertEqual(first_plan["plan_id"], second_plan["plan_id"])

        compiled: list[dict[str, object]] = []
        for label, plan in (("first", first_plan), ("second", second_plan)):
            bundle = self.root / f"{label}-owned-bundle"
            response = self.client.post(
                "/api/v1/compile",
                json={
                    "plan_id": plan["plan_id"],
                    "output_dir": str(bundle),
                    "project_id": plan["project_id"],
                    "expected_project_revision_id": plan["project_revision_id"],
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            compiled.append(response.json())

        for label, plan, bundle in (
            ("first", first_plan, compiled[0]),
            ("second", second_plan, compiled[1]),
        ):
            revision = self.client.get(
                f"/api/v1/projects/{plan['project_id']}/revisions/"
                f"{bundle['project_revision_id']}"
            ).json()
            self.assertEqual(
                revision["bundle"]["bundle_dir"],
                str((self.root / f"{label}-owned-bundle").resolve()),
            )
            validated = self.client.post(
                "/api/v1/validate",
                json={
                    "bundle_dir": bundle["bundle_dir"],
                    "project_id": plan["project_id"],
                    "expected_project_revision_id": bundle["project_revision_id"],
                    "level": "static",
                    "run": False,
                },
            )
            self.assertEqual(validated.status_code, 200, validated.text)
            self.assertEqual(validated.json()["project_id"], plan["project_id"])

    def test_replaced_bundle_at_same_path_is_rejected_before_validation_or_job(
        self,
    ) -> None:
        first_plan_response = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(first_plan_response.status_code, 200, first_plan_response.text)
        first_plan = first_plan_response.json()
        first_bundle = self.root / "first-bundle"
        first_compile_response = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": first_plan["plan_id"],
                "output_dir": str(first_bundle),
                "project_id": first_plan["project_id"],
                "expected_project_revision_id": first_plan["project_revision_id"],
            },
        )
        self.assertEqual(
            first_compile_response.status_code, 200, first_compile_response.text
        )
        first_compile = first_compile_response.json()

        second_payload = self.plan_payload()
        target = dict(second_payload["target"])  # type: ignore[arg-type]
        target["sequence_length"] = 96
        second_payload["target"] = target
        second_plan_response = self.client.post("/api/v1/plan", json=second_payload)
        self.assertEqual(
            second_plan_response.status_code, 200, second_plan_response.text
        )
        second_plan = second_plan_response.json()
        self.assertNotEqual(first_plan["plan_id"], second_plan["plan_id"])
        second_bundle = self.root / "second-bundle"
        second_compile_response = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": second_plan["plan_id"],
                "output_dir": str(second_bundle),
                "project_id": second_plan["project_id"],
                "expected_project_revision_id": second_plan["project_revision_id"],
            },
        )
        self.assertEqual(
            second_compile_response.status_code, 200, second_compile_response.text
        )

        shutil.rmtree(first_bundle)
        shutil.copytree(second_bundle, first_bundle)
        report_path = first_bundle / "validation-report.json"
        report_before = report_path.read_bytes()
        jobs_before = list(self.client.app.state.aptus.jobs.root.glob("job_*.json"))
        project_before = self.client.app.state.aptus.projects.get(
            first_plan["project_id"]
        )

        validation = self.client.post(
            "/api/v1/validate",
            json={
                "bundle_dir": str(first_bundle),
                "project_id": first_plan["project_id"],
                "expected_project_revision_id": first_compile["project_revision_id"],
                "level": "static",
                "run": False,
            },
        )
        job = self.client.post(
            "/api/v1/jobs",
            json={
                "bundle_dir": str(first_bundle),
                "project_id": first_plan["project_id"],
                "expected_project_revision_id": first_compile["project_revision_id"],
                "action": "preflight",
            },
        )
        project_after = self.client.app.state.aptus.projects.get(
            first_plan["project_id"]
        )
        jobs_after = list(self.client.app.state.aptus.jobs.root.glob("job_*.json"))

        self.assertEqual(validation.status_code, 409, validation.text)
        self.assertEqual(validation.json()["error"], "project_bundle_binding_mismatch")
        self.assertEqual(job.status_code, 409, job.text)
        self.assertEqual(job.json()["error"], "project_bundle_binding_mismatch")
        self.assertEqual(report_path.read_bytes(), report_before)
        self.assertEqual(jobs_after, jobs_before)
        self.assertEqual(
            project_after["latest_revision_id"],
            project_before["latest_revision_id"],
        )

    def test_compile_conflict_removes_uncommitted_bundle_and_archive(self) -> None:
        plan_response = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(plan_response.status_code, 200, plan_response.text)
        plan = plan_response.json()
        output = self.root / "raced-bundle"
        projects = self.client.app.state.aptus.projects
        original_create_revision = projects.create_revision
        raced = False

        def create_with_competing_revision(
            project_id: str, **changes: object
        ) -> dict[str, object]:
            nonlocal raced
            if changes.get("reason") == "bundle-compiled" and not raced:
                raced = True
                original_create_revision(project_id, reason="competing-update")
            return original_create_revision(project_id, **changes)

        with patch.object(
            projects, "create_revision", side_effect=create_with_competing_revision
        ):
            response = self.client.post(
                "/api/v1/compile",
                json={
                    "plan_id": plan["plan_id"],
                    "output_dir": str(output),
                    "project_id": plan["project_id"],
                    "expected_project_revision_id": plan["project_revision_id"],
                },
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"], "project_revision_conflict")
        self.assertTrue(raced)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".zip").exists())
        self.assertIsNone(self.client.app.state.aptus.load_bundle_reference())

    def test_compile_conflict_never_deletes_same_path_replacements(self) -> None:
        plan_response = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(plan_response.status_code, 200, plan_response.text)
        plan = plan_response.json()
        output = self.root / "swapped-bundle"
        archive = output.with_suffix(".zip")
        moved_output = self.root / "generated-bundle-moved"
        moved_archive = self.root / "generated-archive-moved.zip"
        projects = self.client.app.state.aptus.projects
        original_create_revision = projects.create_revision
        swapped = False

        def create_after_path_swap(
            project_id: str, **changes: object
        ) -> dict[str, object]:
            nonlocal swapped
            if changes.get("reason") == "bundle-compiled" and not swapped:
                swapped = True
                output.rename(moved_output)
                archive.rename(moved_archive)
                output.mkdir()
                (output / "unrelated.txt").write_text(
                    "keep directory\n", encoding="utf-8"
                )
                archive.write_text("keep archive\n", encoding="utf-8")
                original_create_revision(project_id, reason="competing-update")
            return original_create_revision(project_id, **changes)

        with patch.object(
            projects, "create_revision", side_effect=create_after_path_swap
        ):
            response = self.client.post(
                "/api/v1/compile",
                json={
                    "plan_id": plan["plan_id"],
                    "output_dir": str(output),
                    "project_id": plan["project_id"],
                    "expected_project_revision_id": plan["project_revision_id"],
                },
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertTrue(swapped)
        self.assertEqual(
            (output / "unrelated.txt").read_text(encoding="utf-8"),
            "keep directory\n",
        )
        self.assertEqual(archive.read_text(encoding="utf-8"), "keep archive\n")
        self.assertTrue((moved_output / "bundle-manifest.json").is_file())
        self.assertTrue(moved_archive.is_file())

    def test_bootstrap_omits_an_archive_that_changed_after_compile(self) -> None:
        plan_response = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(plan_response.status_code, 200, plan_response.text)
        plan = plan_response.json()
        bundle = self.root / "archive-bound-bundle"
        compile_response = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": plan["plan_id"],
                "output_dir": str(bundle),
                "project_id": plan["project_id"],
                "expected_project_revision_id": plan["project_revision_id"],
            },
        )
        self.assertEqual(compile_response.status_code, 200, compile_response.text)
        archive = Path(compile_response.json()["archive_path"])
        archive.write_bytes(b"substituted archive")

        bootstrap = self.client.get("/api/v1/bootstrap")

        self.assertEqual(bootstrap.status_code, 200, bootstrap.text)
        self.assertEqual(
            bootstrap.json()["bundle"]["bundle_dir"], str(bundle.resolve())
        )
        self.assertIsNone(bootstrap.json()["bundle"]["archive_path"])

    def test_bootstrap_restores_only_jobs_bound_to_the_exact_current_revision(
        self,
    ) -> None:
        planned = self.client.post("/api/v1/plan", json=self.plan_payload()).json()
        job_id = "job_" + "d" * 32
        active_job = {
            "schema_version": "aptus.job-record.v1",
            "id": job_id,
            "job_id": job_id,
            "state": "running",
            "action": "pilot",
            "bundle_dir": str(self.root / "unrelated-bundle"),
            "created_at": "2026-07-27T12:00:00Z",
        }
        jobs = self.client.app.state.aptus.jobs
        with patch.object(jobs, "list", return_value=[active_job]):
            unrelated = self.client.get("/api/v1/bootstrap")
        self.assertEqual(unrelated.status_code, 200, unrelated.text)
        self.assertIsNone(unrelated.json().get("job"))

        projects = self.client.app.state.aptus.projects
        plan_only = projects.create_revision(
            planned["project_id"],
            reason="job-submitted-fixture",
            base_revision_id=planned["project_revision_id"],
            expected_latest_revision_id=planned["project_revision_id"],
            job_ids=[job_id],
        )
        with patch.object(jobs, "list", return_value=[active_job]):
            plan_only_response = self.client.get("/api/v1/bootstrap")
        self.assertEqual(plan_only_response.status_code, 200, plan_only_response.text)
        self.assertIsNone(plan_only_response.json().get("job"))

        bundle_dir = self.root / "bound-job-bundle"
        compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": planned["plan_id"],
                "output_dir": str(bundle_dir),
                "project_id": planned["project_id"],
                "expected_project_revision_id": plan_only["revision_id"],
            },
        )
        self.assertEqual(compiled.status_code, 200, compiled.text)
        compiled_revision_id = compiled.json()["project_revision_id"]
        projects.create_revision(
            planned["project_id"],
            reason="bundle-and-job-fixture",
            base_revision_id=compiled_revision_id,
            expected_latest_revision_id=compiled_revision_id,
            job_ids=[job_id],
        )
        bound_active_job = {**active_job, "bundle_dir": str(bundle_dir.resolve())}
        with patch.object(jobs, "list", return_value=[bound_active_job]):
            bound = self.client.get("/api/v1/bootstrap")
        self.assertEqual(bound.status_code, 200, bound.text)
        self.assertEqual(bound.json()["job"]["id"], job_id)

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
        bundle = self.root / "bundle"
        identity = self.owned_bundle_request(bundle)
        with patch.object(
            self.client.app.state.aptus.jobs,
            "submit",
            side_effect=ActiveJobError("one local GPU job is already active"),
        ):
            response = self.client.post(
                "/api/v1/jobs",
                json={"bundle_dir": str(bundle), "action": "pilot", **identity},
            )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"], "active_job_conflict")

    def test_job_prerequisite_failure_is_typed_as_http_409(self) -> None:
        bundle = self.root / "bundle"
        identity = self.owned_bundle_request(bundle)
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
                json={"bundle_dir": str(bundle), "action": "pilot", **identity},
            )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"], "job_prerequisite_not_met")
        self.assertEqual(response.json()["action"], "pilot")
        self.assertEqual(response.json()["required_state"], "measured-preflight-pass")
        self.assertEqual(response.json()["current_state"], "model-data-pass")
        self.assertEqual(response.json()["reason"], "insufficient_state")

    def test_validate_response_defers_deep_pilot_authorization_to_submit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            static = root / "web"
            static.mkdir()
            (static / "index.html").write_text("ok", encoding="utf-8")
            with patch("aptus.validation.validate_bundle") as validate_mock:
                app = create_app(state_dir=root / "state", static_dir=static)
            with patch.object(
                app.state.aptus.jobs,
                "pilot_authorization",
            ) as authorization:
                with TestClient(app) as client:
                    plan_response = client.post(
                        "/api/v1/plan", json=self.plan_payload()
                    )
                    self.assertEqual(plan_response.status_code, 200, plan_response.text)
                    plan = plan_response.json()
                    compile_response = client.post(
                        "/api/v1/compile",
                        json={
                            "plan_id": plan["plan_id"],
                            "output_dir": str(root / "bundle"),
                            "project_id": plan["project_id"],
                            "expected_project_revision_id": plan["project_revision_id"],
                        },
                    )
                    self.assertEqual(
                        compile_response.status_code, 200, compile_response.text
                    )
                    compiled = compile_response.json()
                    validate_mock.return_value = ValidationReport(
                        state=ValidationState.PILOT_PASS,
                        findings=(),
                        checked_files=(),
                        artifact_fingerprint=compiled["report"]["artifact_fingerprint"],
                        validation_level="pilot",
                    )
                    response = client.post(
                        "/api/v1/validate",
                        json={
                            "bundle_dir": str(root / "bundle"),
                            "project_id": plan["project_id"],
                            "expected_project_revision_id": compiled[
                                "project_revision_id"
                            ],
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
                    "interpreters": [],
                    "available": {"mlx-lm": ["/managed/python"]},
                    "compatible": {"mlx-lm": ["/managed/python"]},
                    "configuration": {},
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
                "endpoint": "http://127.0.0.1:8080/v1",
                "model": "local/model",
                "content": values["messages"][0]["content"],
                "usage": None,
                "response_id": "response-test",
                "payload": {},
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
