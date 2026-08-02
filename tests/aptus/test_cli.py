import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aptus.cli import main
from aptus.domain import to_primitive
from aptus.execution import JobPrerequisiteError
from aptus.model_compatibility import (
    create_model_inspection_receipt,
    subject_from_model,
)
from aptus.profiling import build_model_spec


def fact_arguments(dataset: Path) -> list[str]:
    return [
        "--model-id",
        "example/model",
        "--revision",
        "a" * 40,
        "--family",
        "llama",
        "--parameters-b",
        "1",
        "--hidden-size",
        "2048",
        "--intermediate-size",
        "8192",
        "--layers",
        "24",
        "--context-length",
        "4096",
        "--license",
        "apache-2.0",
        "--confirm-training-allowed",
        "--dataset",
        str(dataset),
        "--gpu-count",
        "1",
        "--vram-gib",
        "24",
        "--bf16",
        "--four-bit",
        "--host-ram-gib",
        "64",
        "--disk-free-gib",
        "500",
        "--objective",
        "memory",
        "--sequence-length",
        "128",
        "--effective-batch-size",
        "8",
        "--epochs",
        "1",
        "--checkpoint-steps",
        "10",
    ]


def inspection_receipt_payload() -> dict[str, object]:
    model = build_model_spec(
        model_id="example/model",
        revision="a" * 40,
        family="llama",
        parameters_b=1,
        hidden_size=2048,
        intermediate_size=8192,
        layers=24,
        context_length=4096,
        license_name="apache-2.0",
        training_allowed=True,
    )
    observed_at = "2026-07-29T12:00:00+00:00"
    facts = {
        field: getattr(model, field)
        for field in (
            "architecture",
            "context_length",
            "family",
            "hidden_size",
            "intermediate_size",
            "layers",
            "license_name",
            "model_type",
            "moe",
            "quantization_bits",
            "quantization_layout",
        )
    }
    provenance = {
        field: {
            "kind": "inferred" if field == "family" else "provider-declared",
            "source": (
                "Aptus exact model-type compatibility mapping"
                if field == "family"
                else "https://huggingface.co/example/model/config.json"
            ),
            "observed_at": observed_at,
            "resolved_revision": model.revision,
        }
        for field, value in facts.items()
        if value is not None
    }
    return to_primitive(
        create_model_inspection_receipt(
            model_id=model.model_id,
            resolved_revision=model.revision,
            facts=facts,
            provenance=provenance,
            subject=subject_from_model(model),
            evaluated_at=observed_at,
        )
    )


class CliIntegrationTests(unittest.TestCase):
    def test_spec_plan_accepts_receipt_and_rejects_tampering_without_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text('{"text":"example"}\n', encoding="utf-8")
            receipt_path = root / "inspection-receipt.json"
            receipt = inspection_receipt_payload()
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            plan_path = root / "inspected-plan.json"

            self.assertEqual(
                main(
                    [
                        "spec-plan",
                        *fact_arguments(dataset),
                        "--inspection-receipt",
                        str(receipt_path),
                        "--output",
                        str(plan_path),
                    ]
                ),
                0,
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(
                plan["model_policy_decision_source"], "provider-inspection"
            )
            self.assertEqual(
                plan["inspection_receipt"]["receipt_id"], receipt["receipt_id"]
            )

            receipt_path.write_text(
                json.dumps({"status": "ok", "inspection_receipt": receipt}),
                encoding="utf-8",
            )
            wrapped_plan_path = root / "wrapped-inspection-plan.json"
            self.assertEqual(
                main(
                    [
                        "spec-plan",
                        *fact_arguments(dataset),
                        "--inspection-receipt",
                        str(receipt_path),
                        "--output",
                        str(wrapped_plan_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                json.loads(wrapped_plan_path.read_text(encoding="utf-8"))[
                    "inspection_receipt"
                ]["receipt_id"],
                receipt["receipt_id"],
            )

            receipt["observed_facts_sha256"] = "0" * 64
            receipt_path.write_text(
                json.dumps({"status": "ok", "inspection_receipt": receipt}),
                encoding="utf-8",
            )
            rejected_path = root / "rejected-plan.json"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = main(
                    [
                        "spec-plan",
                        *fact_arguments(dataset),
                        "--inspection-receipt",
                        str(receipt_path),
                        "--output",
                        str(rejected_path),
                    ]
                )

            self.assertEqual(result, 2)
            self.assertIn("receipt", stderr.getvalue().lower())
            self.assertFalse(rejected_path.exists())

            receipt_path.write_text(
                json.dumps({"status": "ok", "inspection_receipt": None}),
                encoding="utf-8",
            )
            missing_path = root / "missing-receipt-plan.json"
            with contextlib.redirect_stderr(io.StringIO()):
                result = main(
                    [
                        "spec-plan",
                        *fact_arguments(dataset),
                        "--inspection-receipt",
                        str(receipt_path),
                        "--output",
                        str(missing_path),
                    ]
                )
            self.assertEqual(result, 2)
            self.assertFalse(missing_path.exists())

            malformed_receipt = inspection_receipt_payload()
            malformed_receipt["provenance_summary"] = ["not-an-object"]
            receipt_path.write_text(
                json.dumps(malformed_receipt),
                encoding="utf-8",
            )
            malformed_path = root / "malformed-receipt-plan.json"
            with contextlib.redirect_stderr(io.StringIO()):
                result = main(
                    [
                        "spec-plan",
                        *fact_arguments(dataset),
                        "--inspection-receipt",
                        str(receipt_path),
                        "--output",
                        str(malformed_path),
                    ]
                )
            self.assertEqual(result, 2)
            self.assertFalse(malformed_path.exists())

    def test_profile_spec_plan_and_compile_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text('{"text":"example"}\n', encoding="utf-8")
            profile_path, plan_path, bundle = (
                root / "profile.json",
                root / "plan.json",
                root / "bundle",
            )
            self.assertEqual(
                main(
                    [
                        "profile",
                        "--dataset",
                        str(dataset),
                        "--output",
                        str(profile_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    ["spec-plan", *fact_arguments(dataset), "--output", str(plan_path)]
                ),
                0,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        ["compile", "--plan", str(plan_path), "--output", str(bundle)]
                    ),
                    0,
                )
            self.assertTrue(profile_path.is_file())
            self.assertEqual(
                json.loads(plan_path.read_text())["schema_version"],
                "aptus.training-plan.v5",
            )
            self.assertIsNone(
                json.loads(plan_path.read_text())["target"]["training_runtime"]
            )
            self.assertFalse(
                json.loads(plan_path.read_text())["hardware"]["devices"][0][
                    "supports_8bit"
                ]
            )
            self.assertTrue((bundle / "bundle-manifest.json").is_file())

    def test_compile_rejects_legacy_plan_without_rewriting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text('{"text":"example"}\n', encoding="utf-8")
            current_plan_path = root / "current-plan.json"
            self.assertEqual(
                main(
                    [
                        "spec-plan",
                        *fact_arguments(dataset),
                        "--output",
                        str(current_plan_path),
                    ]
                ),
                0,
            )
            current_payload = json.loads(current_plan_path.read_text(encoding="utf-8"))

            for index, found_schema in enumerate(
                ("aptus.training-plan.v3", "aptus.training-plan.v2", None)
            ):
                with self.subTest(found_schema=found_schema):
                    plan_path = root / f"legacy-plan-{index}.json"
                    bundle = root / f"bundle-{index}"
                    payload = dict(current_payload)
                    if found_schema is None:
                        payload.pop("schema_version")
                    else:
                        payload["schema_version"] = found_schema
                    plan_path.write_text(
                        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    before = plan_path.read_bytes()
                    stderr = io.StringIO()

                    with contextlib.redirect_stderr(stderr):
                        result = main(
                            [
                                "compile",
                                "--plan",
                                str(plan_path),
                                "--output",
                                str(bundle),
                            ]
                        )

                    self.assertEqual(result, 2)
                    self.assertIn("Replan required", stderr.getvalue())
                    self.assertEqual(plan_path.read_bytes(), before)
                    self.assertFalse(bundle.exists())

    def test_exact_qwen3_moe_flags_persist_derived_sparse_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text('{"text":"example"}\n', encoding="utf-8")
            plan_path = root / "moe-plan.json"
            arguments = fact_arguments(dataset)
            replacements = {
                "--model-id": "Qwen/Qwen3-30B-A3B-MLX-4bit",
                "--family": "qwen3_moe",
                "--parameters-b": "30.5",
                "--hidden-size": "2048",
                "--intermediate-size": "6144",
                "--layers": "48",
                "--context-length": "40960",
                "--vram-gib": "64",
                "--host-ram-gib": "64",
                "--prefer-method": "qlora",
            }
            for flag, value in replacements.items():
                if flag in arguments:
                    arguments[arguments.index(flag) + 1] = value
                else:
                    arguments.extend((flag, value))
            arguments.extend(
                (
                    "--model-type",
                    "qwen3_moe",
                    "--architecture",
                    "Qwen3MoeForCausalLM",
                    "--quantization-bits",
                    "4",
                    "--quantization-layout-profile",
                    "qwen3-moe-4bit-group64-router-gates-8bit",
                    "--moe-expert-count",
                    "128",
                    "--moe-experts-per-token",
                    "8",
                    "--moe-expert-intermediate-size",
                    "768",
                    "--moe-decoder-sparse-step",
                    "1",
                    "--backend",
                    "mps",
                    "--training-runtime",
                    "mlx-lm",
                )
            )

            self.assertEqual(
                main(["spec-plan", *arguments, "--output", str(plan_path)]),
                0,
            )

            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["schema_version"], "aptus.training-plan.v5")
            self.assertEqual(plan["model"]["model_type"], "qwen3_moe")
            self.assertEqual(plan["model"]["architecture"], "Qwen3MoeForCausalLM")
            self.assertEqual(plan["model"]["quantization_bits"], 4)
            self.assertEqual(
                len(plan["model"]["quantization_layout"]["module_overrides"]), 48
            )
            self.assertEqual(plan["model"]["moe"]["expert_count"], 128)
            self.assertEqual(plan["model"]["sparse_layer_count"], 48)
            self.assertLess(
                plan["model"]["active_parameters"], plan["model"]["parameters"]
            )
            self.assertEqual(plan["recommended"]["method"], "qlora")
            self.assertEqual(
                plan["recommended"]["runtime_contract"]["training_runtime"],
                "mlx-lm",
            )

    def test_partial_moe_topology_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text('{"text":"example"}\n', encoding="utf-8")
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "spec-plan",
                        *fact_arguments(dataset),
                        "--moe-expert-count",
                        "128",
                        "--output",
                        str(root / "plan.json"),
                    ]
                )

            self.assertEqual(code, 2)
            self.assertIn("--moe-experts-per-token", stderr.getvalue())
            self.assertIn("--moe-expert-intermediate-size", stderr.getvalue())
            self.assertIn("--moe-decoder-sparse-step", stderr.getvalue())

    def test_explicit_mlx_runtime_is_persisted_for_mps_planning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text('{"text":"example"}\n', encoding="utf-8")
            plan_path = root / "plan.json"
            arguments = fact_arguments(dataset)
            arguments.extend(("--backend", "mps"))
            arguments.extend(("--training-runtime", "mlx-lm"))

            self.assertEqual(
                main(["spec-plan", *arguments, "--output", str(plan_path)]),
                0,
            )

            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["target"]["training_runtime"], "mlx-lm")
            self.assertEqual(
                plan["recommended"]["runtime_contract"]["compute_backend"],
                "mps",
            )
            self.assertEqual(
                plan["recommended"]["runtime_contract"]["training_runtime"],
                "mlx-lm",
            )
            self.assertEqual(
                plan["hardware"]["reserve_per_device_bytes"],
                8 * 1024**3,
            )

    def test_inferred_mps_runtime_enforces_the_apple_memory_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text('{"text":"example"}\n', encoding="utf-8")
            plan_path = root / "plan.json"

            self.assertEqual(
                main(
                    [
                        "spec-plan",
                        *fact_arguments(dataset),
                        "--backend",
                        "mps",
                        "--output",
                        str(plan_path),
                    ]
                ),
                0,
            )

            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertIsNone(plan["target"]["training_runtime"])
            self.assertEqual(
                plan["recommended"]["runtime_contract"]["training_runtime"],
                "mlx-lm",
            )
            self.assertEqual(
                plan["hardware"]["reserve_per_device_bytes"],
                8 * 1024**3,
            )

    def test_explicit_training_runtime_must_match_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.txt"
            dataset.write_text("example\n", encoding="utf-8")
            for runtime, backend, required_backend in (
                ("mlx-lm", "cuda", "mps"),
                ("pytorch-mps", "cuda", "mps"),
                ("transformers-peft-cuda", "mps", "cuda"),
            ):
                stderr = io.StringIO()
                with (
                    self.subTest(runtime=runtime, backend=backend),
                    contextlib.redirect_stderr(stderr),
                ):
                    code = main(
                        [
                            "spec-plan",
                            *fact_arguments(dataset),
                            "--backend",
                            backend,
                            "--training-runtime",
                            runtime,
                            "--output",
                            str(root / f"{runtime}.json"),
                        ]
                    )
                    self.assertEqual(code, 2)
                    self.assertIn(
                        f"Training runtime {runtime} requires "
                        f"--backend {required_backend}.",
                        stderr.getvalue(),
                    )

    def test_combined_plan_flow_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text('{"text":"example"}\n', encoding="utf-8")
            output = root / "bundle"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["plan", *fact_arguments(dataset), "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertTrue(output.with_suffix(".zip").is_file())

    def test_explicit_zero_sequence_length_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.txt"
            dataset.write_text("example\n", encoding="utf-8")
            arguments = fact_arguments(dataset)
            index = arguments.index("--sequence-length") + 1
            arguments[index] = "0"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(["plan", *arguments, "--output", str(root / "bundle")])
            self.assertEqual(code, 2)
            self.assertIn("positive", stderr.getvalue())

    def test_sequence_length_is_an_explicit_required_fact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.txt"
            dataset.write_text("example\n", encoding="utf-8")
            arguments = fact_arguments(dataset)
            index = arguments.index("--sequence-length")
            del arguments[index : index + 2]
            with self.assertRaises(SystemExit):
                main(["spec-plan", *arguments, "--output", str(root / "plan.json")])

    def test_negative_intermediate_size_and_checkpoint_steps_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.txt"
            dataset.write_text("example\n", encoding="utf-8")
            for option, value in (
                ("--intermediate-size", "-1"),
                ("--checkpoint-steps", "-1"),
            ):
                arguments = fact_arguments(dataset)
                index = arguments.index(option) + 1
                arguments[index] = value
                with (
                    self.subTest(option=option),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(
                        main(
                            [
                                "spec-plan",
                                *arguments,
                                "--output",
                                str(root / f"{option}.json"),
                            ]
                        ),
                        2,
                    )

    def test_runtime_validation_uses_persisted_job_service(self) -> None:
        service = MagicMock()
        service.submit.return_value = {
            "id": "job_" + "a" * 32,
            "state": "completed",
        }
        with patch("aptus.execution.JobService", return_value=service) as job_service:
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "validate",
                        "/tmp/bundle",
                        "--level",
                        "model-data",
                        "--run",
                        "--state-dir",
                        "/tmp/aptus-state-test",
                    ]
                )
        self.assertEqual(code, 0)
        job_service.assert_called_once_with(Path("/tmp/aptus-state-test") / "jobs")
        service.submit.assert_called_once_with(Path("/tmp/bundle"), action="model-data")

    def test_job_prerequisite_failure_is_a_stable_cli_error(self) -> None:
        service = MagicMock()
        service.submit.side_effect = JobPrerequisiteError(
            action="pilot",
            required_state="measured-preflight-pass",
            current_state="model-data-pass",
            reason="insufficient_state",
        )
        stderr = io.StringIO()
        with (
            patch("aptus.execution.JobService", return_value=service),
            contextlib.redirect_stderr(stderr),
        ):
            code = main(["run", "/tmp/bundle", "--action", "pilot"])
        self.assertEqual(code, 2)
        self.assertIn("Aptus error: Cannot start pilot", stderr.getvalue())
        self.assertIn("measured-preflight-pass", stderr.getvalue())

    def test_ctrl_c_requests_owned_job_cancellation(self) -> None:
        service = MagicMock()
        job_id = "job_" + "b" * 32
        service.submit.return_value = {"id": job_id, "state": "queued"}
        service.get.side_effect = KeyboardInterrupt
        service.cancel.return_value = {"id": job_id, "state": "cancelled"}
        with patch("aptus.execution.JobService", return_value=service):
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["run", "/tmp/bundle"])
        self.assertEqual(code, 130)
        service.cancel.assert_called_once_with(job_id)

    def test_missing_training_permission_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.txt"
            dataset.write_text("example\n", encoding="utf-8")
            arguments = fact_arguments(dataset)
            arguments.remove("--confirm-training-allowed")
            with contextlib.redirect_stderr(io.StringIO()):
                code = main(["plan", *arguments, "--output", str(root / "bundle")])
            self.assertEqual(code, 2)

    def test_serve_blocks_accidental_non_loopback_execution_api(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(["serve", "--host", "0.0.0.0"])
        self.assertEqual(code, 2)
        self.assertIn("Non-loopback serving is blocked", stderr.getvalue())

    def test_doctor_and_diagnostics_commands_use_bounded_support_contracts(
        self,
    ) -> None:
        ready = {
            "status": "ready",
            "schema_version": "aptus.environment-doctor.v1",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stdout = io.StringIO()
            with (
                patch("aptus.diagnostics.build_doctor_report", return_value=ready),
                contextlib.redirect_stdout(stdout),
            ):
                self.assertEqual(main(["doctor", "--state-dir", str(root)]), 0)
            self.assertEqual(json.loads(stdout.getvalue()), ready)

            archive = root / "support.zip"
            with (
                patch(
                    "aptus.diagnostics.create_diagnostic_archive",
                    return_value=archive,
                ) as create,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            "diagnostics",
                            "--state-dir",
                            str(root),
                            "--output",
                            str(archive),
                        ]
                    ),
                    0,
                )
            create.assert_called_once_with(root, archive)

    def test_serve_generates_and_hands_off_an_authenticated_session(self) -> None:
        token = "generated-session-token-that-is-long-enough-123"
        application = object()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary) / "state"
            with (
                patch("aptus.cli.secrets.token_urlsafe", return_value=token),
                patch("aptus.api.create_app", return_value=application) as create,
                patch("uvicorn.run") as run_server,
                contextlib.redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "serve",
                        "--port",
                        "9001",
                        "--state-dir",
                        str(state_dir),
                    ]
                )

        self.assertEqual(code, 0)
        self.assertEqual(create.call_args.kwargs["session_token"], token)
        self.assertEqual(create.call_args.kwargs["state_dir"], state_dir)
        run_server.assert_called_once_with(
            application,
            host="127.0.0.1",
            port=9001,
            access_log=False,
        )
        self.assertIn(
            f"http://127.0.0.1:9001/?aptus_session_token={token}",
            stderr.getvalue(),
        )
        self.assertIn(f"Aptus API bearer token: {token}", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
