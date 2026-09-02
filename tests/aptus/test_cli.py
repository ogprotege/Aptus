import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aptus.catalog import reviewed_gemma4_moe_quantization_layout
from aptus.cli import main
from aptus.domain import to_primitive
from aptus.execution import JobDispositionError, JobPrerequisiteError
from aptus.model_compatibility import (
    create_model_inspection_receipt,
    subject_from_model,
)
from aptus.profiling import build_model_spec
from tests.aptus.helpers import (
    GEMMA4_MOE_LAYERS,
    GEMMA4_MOE_MODEL_ID,
    GEMMA4_MOE_REVISION,
)


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
        "--free-vram-gib",
        "22",
        "--bf16",
        "--four-bit",
        "--host-ram-gib",
        "64",
        "--host-ram-free-gib",
        "56",
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


GEMMA4_MOE_LAYOUT_PROFILE = "gemma4-moe-4bit-group64-router-proj-8bit"
QWEN3_MOE_LAYOUT_PROFILE = "qwen3-moe-4bit-group64-router-gates-8bit"


def gemma4_moe_fact_arguments(dataset: Path) -> list[str]:
    arguments = fact_arguments(dataset)
    replacements = {
        "--model-id": GEMMA4_MOE_MODEL_ID,
        "--revision": GEMMA4_MOE_REVISION,
        "--family": "gemma4_moe",
        "--parameters-b": "25.2",
        "--hidden-size": "2816",
        "--intermediate-size": "2112",
        "--layers": str(GEMMA4_MOE_LAYERS),
        "--context-length": "262144",
        "--vram-gib": "64",
        "--host-ram-gib": "64",
        "--host-ram-free-gib": "56",
    }
    for flag, value in replacements.items():
        arguments[arguments.index(flag) + 1] = value
    arguments.extend(
        (
            "--prefer-method",
            "qlora",
            "--model-type",
            "gemma4_text",
            "--architecture",
            "Gemma4ForConditionalGeneration",
            "--quantization-bits",
            "4",
            "--moe-expert-count",
            "128",
            "--moe-experts-per-token",
            "8",
            "--moe-expert-intermediate-size",
            "704",
            "--moe-decoder-sparse-step",
            "1",
            "--backend",
            "mps",
            "--training-runtime",
            "mlx-lm",
        )
    )
    return arguments


def gemma4_moe_inspection_receipt_payload() -> dict[str, object]:
    model = build_model_spec(
        model_id=GEMMA4_MOE_MODEL_ID,
        revision=GEMMA4_MOE_REVISION,
        family="gemma4_moe",
        parameters_b=25.2,
        hidden_size=2816,
        intermediate_size=2112,
        layers=GEMMA4_MOE_LAYERS,
        context_length=262144,
        license_name="apache-2.0",
        training_allowed=True,
        architecture="Gemma4ForConditionalGeneration",
        model_type="gemma4_text",
        quantization_bits=4,
        quantization_layout=reviewed_gemma4_moe_quantization_layout(GEMMA4_MOE_LAYERS),
        moe={
            "expert_count": 128,
            "experts_per_token": 8,
            "expert_intermediate_size": 704,
            "decoder_sparse_step": 1,
            "mlp_only_layers": (),
            "shared_expert_intermediate_size": None,
        },
    )
    observed_at = "2026-09-02T12:00:00+00:00"
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
                else f"https://huggingface.co/{GEMMA4_MOE_MODEL_ID}/resolve/{GEMMA4_MOE_REVISION}/config.json"
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
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(
                    main(
                        [
                            "spec-plan",
                            *fact_arguments(dataset),
                            "--output",
                            str(plan_path),
                        ]
                    ),
                    0,
                )
            plan_payload = json.loads(plan_path.read_text())
            self.assertNotIn(
                "correction",
                plan_payload,
                "correction is presentation-only and must not enter plan JSON",
            )
            self.assertIn("Aptus correction", stderr.getvalue())
            self.assertIn("select-candidate", stderr.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        ["compile", "--plan", str(plan_path), "--output", str(bundle)]
                    ),
                    0,
                )
            self.assertTrue(profile_path.is_file())
            self.assertEqual(
                plan_payload["schema_version"],
                "aptus.training-plan.v6",
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

    def test_eval_generate_refuses_bundles_without_eval_program(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            gold = root / "gold.jsonl"
            gold.write_text(
                '{"prompt":"q","completion":"a","id":"1"}\n', encoding="utf-8"
            )
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapters.safetensors").write_bytes(b"not-real")
            self.assertEqual(
                main(
                    [
                        "eval-generate",
                        "--bundle",
                        str(bundle),
                        "--gold",
                        str(gold),
                        "--adapter",
                        str(adapter),
                        "--output",
                        str(root / "pred.jsonl"),
                    ]
                ),
                2,
            )

    def test_select_candidate_writes_new_plan_identity_without_clobbering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text('{"text":"example"}\n', encoding="utf-8")
            source_path = root / "plan.json"
            selected_path = root / "selected.json"
            self.assertEqual(
                main(
                    [
                        "spec-plan",
                        *fact_arguments(dataset),
                        "--optimizer-steps",
                        "128",
                        "--training-seed",
                        "101",
                        "--data-order-seed",
                        "1000101",
                        "--output",
                        str(source_path),
                    ]
                ),
                0,
            )
            source = json.loads(source_path.read_text(encoding="utf-8"))
            alternative = next(
                item
                for item in source["candidates"]
                if item["feasible"]
                and item["candidate_id"] != source["recommended"]["candidate_id"]
            )

            self.assertEqual(
                main(
                    [
                        "select-candidate",
                        "--plan",
                        str(source_path),
                        "--candidate-id",
                        alternative["candidate_id"],
                        "--output",
                        str(selected_path),
                    ]
                ),
                0,
            )
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            self.assertEqual(
                selected["recommended"]["candidate_id"],
                alternative["candidate_id"],
            )
            self.assertNotEqual(selected["plan_id"], source["plan_id"])
            original = selected_path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "select-candidate",
                            "--plan",
                            str(source_path),
                            "--candidate-id",
                            alternative["candidate_id"],
                            "--output",
                            str(selected_path),
                        ]
                    ),
                    2,
                )
            self.assertEqual(selected_path.read_bytes(), original)

    def test_compile_normalizes_json_parser_resource_errors(self) -> None:
        invalid_documents = (
            ("oversized-integer", '{"value":' + "9" * 5000 + "}\n"),
            (
                "excessive-nesting",
                '{"value":' + "[" * 10000 + "0" + "]" * 10000 + "}\n",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, contents in invalid_documents:
                with self.subTest(name=name):
                    plan_path = root / f"{name}.json"
                    bundle = root / f"{name}-bundle"
                    plan_path.write_text(contents, encoding="utf-8")
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
                    self.assertIn(
                        "Aptus error: Plan is unreadable or invalid JSON.",
                        stderr.getvalue(),
                    )
                    self.assertNotIn("RecursionError", stderr.getvalue())
                    self.assertFalse(bundle.exists())

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
                "--host-ram-free-gib": "56",
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
            self.assertEqual(plan["schema_version"], "aptus.training-plan.v6")
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

    def test_gemma4_26b_inspect_layout_is_nameable_and_receipt_matched(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text('{"text":"example"}\n', encoding="utf-8")
            plan_path = root / "gemma4-moe-plan.json"
            receipt_path = root / "gemma4-moe-receipt.json"
            receipt = gemma4_moe_inspection_receipt_payload()
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            expected_layout = to_primitive(
                reviewed_gemma4_moe_quantization_layout(GEMMA4_MOE_LAYERS)
            )

            self.assertEqual(
                main(
                    [
                        "spec-plan",
                        *gemma4_moe_fact_arguments(dataset),
                        "--quantization-layout-profile",
                        GEMMA4_MOE_LAYOUT_PROFILE,
                        "--inspection-receipt",
                        str(receipt_path),
                        "--output",
                        str(plan_path),
                    ]
                ),
                0,
            )

            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["model"]["family"], "gemma4_moe")
            self.assertEqual(plan["model"]["quantization_layout"], expected_layout)
            self.assertEqual(
                plan["model"]["quantization_layout"]["module_overrides"][0][
                    "module_path"
                ],
                "model.layers.0.router.proj",
            )
            self.assertTrue(
                all(
                    item["module_path"].endswith(".router.proj")
                    for item in plan["model"]["quantization_layout"]["module_overrides"]
                )
            )
            self.assertEqual(
                plan["model_policy_decision"]["policy_id"],
                "model.gemma4-moe.mlx.v1",
            )
            self.assertNotEqual(
                plan["model_policy_decision"]["policy_id"],
                "model.gemma4.mlx.v1",
            )
            self.assertEqual(
                plan["inspection_receipt"]["receipt_id"],
                receipt["receipt_id"],
            )
            self.assertEqual(
                plan["model_policy_decision_source"],
                "provider-inspection",
            )

    def test_qwen3_layout_profile_name_still_only_matches_qwen3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text('{"text":"example"}\n', encoding="utf-8")
            plan_path = root / "qwen3-named-gemma-plan.json"
            receipt_path = root / "gemma4-moe-receipt.json"
            receipt_path.write_text(
                json.dumps(gemma4_moe_inspection_receipt_payload()),
                encoding="utf-8",
            )
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "spec-plan",
                        *gemma4_moe_fact_arguments(dataset),
                        "--quantization-layout-profile",
                        QWEN3_MOE_LAYOUT_PROFILE,
                        "--inspection-receipt",
                        str(receipt_path),
                        "--output",
                        str(plan_path),
                    ]
                )

            self.assertEqual(code, 2)
            self.assertIn(
                "Inspection receipt observed facts do not match the plan facts.",
                stderr.getvalue(),
            )
            self.assertFalse(plan_path.exists())

    def test_dense_quantization_group_size_plans_reviewed_qwen2_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text('{"text":"example"}\n', encoding="utf-8")
            plan_path = root / "qwen2-plan.json"
            arguments = fact_arguments(dataset)
            replacements = {
                "--model-id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "--revision": "53a32aee5e9447773fd2b85988395066aef3700a",
                "--family": "qwen",
                "--parameters-b": "0.494",
                "--hidden-size": "896",
                "--intermediate-size": "4864",
                "--layers": "24",
                "--context-length": "32768",
                "--vram-gib": "64",
                "--host-ram-gib": "64",
                "--host-ram-free-gib": "56",
                "--effective-batch-size": "1",
            }
            for flag, value in replacements.items():
                arguments[arguments.index(flag) + 1] = value
            arguments.extend(
                (
                    "--model-type",
                    "qwen2",
                    "--architecture",
                    "Qwen2ForCausalLM",
                    "--quantization-bits",
                    "4",
                    "--quantization-group-size",
                    "64",
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
            self.assertEqual(
                plan["model"]["quantization_layout"],
                {
                    "default_bits": 4,
                    "default_group_size": 64,
                    "module_overrides": [],
                },
            )
            self.assertEqual(
                plan["model_policy_decision"]["policy_id"],
                "model.qwen2-24l.mlx-qlora",
            )
            self.assertEqual(
                plan["recommended"]["policy_binding"]["path_id"],
                "mlx-lm.qlora.single.dense-causal-lm.v1",
            )

    def test_quantization_group_size_requires_bits_and_excludes_profiles(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "data.jsonl"
            dataset.write_text('{"text":"example"}\n', encoding="utf-8")
            cases = (
                ("missing bits", ("--quantization-group-size", "64")),
                (
                    "zero group size",
                    ("--quantization-bits", "4", "--quantization-group-size", "0"),
                ),
                (
                    "negative group size",
                    ("--quantization-bits", "4", "--quantization-group-size", "-1"),
                ),
                (
                    "profile conflict",
                    (
                        "--quantization-bits",
                        "4",
                        "--quantization-group-size",
                        "64",
                        "--quantization-layout-profile",
                        "qwen3-moe-4bit-group64-router-gates-8bit",
                    ),
                ),
            )
            for name, extra in cases:
                stderr = io.StringIO()
                output = Path(temporary) / f"{name}.json"
                with self.subTest(name=name), contextlib.redirect_stderr(stderr):
                    code = main(
                        [
                            "spec-plan",
                            *fact_arguments(dataset),
                            *extra,
                            "--output",
                            str(output),
                        ]
                    )
                    self.assertEqual(code, 2)
                    self.assertIn("quantization-group-size", stderr.getvalue())
                    self.assertFalse(output.exists())

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

    def test_jobs_id_prints_training_signal_correction_block(self) -> None:
        service = MagicMock()
        job_id = "job_" + "r" * 32
        service.get.return_value = {
            "schema_version": "aptus.job-record.v1",
            "id": job_id,
            "job_id": job_id,
            "state": "completed",
            "action": "train",
            "bundle_dir": "/tmp/bundle",
            "created_at": "2026-01-01T00:00:00+00:00",
            "run_correction": {
                "schema_version": "aptus.run-correction.v1",
                "kind": "loss-flat",
                "summary": "Train loss stayed relatively flat.",
                "source": "train_loss_observations+validation_loss_observations",
                "next_plan_hints": [
                    {
                        "fact": "target.max_epochs",
                        "direction": "increase",
                        "why": "Train loss stayed flat.",
                    }
                ],
                "disallowed_suggestions": [
                    {
                        "code": "no_automl",
                        "message": "Do not start a hyperparameter search.",
                    }
                ],
                "operator_next_step": {
                    "action": "replan-with-fact-hints",
                    "label": "Replan with more epochs or review rank",
                },
                "non_claims": [
                    "Training loss is not model quality.",
                    "Validation split loss is not an aptus.evaluation-result.v1 decision.",
                ],
            },
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("aptus.execution.JobService", return_value=service),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = main(["jobs", "--id", job_id])
        self.assertEqual(code, 0)
        self.assertIn(
            "Aptus training-signal correction (presentation only; not quality):",
            stderr.getvalue(),
        )
        self.assertIn("kind: loss-flat", stderr.getvalue())
        self.assertIn(
            "non_claim: Training loss is not model quality.", stderr.getvalue()
        )
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["run_correction"]["kind"], "loss-flat")

    def test_dispose_writes_disposition_and_prints_block(self) -> None:
        service = MagicMock()
        job_id = "job_" + "d" * 32
        service.save_disposition.return_value = {
            "schema_version": "aptus.job-record.v1",
            "id": job_id,
            "job_id": job_id,
            "state": "completed",
            "action": "train",
            "bundle_dir": "/tmp/bundle",
            "created_at": "2026-01-01T00:00:00+00:00",
            "run_disposition": {
                "schema_version": "aptus.run-disposition.v1",
                "kind": "use",
                "source": "operator-attested",
                "operator_next_step": {
                    "action": "load-adapter",
                    "label": "Load this adapter",
                },
            },
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("aptus.execution.JobService", return_value=service) as job_service,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = main(
                [
                    "dispose",
                    job_id,
                    "--kind",
                    "use",
                    "--state-dir",
                    "/tmp/aptus-state-test",
                ]
            )
        self.assertEqual(code, 0)
        job_service.assert_called_once_with(Path("/tmp/aptus-state-test") / "jobs")
        service.save_disposition.assert_called_once_with(job_id, "use")
        self.assertIn(
            "Aptus run disposition (operator-attested; not quality):",
            stderr.getvalue(),
        )
        self.assertIn("kind: use", stderr.getvalue())
        self.assertIn("next: load-adapter — Load this adapter", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["run_disposition"]["kind"], "use")

    def test_dispose_refuses_without_kind(self) -> None:
        job_id = "job_" + "k" * 32
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                main(["dispose", job_id])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--kind", stderr.getvalue())

    def test_jobs_id_prints_disposition_block(self) -> None:
        service = MagicMock()
        job_id = "job_" + "s" * 32
        service.get.return_value = {
            "schema_version": "aptus.job-record.v1",
            "id": job_id,
            "job_id": job_id,
            "state": "completed",
            "action": "train",
            "bundle_dir": "/tmp/bundle",
            "created_at": "2026-01-01T00:00:00+00:00",
            "run_disposition": {
                "schema_version": "aptus.run-disposition.v1",
                "kind": "done",
                "source": "operator-attested",
                "operator_next_step": {
                    "action": "none",
                    "label": "I'm finished training this.",
                },
            },
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("aptus.execution.JobService", return_value=service),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = main(["jobs", "--id", job_id])
        self.assertEqual(code, 0)
        self.assertIn(
            "Aptus run disposition (operator-attested; not quality):",
            stderr.getvalue(),
        )
        self.assertIn("kind: done", stderr.getvalue())
        self.assertIn(
            "next: none — I'm finished training this.",
            stderr.getvalue(),
        )
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["run_disposition"]["kind"], "done")

    def test_dispose_maps_disposition_error_to_aptus_error(self) -> None:
        service = MagicMock()
        service.save_disposition.side_effect = JobDispositionError(
            "Cannot attest a run disposition unless the job action is train "
            "and the state is completed; observed action='pilot' "
            "state='completed'."
        )
        stderr = io.StringIO()
        with (
            patch("aptus.execution.JobService", return_value=service),
            contextlib.redirect_stderr(stderr),
        ):
            code = main(["dispose", "job_" + "f" * 32, "--kind", "use"])
        self.assertEqual(code, 2)
        self.assertIn("Aptus error:", stderr.getvalue())
        self.assertIn("Cannot attest a run disposition", stderr.getvalue())

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
        self.assertIn("Aptus workbench: http://127.0.0.1:9001/", stderr.getvalue())
        self.assertNotIn("aptus_session_token", stderr.getvalue())
        self.assertNotIn(f"?aptus_session_token={token}", stderr.getvalue())
        self.assertIn(f"Aptus API bearer token: {token}", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
