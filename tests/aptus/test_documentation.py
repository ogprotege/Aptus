from __future__ import annotations

import argparse
import json
import re
import unittest
from collections import deque
from pathlib import Path
from urllib.parse import unquote

from aptus.domain import Backend, Distribution, Method, TrainingRuntime
from aptus.cli import _parser
from aptus.methods.registry import method_descriptors
from tools.generate_openapi import render_openapi


REPOSITORY = Path(__file__).resolve().parents[2]
ROOT_DOCUMENTS = (
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "ROADMAP.md",
    "SECURITY.md",
    "SUPPORT.md",
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
API_ROUTE = re.compile(r'@app\.(?:get|post|put|patch|delete)\("([^"]+)"')
STATIC_ERROR_CODE = re.compile(r'["\']error["\']\s*:\s*["\']([^"\']+)["\']')
STALE_CONTRACTS = (
    "aptus.training-plan.v1",
    "aptus.bundle.v1",
    "aptus.validation.v1",
    "aptus.run.v1",
    "aptus-workbench-v1",
    "aptus-memory-v1",
)


def maintained_documentation() -> list[Path]:
    documents = [REPOSITORY / name for name in ROOT_DOCUMENTS]
    documents.extend(
        path
        for path in sorted((REPOSITORY / "docs").rglob("*.md"))
        if "audits/aptus-legacy" not in path.as_posix() or path.name == "README.md"
    )
    documents.extend(sorted((REPOSITORY / "examples").glob("*.md")))
    documents.extend(sorted((REPOSITORY / "Reference").glob("*.md")))
    return documents


def link_parts(raw: str) -> tuple[str, str]:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        value = value[1 : value.index(">")]
    else:
        value = value.split(maxsplit=1)[0]
    destination, separator, fragment = unquote(value).partition("#")
    return destination, fragment if separator else ""


def local_link_target(document: Path, raw: str) -> tuple[Path, str] | None:
    destination, fragment = link_parts(raw)
    if destination.startswith(("http://", "https://", "mailto:", "/")):
        return None
    resolved = (document.parent / destination).resolve() if destination else document
    return resolved, fragment


def markdown_anchors(document: Path) -> set[str]:
    anchors: set[str] = set()
    occurrences: dict[str, int] = {}
    in_fence = False
    for line in document.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = MARKDOWN_HEADING.match(line)
        if match is None:
            continue
        heading = re.sub(r"<[^>]+>", "", match.group(1))
        slug = re.sub(r"[^\w\-\s]", "", heading.lower())
        slug = re.sub(r"\s+", "-", slug.strip())
        duplicate = occurrences.get(slug, 0)
        occurrences[slug] = duplicate + 1
        anchors.add(f"{slug}-{duplicate}" if duplicate else slug)
    return anchors


def subparser_actions(
    parser: argparse.ArgumentParser,
) -> list[argparse._SubParsersAction]:
    return [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]


def cli_surface() -> tuple[set[str], set[str]]:
    commands: set[str] = set()
    options: set[str] = set()
    pending = deque([("aptus", _parser())])
    while pending:
        prefix, parser = pending.popleft()
        for action in parser._actions:
            options.update(
                option for option in action.option_strings if option != "--help"
            )
        for action in subparser_actions(parser):
            for name, child in action.choices.items():
                command = f"{prefix} {name}"
                commands.add(command)
                pending.append((command, child))
    return commands, options


class DocumentationTests(unittest.TestCase):
    def test_checked_openapi_contract_matches_the_application(self) -> None:
        artifact = REPOSITORY / "docs/reference/openapi.v1.json"
        self.assertTrue(artifact.is_file())
        rendered = render_openapi()
        self.assertEqual(artifact.read_text(encoding="utf-8"), rendered)
        schema = json.loads(rendered)
        self.assertEqual(schema["info"]["x-aptus-contract-version"], "aptus.api.v1")
        for path in (
            "/api/v1/bootstrap",
            "/api/v1/plan",
            "/api/v1/projects",
            "/api/v1/projects/{project_id}/recover",
            "/api/v1/jobs/{job_id}",
        ):
            self.assertIn(path, schema["paths"])
        for name in (
            "ProfileResponse",
            "ModelInspectionResponse",
            "InferenceModelsResponse",
            "InferenceGenerateResponse",
        ):
            self.assertFalse(
                schema["components"]["schemas"][name]["additionalProperties"]
            )
        for name in ("CompileRequest", "ValidateRequest", "JobRequest"):
            required = set(schema["components"]["schemas"][name]["required"])
            self.assertIn("project_id", required)
            self.assertIn("expected_project_revision_id", required)

    def test_model_compatibility_reference_matches_discriminated_contract(
        self,
    ) -> None:
        schema = json.loads(
            (REPOSITORY / "docs/reference/openapi.v1.json").read_text(encoding="utf-8")
        )
        schemas = schema["components"]["schemas"]
        compatibility = schemas["ModelCompatibilityResponse"]
        expected_mapping = {
            status: f"#/components/schemas/{name}"
            for status, name in (
                ("conditional", "ConditionalModelCompatibilityResponse"),
                ("recognized", "RecognizedModelCompatibilityResponse"),
                ("unsupported", "UnsupportedModelCompatibilityResponse"),
            )
        }
        self.assertEqual(
            compatibility["discriminator"],
            {"propertyName": "status", "mapping": expected_mapping},
        )
        self.assertEqual(
            {variant["$ref"] for variant in compatibility["oneOf"]},
            set(expected_mapping.values()),
        )

        required = {
            "status",
            "family",
            "supported_runtime",
            "supported_methods",
            "compute_backend",
            "distribution",
            "evidence_requirement",
            "adapter_profile_id",
            "reason",
        }
        conditional = schemas["ConditionalModelCompatibilityResponse"]
        recognized = schemas["RecognizedModelCompatibilityResponse"]
        unsupported = schemas["UnsupportedModelCompatibilityResponse"]
        for variant in (conditional, recognized, unsupported):
            self.assertFalse(variant["additionalProperties"])
            self.assertEqual(set(variant["required"]), required)

        self.assertEqual(conditional["properties"]["status"]["const"], "conditional")
        for field in ("family", "reason"):
            self.assertEqual(conditional["properties"][field]["type"], "string")
            self.assertEqual(conditional["properties"][field]["minLength"], 1)
            self.assertEqual(
                conditional["properties"][field]["pattern"],
                r"^\S(?:[\s\S]*\S)?$",
            )
        self.assertEqual(conditional["properties"]["supported_methods"]["minItems"], 1)

        def allowed_values(property_schema: dict[str, object]) -> set[str]:
            resolved = property_schema
            reference = resolved.get("$ref")
            if isinstance(reference, str):
                resolved = schemas[reference.rsplit("/", 1)[-1]]
            values = resolved.get("enum")
            if isinstance(values, list):
                return {str(value) for value in values}
            if "const" in resolved:
                return {str(resolved["const"])}
            self.fail(f"Compatibility property has no closed vocabulary: {resolved}")

        self.assertEqual(
            allowed_values(conditional["properties"]["supported_runtime"]),
            {item.value for item in TrainingRuntime},
        )
        self.assertEqual(
            allowed_values(conditional["properties"]["compute_backend"]),
            {item.value for item in Backend},
        )
        self.assertEqual(
            allowed_values(conditional["properties"]["supported_methods"]["items"]),
            {item.value for item in Method},
        )
        self.assertEqual(
            allowed_values(conditional["properties"]["distribution"]),
            {item.value for item in Distribution},
        )
        self.assertEqual(
            allowed_values(conditional["properties"]["adapter_profile_id"]),
            {"attention-qkvo.v1"},
        )
        self.assertEqual(
            conditional["properties"]["evidence_requirement"]["const"],
            "pilot-required",
        )
        for variant in (recognized, unsupported):
            self.assertEqual(variant["properties"]["supported_runtime"]["type"], "null")
            self.assertEqual(variant["properties"]["compute_backend"]["type"], "null")
            self.assertEqual(variant["properties"]["distribution"]["type"], "null")
            self.assertEqual(
                variant["properties"]["adapter_profile_id"]["type"], "null"
            )
            self.assertEqual(variant["properties"]["supported_methods"]["maxItems"], 0)
        self.assertEqual(
            recognized["properties"]["evidence_requirement"]["const"],
            "pilot-required",
        )
        self.assertEqual(
            unsupported["properties"]["evidence_requirement"]["const"],
            "implementation-required",
        )

        reference = (REPOSITORY / "docs/reference/api.md").read_text(encoding="utf-8")
        normalized_reference = " ".join(reference.split())
        self.assertIn("`conditional` requires", reference)
        self.assertIn("`recognized` carries no executable runtime", reference)
        self.assertIn("`unsupported` carries no executable runtime", reference)
        self.assertIn("`compute_backend`", reference)
        self.assertIn("`adapter_profile_id`", reference)
        self.assertIn("`attention-qkvo.v1`", reference)
        self.assertIn(
            "Unknown IDs and malformed combinations fail closed",
            reference,
        )
        self.assertIn("without copying the method registry", reference)
        self.assertIn("eligible for the reviewed pilot path", normalized_reference)

        claim_language = (REPOSITORY / "docs/product/claim-language.md").read_text(
            encoding="utf-8"
        )
        ui_contract = (REPOSITORY / "docs/product/ui-ux.md").read_text(encoding="utf-8")
        capabilities = (REPOSITORY / "docs/product/current-capabilities.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("eligible for the reviewed pilot path", claim_language)
        self.assertIn("based only on model inspection", claim_language)
        self.assertIn("selected runtime and backend match", ui_contract)
        self.assertIn("eligible for the reviewed pilot path", ui_contract)
        self.assertIn("`attention-qkvo.v1`", capabilities)

    def test_model_compatibility_policy_has_one_host_authority(self) -> None:
        policy = (REPOSITORY / "src/aptus/model_compatibility.py").read_text(
            encoding="utf-8"
        )
        inspection = (REPOSITORY / "src/aptus/inspection.py").read_text(
            encoding="utf-8"
        )
        planning = (REPOSITORY / "src/aptus/planning.py").read_text(encoding="utf-8")
        api_contracts = (REPOSITORY / "src/aptus/api_contracts.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("MODEL_COMPATIBILITY_POLICIES", policy)
        self.assertIn("def evaluate_model_compatibility(", policy)
        self.assertIn("def validate_execution_path_selection(", policy)
        self.assertIn("def validate_registered_compatibility_path(", policy)
        self.assertIn("def compatibility_response_v1(", policy)
        self.assertIn("runtime_contract_for", policy)
        self.assertIn("evaluate_model_compatibility", inspection)
        self.assertIn("compatibility_response_v1", inspection)
        self.assertNotIn("def _compatibility_response_v1(", inspection)
        self.assertNotIn("from .api_contracts import", inspection)
        self.assertIn("evaluate_model_compatibility", planning)
        self.assertIn("model_policy_rejection_reasons", planning)
        self.assertIn("def _estimate_candidate_with_policy(", planning)
        self.assertNotIn("is_exact_qwen3_moe", planning)
        self.assertNotIn("has_reviewed_qwen3_moe_quantization_layout", planning)
        self.assertIn("validate_registered_compatibility_path", api_contracts)
        self.assertNotIn("runtime_binding", api_contracts)

        code_map = (REPOSITORY / "docs/architecture/code-map.md").read_text(
            encoding="utf-8"
        )
        system = " ".join(
            (REPOSITORY / "docs/architecture/system.md")
            .read_text(encoding="utf-8")
            .split()
        )
        self.assertIn("model_compatibility.py", code_map)
        self.assertIn("same host-side model policy registry", system)
        self.assertIn("method registry constructs its runtime contract", system)

    def test_local_markdown_links_and_anchors_resolve(self) -> None:
        failures: list[str] = []
        anchor_cache: dict[Path, set[str]] = {}
        for document in maintained_documentation():
            text = document.read_text(encoding="utf-8")
            for raw in MARKDOWN_LINK.findall(text):
                target = local_link_target(document, raw)
                if target is None:
                    continue
                resolved, fragment = target
                if not resolved.exists():
                    failures.append(
                        f"{document.relative_to(REPOSITORY)} -> {link_parts(raw)[0]}"
                    )
                    continue
                if fragment and resolved.suffix.lower() == ".md":
                    anchors = anchor_cache.setdefault(
                        resolved, markdown_anchors(resolved)
                    )
                    if fragment not in anchors:
                        failures.append(
                            f"{document.relative_to(REPOSITORY)} -> "
                            f"{resolved.relative_to(REPOSITORY)}#{fragment}"
                        )
        self.assertEqual(failures, [], "Broken local documentation links found")

    def test_code_fences_are_balanced(self) -> None:
        failures = [
            str(document.relative_to(REPOSITORY))
            for document in maintained_documentation()
            if document.read_text(encoding="utf-8").count("```") % 2
        ]
        self.assertEqual(failures, [], "Unbalanced Markdown code fences found")

    def test_current_docs_do_not_reference_v1_contracts(self) -> None:
        failures: list[str] = []
        for document in maintained_documentation():
            text = document.read_text(encoding="utf-8")
            for contract in STALE_CONTRACTS:
                if contract in text:
                    failures.append(f"{document.relative_to(REPOSITORY)}: {contract}")
        self.assertEqual(failures, [], "Stale contract references found")

    def test_documentation_drift_audit_closeout_invariants(self) -> None:
        def normalized(relative_path: str) -> str:
            text = (REPOSITORY / relative_path).read_text(encoding="utf-8")
            return " ".join(text.split())

        requirements = {
            "docs/reference/validation-states.md": (
                "aptus.mlx-model-data-evidence.v1",
                "aptus.mlx-unified-memory-admission.v2",
                "2026-07-28-qwen3-moe-admission/README.md",
                "18.932 GiB",
                "validation report binds the artifact's current SHA-256",
            ),
            "docs/operations/operator-checklist.md": (
                "packed-checkpoint admission and `model-data-evidence.json`",
                "aptus.mlx-model-data-evidence.v1",
                "aptus.mlx-unified-memory-admission.v2",
                "before any weight load",
                "validation report binds the artifact's current SHA-256",
            ),
            "docs/methodology/preflight-calibration.md": (
                "model type, architecture class, expert topology, and canonical "
                "quantization layout",
                "aptus.mlx-model-data-evidence.v1",
                "aptus.mlx-unified-memory-admission.v2",
                "2026-07-28-qwen3-moe-admission/README.md",
                "validation report seals the current contents by binding the "
                "artifact's SHA-256",
            ),
        }
        failures: list[str] = []
        for relative_path, claims in requirements.items():
            text = normalized(relative_path)
            failures.extend(
                f"{relative_path}: {claim}" for claim in claims if claim not in text
            )
            if "immutable `model-data-evidence.json`" in text:
                failures.append(
                    f"{relative_path}: mutable runtime artifact called immutable"
                )
        self.assertEqual(failures, [], "MLX model-data closeout claims are incomplete")

        source = (REPOSITORY / "src/aptus/_bundle_programs/mlx/validate.py").read_text(
            encoding="utf-8"
        )
        model_data = source[source.index("def require_model_data(plan: dict)") :]
        ordered_steps = (
            "require_method_model(plan, candidate, model_path)",
            "require_unified_memory_admission(plan, model_path)",
            "model, tokenizer, config = load(",
            'evidence_path = ROOT / "model-data-evidence.json"',
        )
        positions = [model_data.index(step) for step in ordered_steps]
        self.assertEqual(positions, sorted(positions))

    def test_qwen3_documentation_slice_is_complete(self) -> None:
        def normalized(relative_path: str) -> str:
            text = (REPOSITORY / relative_path).read_text(encoding="utf-8")
            return " ".join(text.split())

        index = normalized("docs/index.md")
        for claim in (
            "Inspect the Qwen3 MoE admission attempt",
            "operations/evidence/2026-07-28-qwen3-moe-admission/README.md",
            "exact `qwen3_moe` MLX-LM QLoRA row remains conditional",
            "only safe-refusal evidence",
            "stopped before model loading",
        ):
            self.assertIn(claim, index)

        changing_contracts = normalized("docs/contributing/changing-contracts.md")
        for contract in (
            "aptus.model-architecture-contract.v1",
            "aptus.mlx-model-load-binding.v3",
            "aptus.mlx-model-parameter-census.v1",
            "aptus.mlx-packed-checkpoint.v1",
            "aptus.mlx-unified-memory-admission.v2",
            "aptus.mlx-model-data-evidence.v1",
        ):
            self.assertIn(contract, changing_contracts)

        code_map = normalized("docs/architecture/code-map.md")
        for claim in (
            "exact Qwen3 MoE identity",
            "portable MLX unified-memory formula",
            "model-architecture and quantization-layout contracts",
        ):
            self.assertIn(claim, code_map)

        registry = normalized("docs/reference/method-registry.md")
        for claim in (
            "`qwen3_moe`, whose targets are attention-only",
            "`mlx-lm.qlora.v1`",
            "`aptus-memory-mlx-v2`",
            "`mlx-lm-adapter`",
            "exact MoE model identity, architecture, and expert-topology policy",
            "canonical quantization-layout equality for the reviewed Qwen3 MoE slice",
        ):
            self.assertIn(claim, registry)

    def test_bundle_manifest_distinguishes_runtime_validation_ownership(self) -> None:
        manifest = " ".join(
            (REPOSITORY / "docs/reference/bundle-manifest.md")
            .read_text(encoding="utf-8")
            .split()
        )
        required_claims = (
            "CUDA cumulative `--level` executor and lease-bound action owner",
            "MLX argument-free Apple silicon and pinned-dependency gate with no lease",
            "In a CUDA bundle it binds selected device visibility, acquires the "
            "portable execution lease",
            "invokes `preflight.py --level <requested>` under the inherited lease token",
            "In an MLX-LM bundle, `validate.py` owns `--level`",
            "does not acquire the execution lease itself",
            "invokes the argument-free `preflight.py`",
            "`run.py --bounded-smoke` or `run.py --pilot`",
            "The MLX-LM bundle ships a different `preflight.py`. It takes no "
            "arguments, acquires no execution lease",
            "lease-owning MLX `run.py`",
        )
        missing = [claim for claim in required_claims if claim not in manifest]
        self.assertEqual(
            missing, [], "Bundle reference conflates CUDA and MLX ownership"
        )
        for stale_claim in (
            "Cumulative level executor used by validate.py",
            "It acquires the portable lease, invokes preflight.py at the requested level",
        ):
            self.assertNotIn(stale_claim, manifest)

        cuda_validate = (
            REPOSITORY / "src/aptus/_bundle_programs/cuda/validate.py"
        ).read_text(encoding="utf-8")
        cuda_preflight = (
            REPOSITORY / "src/aptus/_bundle_programs/cuda/preflight.py"
        ).read_text(encoding="utf-8")
        mlx_validate = (
            REPOSITORY / "src/aptus/_bundle_programs/mlx/validate.py"
        ).read_text(encoding="utf-8")
        mlx_preflight = (
            REPOSITORY / "src/aptus/_bundle_programs/mlx/preflight.py"
        ).read_text(encoding="utf-8")
        mlx_run = (REPOSITORY / "src/aptus/_bundle_programs/mlx/run.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("portable_execution_lease", cuda_validate)
        self.assertIn('str(ROOT / "preflight.py")', cuda_validate)
        self.assertIn('"--level"', cuda_validate)
        self.assertIn("portable_execution_lease", cuda_preflight)
        self.assertIn('parser.add_argument("--level"', cuda_preflight)

        self.assertNotIn("portable_execution_lease", mlx_validate)
        self.assertIn('parser.add_argument(\n        "--level"', mlx_validate)
        self.assertIn('"--bounded-smoke"', mlx_validate)
        self.assertIn('"--pilot"', mlx_validate)
        self.assertIn(
            'subprocess.run([sys.executable, str(ROOT / "preflight.py")], cwd=ROOT)',
            mlx_validate,
        )
        self.assertNotIn("argparse", mlx_preflight)
        self.assertNotIn("portable_execution_lease", mlx_preflight)
        self.assertIn("portable_execution_lease", mlx_run)

    def test_maintained_docs_have_review_metadata(self) -> None:
        failures: list[str] = []
        for document in maintained_documentation():
            metadata = document.read_text(encoding="utf-8")[:1600]
            missing: list[str] = []
            if not re.search(
                r"(?:\*\*(?:Documentation status|Status):\*\*|\| Status \|)",
                metadata,
            ):
                missing.append("status")
            if not re.search(
                r"(?:\*\*(?:Last reviewed|Snapshot):\*\*|"
                r"\| Last reviewed \|)",
                metadata,
            ):
                missing.append("last reviewed")
            if not re.search(
                r"(?:\*\*(?:Next scheduled review|Review by|Review):\*\*|"
                r"\| Next review \|)",
                metadata,
            ):
                missing.append("next review")
            if missing:
                failures.append(
                    f"{document.relative_to(REPOSITORY)}: {', '.join(missing)}"
                )
        self.assertEqual(failures, [], "Maintained-document metadata is incomplete")

    def test_maintained_docs_are_reachable_from_a_primary_index(self) -> None:
        documents = {path.resolve() for path in maintained_documentation()}
        pending = deque(
            [
                (REPOSITORY / "README.md").resolve(),
                (REPOSITORY / "docs/index.md").resolve(),
            ]
        )
        reached: set[Path] = set()
        while pending:
            document = pending.popleft()
            if document in reached or document not in documents:
                continue
            reached.add(document)
            for raw in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
                target = local_link_target(document, raw)
                if target is None:
                    continue
                resolved, _fragment = target
                if resolved in documents and resolved not in reached:
                    pending.append(resolved)
        unreachable = sorted(
            str(path.relative_to(REPOSITORY)) for path in documents - reached
        )
        self.assertEqual(unreachable, [], "Maintained docs are missing from navigation")

    def test_cli_reference_covers_executable_surface(self) -> None:
        reference = (REPOSITORY / "docs/reference/cli.md").read_text(encoding="utf-8")
        commands, options = cli_surface()
        missing_commands = sorted(
            command for command in commands if command not in reference
        )
        missing_options = sorted(
            option for option in options if option not in reference
        )
        self.assertEqual(
            missing_commands, [], "CLI commands are missing from the reference"
        )
        self.assertEqual(
            missing_options, [], "CLI options are missing from the reference"
        )

    def test_api_reference_covers_routes_and_static_errors(self) -> None:
        source = (REPOSITORY / "src/aptus/api.py").read_text(encoding="utf-8")
        api_reference = (REPOSITORY / "docs/reference/api.md").read_text(
            encoding="utf-8"
        )
        error_reference = (REPOSITORY / "docs/reference/error-codes.md").read_text(
            encoding="utf-8"
        )
        missing_routes = sorted(
            route
            for route in set(API_ROUTE.findall(source))
            if route not in api_reference
        )
        missing_errors = sorted(
            code
            for code in set(STATIC_ERROR_CODE.findall(source))
            if code not in error_reference
        )
        self.assertEqual(
            missing_routes, [], "API routes are missing from the reference"
        )
        self.assertEqual(
            missing_errors, [], "Static API errors are missing from the reference"
        )

    def test_method_reference_covers_runtime_method_ids(self) -> None:
        reference = (REPOSITORY / "docs/reference/method-registry.md").read_text(
            encoding="utf-8"
        )
        missing = sorted(
            method.value for method in Method if method.value not in reference
        )
        self.assertEqual(missing, [], "Runtime methods are missing from the reference")

    def test_method_catalog_matches_overlapping_runtime_contracts(self) -> None:
        catalog = json.loads(
            (REPOSITORY / "docs/methodology/method-catalog.json").read_text(
                encoding="utf-8"
            )
        )
        descriptors = {item.method_id: item for item in method_descriptors()}
        matrix = catalog["current_matrix"]
        matrix_methods = {item["method_label"] for item in matrix}
        selectable = {
            item.method_id for item in descriptors.values() if item.selectable
        }
        self.assertEqual(matrix_methods, selectable)

        failures: list[str] = []
        for item in matrix:
            descriptor = descriptors[item["method_label"]]
            for field in ("parameter_scope", "parameterization", "base_storage"):
                if item[field] != getattr(descriptor, field):
                    failures.append(f"{item['id']}: {field}")
            if not item["status"].startswith("unsupported"):
                if item["distribution"] not in descriptor.supported_distributions:
                    failures.append(f"{item['id']}: distribution")
                if item["export"] != descriptor.export_kind:
                    failures.append(f"{item['id']}: export")

        research_parameterizations = set(catalog["research_index"]["parameterizations"])
        for descriptor in descriptors.values():
            if (
                not descriptor.selectable
                and descriptor.method_id not in research_parameterizations
                and descriptor.parameterization not in research_parameterizations
            ):
                failures.append(f"{descriptor.method_id}: research parameterization")
        self.assertEqual(failures, [], "Method catalog has drifted from the registry")

    def test_docs_do_not_create_an_environment_inside_a_bundle(self) -> None:
        unsafe = re.compile(
            r"cd\s+[^\n]*bundle[^\n]*\n(?:[^\n]*\n){0,2}"
            r"(?:python|python3)\s+-m\s+venv\s+\.venv",
            flags=re.IGNORECASE,
        )
        failures = [
            str(document.relative_to(REPOSITORY))
            for document in maintained_documentation()
            if unsafe.search(document.read_text(encoding="utf-8"))
        ]
        self.assertEqual(
            failures, [], "Documentation creates an environment inside a sealed bundle"
        )


if __name__ == "__main__":
    unittest.main()
