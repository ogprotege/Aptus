from __future__ import annotations

import argparse
import hashlib
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


def repository_markdown_documents() -> list[Path]:
    documents = {REPOSITORY / name for name in ROOT_DOCUMENTS}
    for directory in ("docs", "Reference", "examples", "dev"):
        documents.update((REPOSITORY / directory).rglob("*.md"))
    documents.update(
        {
            REPOSITORY / ".github/PULL_REQUEST_TEMPLATE.md",
            REPOSITORY / "desktop/macos/README.md",
        }
    )
    return sorted(documents)


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


def cli_parser_contract() -> dict[str, dict[str, dict[str, object]]]:
    commands: dict[str, dict[str, dict[str, object]]] = {}
    pending = deque([("aptus", _parser())])
    while pending:
        prefix, parser = pending.popleft()
        arguments: dict[str, dict[str, object]] = {}
        for action in parser._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            has_default = action.default != argparse.SUPPRESS
            argument = (
                " / ".join(action.option_strings)
                if action.option_strings
                else f"<{action.dest}>"
            )
            contract: dict[str, object] = {}
            if has_default:
                default = action.default
                contract["default"] = (
                    default.as_posix() if isinstance(default, Path) else default
                )
            if action.choices is not None:
                contract["choices"] = list(action.choices)
            arguments[argument] = contract
        commands[prefix] = arguments
        for action in subparser_actions(parser):
            for name, child in action.choices.items():
                command = f"{prefix} {name}"
                pending.append((command, child))
    return commands


def documented_cli_parser_contract(
    reference: str,
) -> dict[str, dict[str, dict[str, object]]]:
    start = "<!-- aptus-cli-parser-contract:start -->"
    end = "<!-- aptus-cli-parser-contract:end -->"
    _before, separator, remainder = reference.partition(start)
    if not separator:
        raise AssertionError("CLI parser contract start marker is missing")
    payload, separator, _remainder = remainder.partition(end)
    if not separator:
        raise AssertionError("CLI parser contract end marker is missing")
    fenced = payload.strip()
    if not fenced.startswith("```json\n") or not fenced.endswith("\n```"):
        raise AssertionError("CLI parser contract must be one fenced JSON document")
    document = json.loads(fenced.removeprefix("```json\n").removesuffix("\n```"))
    if set(document) != {"schema_version", "argument_groups", "commands"}:
        raise AssertionError("CLI parser contract has unexpected top-level fields")
    if document["schema_version"] != "aptus.cli-parser-contract.v1":
        raise AssertionError("CLI parser contract schema version is unsupported")

    groups = document["argument_groups"]
    command_documents = document["commands"]
    if not isinstance(groups, dict) or not isinstance(command_documents, dict):
        raise AssertionError("CLI parser contract groups and commands must be objects")
    referenced_groups: set[str] = set()
    commands: dict[str, dict[str, dict[str, object]]] = {}
    for command, command_document in command_documents.items():
        if not isinstance(command_document, dict):
            raise AssertionError(f"CLI parser contract for {command} must be an object")
        local_document = dict(command_document)
        group_names = local_document.pop("$groups", [])
        if not isinstance(group_names, list) or not all(
            isinstance(name, str) for name in group_names
        ):
            raise AssertionError(
                f"CLI parser contract groups for {command} are invalid"
            )
        arguments: dict[str, dict[str, object]] = {}
        for group_name in group_names:
            if group_name not in groups:
                raise AssertionError(
                    f"CLI parser contract group {group_name!r} is not defined"
                )
            referenced_groups.add(group_name)
            overlap = arguments.keys() & groups[group_name].keys()
            if overlap:
                raise AssertionError(
                    f"CLI parser contract repeats grouped arguments for {command}: "
                    f"{sorted(overlap)}"
                )
            arguments.update(groups[group_name])
        overlap = arguments.keys() & local_document.keys()
        if overlap:
            raise AssertionError(
                f"CLI parser contract repeats local arguments for {command}: "
                f"{sorted(overlap)}"
            )
        arguments.update(local_document)
        commands[command] = arguments
    if referenced_groups != set(groups):
        raise AssertionError(
            "CLI parser contract contains unreferenced argument groups"
        )
    return commands


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
        phase_three_provenance_fields = {
            "TrainingPlanResponse": {
                "schema_version",
                "plan_id",
                "recommended",
                "candidates",
                "warnings",
                "recommendation_rationale",
                "model_policy_decision",
                "model_policy_decision_source",
                "inspection_receipt",
            },
            "PlanCandidateResponse": {
                "candidate_id",
                "model_policy_decision_id",
                "policy_binding",
            },
        }
        for name, expected_fields in phase_three_provenance_fields.items():
            required = set(schema["components"]["schemas"][name]["required"])
            self.assertTrue(expected_fields.issubset(required), (name, required))
        self.assertEqual(
            schema["components"]["schemas"]["TrainingPlanResponse"]["properties"][
                "schema_version"
            ]["const"],
            "aptus.training-plan.v5",
        )

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
            {"attention-qkvo.v1", "dense-causal-lm.v1"},
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
        self.assertIn(
            "`inspection_receipt.decision` as its single inspection-time "
            "model-policy source",
            normalized_reference,
        )
        self.assertIn("legacy browser `compatibility` normalizer", reference)
        self.assertIn(
            "it is not the workbench policy authority",
            normalized_reference,
        )
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
        self.assertIn(
            "The separate Model policy panel presents the v2 decision",
            ui_contract,
        )
        self.assertIn("The rail is a topology and", ui_contract)
        self.assertNotIn("selected runtime and backend match", ui_contract)
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

    def test_phase4_portable_policy_docs_match_persisted_contracts(self) -> None:
        plan_schema = (REPOSITORY / "docs/reference/plan-schema.md").read_text(
            encoding="utf-8"
        )
        for contract in (
            "aptus.training-plan.v5",
            "aptus.model-policy-snapshot.v1",
            "aptus.model-compatibility.v2",
            "aptus.model-inspection-receipt.v1",
            "aptus.model-policy-binding.v1",
            "aptus.runtime-contract.v1",
        ):
            self.assertIn(contract, plan_schema)
        for identity in (
            "model.qwen3-moe.mlx-qlora",
            "1.0.0",
            "mlx-lm.qlora.single.attention-qkvo.v1",
        ):
            self.assertIn(identity, plan_schema)
        for field in (
            "subject_facts_sha256",
            "observed_facts_sha256",
            "model_policy_decision_source",
            "model_policy_decision_id",
            "policy_binding",
            "model_policy_snapshot_sha256",
        ):
            self.assertIn(field, plan_schema)
        self.assertIn("Every candidate carries the plan decision link", plan_schema)
        self.assertIn('`"policy_binding": null`', plan_schema)
        self.assertIn("`parameters` and `training_allowed` never", plan_schema)
        self.assertIn("tamper-evident content bindings, not authenticated", plan_schema)
        normalized_plan_schema = " ".join(plan_schema.split())
        self.assertIn(
            "Every v4, v3, v2, or schema-less plan requires replanning",
            normalized_plan_schema,
        )
        self.assertIn(
            "A coherent v5 plan also requires replanning after a snapshot-digest "
            "or policy-semantic change",
            normalized_plan_schema,
        )

        api = (REPOSITORY / "docs/reference/api.md").read_text(encoding="utf-8")
        cli = (REPOSITORY / "docs/reference/cli.md").read_text(encoding="utf-8")
        self.assertIn("`inspection_receipt`", api)
        self.assertIn("never falls back to the user-attested path", api)
        self.assertIn("`--inspection-receipt PATH`", cli)
        self.assertIn("aptus.training-plan.v5", cli)

        error_codes = (REPOSITORY / "docs/reference/error-codes.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("`aptus.training-plan.v5` contract", error_codes)
        self.assertIn("coherent v5 plan", error_codes)
        self.assertIn("current host registry", error_codes)
        self.assertIn("downgrading a bad receipt", error_codes)

        ui_contract = (REPOSITORY / "docs/product/ui-ux.md").read_text(encoding="utf-8")
        self.assertIn("Saved v4, v3, and v2 plans", ui_contract)
        self.assertIn("plans with no schema identifier", ui_contract)
        self.assertIn("`aptus.model-inspection-receipt.v1`", ui_contract)
        self.assertIn("only the exact registered path", ui_contract)
        self.assertIn("Phase 4", ui_contract)
        self.assertIn("Phase 5", ui_contract)

        overview = (REPOSITORY / "docs/methodology/overview.md").read_text(
            encoding="utf-8"
        )
        for unchanged in (
            "aptus.api.v1",
            "aptus.facts.v3",
            "aptus.runtime-contract.v1",
        ):
            self.assertIn(unchanged, overview)

        system = (REPOSITORY / "docs/architecture/system.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Phase 4", system)
        self.assertIn("Phase 5", system)
        self.assertIn("aptus.model-policy-snapshot.v1", system)
        self.assertIn("aptus.training-plan.v5", system)
        self.assertIn("aptus.bundle.v3", system)

        bundle = (REPOSITORY / "docs/reference/bundle-manifest.md").read_text(
            encoding="utf-8"
        )
        for portable_contract in (
            "policy/model-policy-snapshot.v1.json",
            "policy_snapshot_sha256",
            "generic policy evaluator",
            "does not import Aptus",
        ):
            self.assertIn(portable_contract, bundle)

        debt = (REPOSITORY / "docs/maintenance/documentation-debt.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Phase 2 intentionally preserves `aptus.api.v1`", debt)
        self.assertIn("`aptus.training-plan.v3`", debt)
        self.assertIn("### DOC-021: Bind model-policy provenance", debt)

        health = (REPOSITORY / "docs/maintenance/documentation-health.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("The historical Phase 3 implementation added", health)
        self.assertIn("`aptus.training-plan.v4`", health)
        self.assertIn("The current Phase 4 contract uses", health)
        self.assertIn("`aptus.training-plan.v5`", health)

    def test_policy_snapshot_finding_codes_are_fully_documented(self) -> None:
        validation = (REPOSITORY / "src/aptus/validation.py").read_text(
            encoding="utf-8"
        )
        source_codes = set(re.findall(r'"(POLICY_SNAPSHOT_[A-Z_]+)"', validation))
        expected_codes = {
            "POLICY_SNAPSHOT_CONTRACT",
            "POLICY_SNAPSHOT_DIGEST",
            "POLICY_SNAPSHOT_JSON_ERROR",
            "POLICY_SNAPSHOT_MISSING",
            "POLICY_SNAPSHOT_NONCANONICAL",
            "POLICY_SNAPSHOT_PATH",
        }
        self.assertEqual(source_codes, expected_codes)

        error_codes = (REPOSITORY / "docs/reference/error-codes.md").read_text(
            encoding="utf-8"
        )
        actual_rows = {
            code: next(line for line in error_codes.splitlines() if f"`{code}`" in line)
            for code in source_codes
        }
        expected_rows = {
            "POLICY_SNAPSHOT_MISSING": (
                "| `POLICY_SNAPSHOT_MISSING` | error | "
                "`policy/model-policy-snapshot.v1.json` | "
                "The required snapshot file is absent |"
            ),
            "POLICY_SNAPSHOT_JSON_ERROR": (
                "| `POLICY_SNAPSHOT_JSON_ERROR` | error | "
                "`policy/model-policy-snapshot.v1.json` | Snapshot bytes cannot "
                "be read or decoded and parsed as valid UTF-8 JSON |"
            ),
            "POLICY_SNAPSHOT_CONTRACT": (
                "| `POLICY_SNAPSHOT_CONTRACT` | error | "
                "`policy/model-policy-snapshot.v1.json` | The parsed value is not "
                "an exact valid `aptus.model-policy-snapshot.v1`, including JSON "
                "`null` or malformed constraint operands |"
            ),
            "POLICY_SNAPSHOT_NONCANONICAL": (
                "| `POLICY_SNAPSHOT_NONCANONICAL` | error | "
                "`policy/model-policy-snapshot.v1.json` | Snapshot bytes differ "
                "from the deterministic canonical JSON encoding |"
            ),
            "POLICY_SNAPSHOT_DIGEST": (
                "| `POLICY_SNAPSHOT_DIGEST` | error | "
                "`policy/model-policy-snapshot.v1.json` | One or more `snapshot`, "
                "`plan`, `manifest`, or `host` bindings is not lowercase "
                "64-character hexadecimal text, or a valid `plan`, `manifest`, "
                "or `host` binding differs from the snapshot; the finding message "
                "names each invalid and differing binding |"
            ),
            "POLICY_SNAPSHOT_PATH": (
                "| `POLICY_SNAPSHOT_PATH` | error | `bundle-manifest.json` | "
                "`policy_snapshot_path` is not exactly "
                "`policy/model-policy-snapshot.v1.json` |"
            ),
        }
        self.assertEqual(actual_rows, expected_rows)

    def test_every_plan_identity_enumeration_binds_the_snapshot_digest(self) -> None:
        documents = (
            "docs/reference/plan-schema.md",
            "docs/reference/evidence-records.md",
            "docs/architecture/data-and-identity-flow.md",
            "docs/methodology/candidate-enumeration.md",
            "docs/methodology/method-taxonomy.md",
            "docs/contributing/changing-contracts.md",
        )
        for relative_path in documents:
            text = (REPOSITORY / relative_path).read_text(encoding="utf-8")
            paragraphs = [
                " ".join(paragraph.split()) for paragraph in re.split(r"\n\s*\n", text)
            ]
            identity_enumerations = [
                paragraph
                for paragraph in paragraphs
                if re.search(
                    r"\b(?:plan ID|plan identity)\b.*\b"
                    r"(?:binds?|hashes?|includes?|contains?|covers?|comprises?)\b",
                    paragraph,
                    re.I,
                )
            ]
            self.assertTrue(identity_enumerations, relative_path)
            missing_digest = [
                paragraph
                for paragraph in identity_enumerations
                if "model_policy_snapshot_sha256" not in paragraph
                and "snapshot digest" not in paragraph.lower()
            ]
            self.assertEqual(
                missing_digest,
                [],
                f"Plan identity enumeration omits snapshot digest: {relative_path}",
            )

    def test_policy_docs_separate_portable_integrity_from_host_currency(
        self,
    ) -> None:
        for relative_path in (
            "README.md",
            "docs/architecture/system.md",
            "docs/architecture/data-and-identity-flow.md",
            "docs/contributing/changing-contracts.md",
            "docs/maintenance/documentation-debt.md",
            "docs/maintenance/documentation-health.md",
            "docs/methodology/overview.md",
            "docs/product/current-capabilities.md",
            "docs/product/ui-ux.md",
            "docs/reference/api.md",
            "docs/reference/bundle-manifest.md",
            "docs/reference/cli.md",
            "docs/reference/error-codes.md",
            "docs/reference/plan-schema.md",
            "docs/reference/validation-states.md",
        ):
            text = " ".join(
                (REPOSITORY / relative_path).read_text(encoding="utf-8").lower().split()
            )
            self.assertRegex(text, r"frozen[- ]snapshot", relative_path)
            for term in ("integrity", "currency"):
                self.assertIn(term, text, relative_path)
            self.assertRegex(
                text,
                r"(?:portable|package-free).{0,800}"
                r"(?:cannot|does not|\bnot\b).{0,200}"
                r"(?:currency|current[- ](?:host[- ])?registry)",
                relative_path,
            )
            self.assertRegex(
                text,
                r"installed[- ](?:host|aptus)",
                relative_path,
            )
            self.assertRegex(
                text,
                r"current[- ](?:host[- ])?registry",
                relative_path,
            )

    def test_phase4_current_contract_wording_preserves_phase3_history(self) -> None:
        required_current_fragments = {
            "README.md": (
                "Write a persisted v5 plan JSON without compiling",
                "Every v5 plan persists",
            ),
            "docs/reference/api.md": (
                "required v5 schema",
                "The OpenAPI response requires the v5 schema",
                "Create a new v5 plan from the source facts",
            ),
            "docs/reference/cli.md": (
                "Write a standalone v5 plan",
                "exact v5 domain contract",
            ),
            "docs/reference/error-codes.md": ("create a new v5 plan",),
            "docs/reference/evidence-records.md": (
                "Every candidate in an `aptus.training-plan.v5` plan",
                "The v5 plan ID binds",
            ),
            "docs/reference/plan-schema.md": (
                "The current schema identifier is `aptus.training-plan.v5`",
                "Every v4, v3, v2, or schema-less plan requires replanning",
            ),
            "docs/reference/validation-states.md": (
                "Every v4, v3, v2, or schema-less plan requires replanning",
            ),
            "docs/architecture/artifact-compiler.md": (
                "Valid `aptus.training-plan.v5` payload",
                "The installed Aptus host compiler",
            ),
            "docs/architecture/data-and-identity-flow.md": (
                "`aptus.training-plan.v5` model payload",
                "compare a v5 decision and snapshot digest",
            ),
            "docs/architecture/system.md": (
                "`aptus.training-plan.v5` and `aptus.bundle.v3` cross-bind",
            ),
            "docs/contributing/changing-contracts.md": (
                "The current plan reader accepts only `aptus.training-plan.v5`",
            ),
            "docs/maintenance/documentation-health.md": (
                "The historical Phase 3 implementation added",
                "The current Phase 4 contract uses `aptus.training-plan.v5`",
            ),
            "docs/methodology/overview.md": (
                "the plan contract from `aptus.training-plan.v4` to "
                "`aptus.training-plan.v5`",
            ),
            "docs/product/current-capabilities.md": (
                "Persisted `aptus.training-plan.v5` compatibility provenance",
            ),
            "docs/product/ui-ux.md": (
                "creates a new deterministic v5 plan",
                "The `aptus.training-plan.v5` carries one",
            ),
            "docs/methodology/facts-and-provenance.md": (
                "Changing an input requires a new `aptus.training-plan.v5` plan",
            ),
        }
        stale_current_v4_claim = re.compile(
            r"\b(?:create|emit|generate|produce|write|persist|rehydrate|carry|use|"
            r"accept|require)\w*\b.{0,80}\b"
            r"(?:a |an )?(?:current |active |new |deterministic |standalone |"
            r"persisted )*v4 plan\b",
            re.I,
        )
        for relative_path, fragments in required_current_fragments.items():
            text = " ".join(
                (REPOSITORY / relative_path).read_text(encoding="utf-8").split()
            )
            self.assertIn("aptus.training-plan.v5", text, relative_path)
            self.assertIsNone(stale_current_v4_claim.search(text), relative_path)
            for fragment in fragments:
                self.assertIn(fragment, text, relative_path)

        debt = (REPOSITORY / "docs/maintenance/documentation-debt.md").read_text(
            encoding="utf-8"
        )
        phase3_record = debt.split(
            "### DOC-021: Bind model-policy provenance into persisted plans", 1
        )[1].split("### DOC-022: Make the model-policy contract portable", 1)[0]
        self.assertIn("Phase 3 intentionally preserves", phase3_record)
        self.assertIn("`aptus.training-plan.v4`", phase3_record)
        self.assertIn("`aptus.bundle.v2`", phase3_record)

    def test_phase4_static_and_bundle_migration_docs_do_not_drift(self) -> None:
        validation_states = (
            REPOSITORY / "docs/reference/validation-states.md"
        ).read_text(encoding="utf-8")
        static_section = validation_states.split("### Static", 1)[1].split(
            "### Dependency", 1
        )[0]
        normalized_static_section = " ".join(static_section.split())
        for expected in (
            "`policy_snapshot.py`",
            "For MLX bundles, the host also parses `reload.py`",
            "The generated CUDA validator parses the seven-file list above",
            "The generated MLX validator's `static` level",
            "it does not AST-parse generated programs",
        ):
            self.assertIn(expected, normalized_static_section)

        overview = (REPOSITORY / "docs/methodology/overview.md").read_text(
            encoding="utf-8"
        )
        unchanged_paragraph = next(
            paragraph
            for paragraph in re.split(r"\n\s*\n", overview)
            if paragraph.startswith("The HTTP API remains")
        )
        self.assertNotIn("aptus.bundle.v3", unchanged_paragraph)
        normalized_overview = " ".join(overview.split())
        self.assertRegex(
            normalized_overview,
            r"bundle.{0,100}`aptus\.bundle\.v2`.{0,100}`aptus\.bundle\.v3`",
        )
        self.assertNotRegex(
            normalized_overview,
            r"`aptus\.bundle\.v3`.{0,100}\b(?:remain|unchanged|preserv)",
        )

    def test_phase4_public_documentation_is_synchronized(self) -> None:
        changelog = (REPOSITORY / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertNotIn("V3 fact and training-plan contracts", changelog)
        for contract in (
            "aptus.facts.v3",
            "aptus.training-plan.v5",
            "aptus.bundle.v3",
            "aptus.model-policy-snapshot.v1",
        ):
            self.assertIn(contract, changelog)
        for behavior in (
            "Package-free bundle programs validate frozen-snapshot integrity",
            "HTTP 409",
            "`replan_required`",
            "controlled invalid input",
        ):
            self.assertIn(behavior, changelog)

        first_plan = (REPOSITORY / "docs/getting-started/first-plan.md").read_text(
            encoding="utf-8"
        )
        for artifact in (
            "policy_snapshot.py",
            "policy/model-policy-snapshot.v1.json",
            "aptus.bundle.v3",
        ):
            self.assertIn(artifact, first_plan)

        workflows = (REPOSITORY / "docs/product/user-workflows.md").read_text(
            encoding="utf-8"
        )
        normalized_workflows = " ".join(workflows.split())
        self.assertIn("HTTP 409 `replan_required`", normalized_workflows)
        self.assertIn("create no new revision", normalized_workflows)
        self.assertIn("leave the saved bytes unchanged", normalized_workflows)

        snapshot = (REPOSITORY / "docs/reference/model-policy-snapshot.md").read_text(
            encoding="utf-8"
        )
        normalized_snapshot = " ".join(snapshot.split())
        for kind in (
            "exact_identity",
            "quantization_layout",
            "sparse_topology",
            "no_shared_expert",
            "field_equals",
        ):
            self.assertIn(f"`{kind}`", normalized_snapshot)
        for invariant in (
            "exactly one trailing line feed",
            "fact_errors`, sorted before hashing",
            "An omitted `fact_errors` field becomes an empty list",
            "Any non-empty `fact_errors` list is handled before ordinary policy matching",
            "compatibility decision `schema_version`",
            "`compat_` plus the first 20 lowercase hex characters",
            "The four named bindings are `snapshot`, `plan`, `manifest`, and `host`",
            "must not claim host currency from frozen-snapshot integrity",
        ):
            self.assertIn(invariant, normalized_snapshot)

        generated_code = (REPOSITORY / "docs/contributing/generated-code.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("`policy_snapshot.py`", generated_code)
        self.assertIn("`policy/model-policy-snapshot.v1.json`", generated_code)

        release_gates = (REPOSITORY / "docs/operations/release-gates.md").read_text(
            encoding="utf-8"
        )
        for code in (
            "POLICY_SNAPSHOT_MISSING",
            "POLICY_SNAPSHOT_JSON_ERROR",
            "POLICY_SNAPSHOT_CONTRACT",
            "POLICY_SNAPSHOT_NONCANONICAL",
            "POLICY_SNAPSHOT_DIGEST",
            "POLICY_SNAPSHOT_PATH",
        ):
            self.assertIn(code, release_gates)
        self.assertIn(
            "2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md",
            release_gates,
        )
        self.assertIn(
            "No real CUDA pilot or full training evidence has completed",
            release_gates,
        )

        capability_matrix = (
            REPOSITORY / "docs/reference/capability-matrix.md"
        ).read_text(encoding="utf-8")
        normalized_capability_matrix = " ".join(capability_matrix.split())
        for evidence_boundary in (
            "2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md",
            "Two fresh, clean Apple Silicon MLX-LM workflows reached `measured-run-pass`",
            "No real CUDA target-host pilot has been recorded",
        ):
            self.assertIn(evidence_boundary, normalized_capability_matrix)

        current_capabilities = (
            REPOSITORY / "docs/product/current-capabilities.md"
        ).read_text(encoding="utf-8")
        opening_boundary = " ".join(
            current_capabilities.split("## Available now", 1)[0].split()
        )
        self.assertIn(
            "2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md",
            opening_boundary,
        )
        self.assertIn(
            "It closes the current-source Phase 6 runtime gate only for the exact recorded Qwen2.5",
            opening_boundary,
        )
        self.assertIn("No CUDA target-runtime pilot has completed", opening_boundary)

        install = (REPOSITORY / "docs/getting-started/install.md").read_text(
            encoding="utf-8"
        )
        normalized_install = " ".join(install.split())
        self.assertIn(
            "Package-free validation covers the plan, manifest, and snapshot boundaries",
            normalized_install,
        )
        self.assertIn(
            "trainer configuration remains compiler-managed input",
            normalized_install,
        )

        documentation_debt = (
            REPOSITORY / "docs/maintenance/documentation-debt.md"
        ).read_text(encoding="utf-8")
        documentation_health = (
            REPOSITORY / "docs/maintenance/documentation-health.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md",
            documentation_debt,
        )
        self.assertIn(
            "Access to approved CUDA target hosts",
            documentation_debt,
        )
        self.assertIn(
            "No qualifying CUDA target-runtime acceptance has been collected",
            " ".join(documentation_health.split()),
        )

        inventory = (
            REPOSITORY / "docs/maintenance/documentation-inventory.md"
        ).read_text(encoding="utf-8")
        repository_documents = set(repository_markdown_documents())
        excluded_documents = {
            REPOSITORY / ".github/PULL_REQUEST_TEMPLATE.md",
            REPOSITORY / "desktop/macos/README.md",
            *(REPOSITORY / "dev/active").rglob("*.md"),
        }
        deprecated_documents = {
            REPOSITORY / "docs/design/aptus-core-vertical-slice.md",
            REPOSITORY / "docs/validation/aptus-core-smoke.md",
        }
        archived_documents = {
            REPOSITORY / "Reference/FineTuneX.README.md",
            REPOSITORY / "Reference/Fine-Tuning_Methods.md",
            REPOSITORY / "Reference/hparam_methods_reference.md",
            REPOSITORY
            / "docs/operations/evidence/2026-07-29-documentation-drift-audit/README.md",
            REPOSITORY
            / "docs/operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/diagnostics/attempt-01-unreceipted-parent-promotion/README.md",
            *(REPOSITORY / "docs/audits/aptus-legacy").glob("*.md"),
        }
        governed_documents = repository_documents - excluded_documents
        active_documents = (
            governed_documents - deprecated_documents - archived_documents
        )
        self.assertEqual(len(repository_documents), 117)
        self.assertEqual(len(excluded_documents), 13)
        self.assertEqual(len(governed_documents), 104)
        self.assertEqual(len(active_documents), 87)
        self.assertEqual(len(deprecated_documents), 2)
        self.assertEqual(len(archived_documents), 15)
        self.assertEqual(
            governed_documents,
            active_documents | deprecated_documents | archived_documents,
        )
        self.assertEqual(len(maintained_documentation()), 95)
        self.assertIn("104 governed tracked Markdown documents", inventory)
        self.assertIn("95 Markdown files", inventory)
        self.assertIn("117 tracked Markdown files", " ".join(inventory.split()))
        self.assertIn("| Active | 87 |", inventory)
        self.assertIn("| Deprecated | 2 |", inventory)
        self.assertIn("| Archived | 15 |", inventory)

    def test_phase5_workbench_policy_authority_is_documented(self) -> None:
        def normalized(relative_path: str) -> str:
            text = (REPOSITORY / relative_path).read_text(encoding="utf-8")
            return " ".join(text.split())

        required_current_claims = {
            "ROADMAP.md": (
                "Phase 5 is complete",
                "typed HTTP 422 `no_feasible_plan` response",
                "three separate records",
                "Validation evidence and launch admission remain separate",
                "typed `authorization_status` values `current`, `deferred`, and `blocked`",
                "recommendation must structurally equal the complete listed candidate record",
                "browser never derives a status from diagnostic prose",
            ),
            "docs/product/current-capabilities.md": (
                "Phase 5's server-authoritative workbench policy boundary",
                "explicit nullable bindings",
                "Three separate model-policy UI records",
                "Separate validation-evidence and launch-admission states",
                "closed failure also requires a `model` subject",
                "decoded recommendation must structurally equal its complete listed candidate record",
                "same exact tuple gates stage completion and validation or run actions",
                "optional typed `authorization_status` is exactly `current`, `deferred`, or `blocked`",
                "separate MoE topology rail",
            ),
            "docs/product/ui-ux.md": (
                "Facts and Compare render three independent records",
                "**Model-policy match**",
                "**Selected candidate path**",
                "**Evidence readiness**",
                "`bindings.plan_id`, `bindings.candidate_id`, and "
                "`bindings.model_revision` match",
                "same exact tuple gates stage completion and validation or run actions",
                "exact `authorization_status` vocabulary of `current`, `deferred`, and `blocked`",
                "never infers a status from diagnostic prose",
                "Phase 5 is complete",
            ),
            "docs/product/user-workflows.md": (
                "typed HTTP 422 response together with the server decision",
                "required model subject must match the submitted model ID and "
                "immutable revision",
                "same plan ID, candidate ID, and model revision",
                "successful recommendation must structurally equal its complete listed candidate record",
                "optional `authorization_status` is exactly `current`, `deferred`, or `blocked`",
                "does not infer status from that prose diagnostic",
            ),
            "docs/architecture/system.md": (
                "Phase 5 is complete at the client boundary",
                "closed HTTP 422 `no_feasible_plan` response",
                "three records separate",
                "Required validation evidence can be incomplete or complete "
                "independently of launch admission",
                "workflow-completion and validation or run action gates reuse that exact binding predicate",
                "optional `authorization_status` vocabulary is exactly `current`, `deferred`, or `blocked`",
                "recommendation must structurally equal its complete listed candidate record",
                "MoE topology rail separately explains routing",
            ),
            "docs/architecture/data-and-identity-flow.md": (
                "HTTP planning boundary preserves this chain on both success and failure",
                "three independent records",
                "model subject to match the submitted model ID and immutable revision",
                "Evidence progress and launch admission are different axes",
                "Stage-completion and validation or run action gates use the same exact three-field predicate",
                "browser branches on the typed status rather than diagnostic prose",
                "recommendation must structurally equal the complete listed candidate record",
                "all checkpoint weights stay resident",
            ),
            "docs/contributing/workbench.md": (
                "exact object keys and supported schema versions",
                "inspection receipt's v2 decision is the one browser policy source",
                "required `model` subject",
                "model-policy match, selected candidate path, and evidence readiness remain three separate records",
                "decoded recommendation structurally equals its complete listed candidate record",
                "optional typed `authorization_status` values `current`, `deferred`, and `blocked`",
                "generic training-request failure surfaces its error without mutating the prior report",
            ),
            "docs/contributing/changing-contracts.md": (
                "Phase 5 completed removal of browser-side policy reconstruction",
                "typed HTTP 422 `no_feasible_plan` response",
                "Require both responses to carry a model subject matching the submitted model ID and immutable revision",
                "require full structural equality across the complete candidate records",
                "Reuse that exact predicate for model-policy evidence, workflow-stage completion, and validation or run action enablement",
                "optional typed tuple: `authorization_status` is exactly `current`, `deferred`, or `blocked`",
                "surface the request error while preserving the prior report",
            ),
            "docs/maintenance/documentation-debt.md": (
                "### DOC-023: Remove browser-side model-policy reconstruction",
                "**Status:** Resolved",
                "receipt decision is the one inspection-time browser policy source",
                "Required validation is incomplete or complete independently from the optional typed `authorization_status` values",
                "required model subject must match the submitted ID and immutable revision",
                "recommendation must structurally equal its complete listed candidate record",
                "tuple with no non-null member means not checked",
            ),
            "docs/maintenance/documentation-health.md": (
                "Phase 5 maintained-guidance closeout",
                "strict v2 decision, path, receipt, and candidate/report ingress",
                "Evidence completeness stays separate from the optional typed `authorization_status` values",
                "Recommendations structurally equal their complete listed candidate records",
                "tuple with no non-null member means not checked",
                "does not infer status from prose or mutate the report",
                "unused flattened compatibility normalizer was removed",
            ),
            "docs/maintenance/documentation-inventory.md": (
                "after the Phase 6 exact-source evidence refresh",
                "`web/src/lib/modelPolicy.ts`",
            ),
            "docs/reference/api.md": (
                "closed typed `422 no_feasible_plan` object",
                "required `model` object carries the submitted `model_id` and immutable `revision`",
                "Every returned candidate requires this presentation tuple",
                "All candidates in `no_feasible_plan` must be rejected",
                "structurally equal that complete decoded candidate record",
                "`authorization_status` vocabulary is exactly `current`, `deferred`, or `blocked`",
                "generic failed training request surfaces its API error and leaves that last report unchanged",
                "`recommended: null`",
            ),
            "README.md": (
                "artifact match, selected candidate path, and evidence readiness",
                "required model subject with the submitted model ID and immutable revision",
                "Evidence readiness does not imply launch permission",
                "successful recommendation must structurally equal its listed candidate",
                "browser never infers a status from diagnostic prose",
            ),
            "CHANGELOG.md": (
                "server-owned v2 decision as separate artifact-match, selected-path, and evidence-readiness records",
                "Validation completeness remains separate from the optional typed `authorization_status` vocabulary",
                "full structural equality between the recommendation and its listed candidate",
                "browser neither derives status from diagnostic prose nor rewrites a report",
                "unused legacy browser compatibility projection was removed",
            ),
        }
        for relative_path, claims in required_current_claims.items():
            text = normalized(relative_path)
            for claim in claims:
                self.assertIn(claim, text, (relative_path, claim))

        stale_current_claims = {
            "ROADMAP.md": "Phase 5 remains limited",
            "docs/product/ui-ux.md": "Phase 5 owns removal",
            "docs/architecture/system.md": "Phase 5 remains the browser",
            "docs/contributing/changing-contracts.md": "Phase 5 owns removal",
        }
        for relative_path, claim in stale_current_claims.items():
            self.assertNotIn(claim, normalized(relative_path), relative_path)

        model_policy = normalized("web/src/lib/modelPolicy.ts")
        for decoder_contract in (
            "const DECISION_KEYS",
            "const PATH_KEYS",
            "const BINDING_KEYS",
            "const RECEIPT_KEYS",
            "export function decodePlanCandidate",
            "export function decodeValidationReport",
            "an execution tuple that exactly matches a policy path cannot omit its binding",
            "no-feasible-plan rows must be rejected",
            "const AUTHORIZATION_STATUSES",
            "authorization fields require a typed authorization status",
            "current authorization requires qualifying evidence, a true current flag, and no error",
            "deferred or blocked authorization requires a false current flag and a reason",
            "export function validationReportMatchesBinding",
            "report.bindings?.plan_id === identity.planId",
            "report.bindings.candidate_id === identity.candidateId",
            "report.bindings.model_revision === identity.modelRevision",
            "expected at least one provider-declared observation",
            "path-matched provider decisions require satisfied provider-declared provenance",
            '"validation-complete"',
            '"admission-deferred"',
            '"authorized"',
            '"authorization-blocked"',
            'authorizationStatus === "deferred"',
            'authorizationStatus === "blocked"',
        ):
            self.assertIn(decoder_contract, model_policy)
        self.assertNotIn("staleAuthorization", model_policy)
        self.assertNotIn("qwen3_moe", model_policy)

        api_client = normalized("web/src/api.ts")
        for ingress_contract in (
            "interface PlanResponseContext",
            "Plan response policy source differs from the submitted request",
            'decodePlanModelSubject(payload.model, "Plan response", context)',
            "model subject differs from the submitted request",
            "Plan response receipt differs from the submitted request",
            "No-feasible-plan policy source differs from the submitted request",
            "No-feasible-plan receipt differs from the submitted request",
            'decodePlanModelSubject( payload.model, "No-feasible-plan response", context',
            "requireRejected: true",
            "function canonicalJsonValue",
            "function structurallyEqualJson",
            "Plan response recommendation differs from its listed candidate",
        ):
            self.assertIn(ingress_contract, api_client)
        self.assertNotIn("normalizeModelCompatibility", api_client)

        app = normalized("web/src/App.tsx")
        for binding_contract in (
            "const boundActiveReport = validationReportMatchesBinding",
            "candidateId: plan.recommended.candidate_id",
            "reportBinding={validationBinding}",
        ):
            self.assertIn(binding_contract, app)
        create_job_handler = app.split("const handleCreateJob = async", 1)[1].split(
            "const handleCancelJob = async", 1
        )[0]
        self.assertIn("setError(errorMessage(caught))", create_job_handler)
        self.assertNotIn("setReport(", create_job_handler)
        self.assertNotIn("authorization_error", create_job_handler)

        validate_stage = normalized("web/src/stages/ValidateStage.tsx")
        run_stage = normalized("web/src/stages/RunStage.tsx")
        self.assertIn(
            "validationReportMatchesBinding(activeReport, reportBinding)",
            validate_stage,
        )
        self.assertIn(
            "validationReportMatchesBinding(activeReport, reportBinding)", run_stage
        )

        model_inspection = normalized("web/src/lib/modelInspection.ts")
        self.assertNotIn("moeCompatibilityFromPlan", model_inspection)
        self.assertFalse((REPOSITORY / "web/src/lib/modelCompatibility.ts").exists())
        self.assertFalse(
            (REPOSITORY / "web/src/lib/modelCompatibility.test.ts").exists()
        )

        panel = normalized("web/src/components/ModelPolicyPanel.tsx")
        for presentation_copy in (
            "Model-policy match",
            "Selected candidate path",
            "Evidence readiness",
            "Evidence complete",
            "Admission deferred",
            "Admission blocked",
            "Not checked",
        ):
            self.assertIn(presentation_copy, panel)
        self.assertNotIn("stale authorization", panel.lower())

        topology = normalized("web/src/components/ExpertTopologyRail.tsx")
        self.assertIn("All checkpoint weights must remain resident", topology)
        self.assertNotIn("compatibility", topology.lower())

        api_contracts = normalized("src/aptus/api_contracts.py")
        for response_contract in (
            "class NoFeasiblePlanResponse(ClosedResponseModel)",
            "model: PlanModelSubjectResponse",
            "model_policy_decision: InspectedModelPolicyDecisionResponse",
            "inspection_receipt: ModelInspectionReceiptResponse | None",
            "Every candidate must link to the policy decision",
            "status: CandidateStatus",
            "feasible: Annotated[bool",
            "runtime_contract: InspectedRuntimeContractResponse",
            "No-feasible-plan candidates must be infeasible or unsupported",
            "The recommended candidate must equal its listed candidate record",
            'authorization_status: Literal["current", "deferred", "blocked"] | None',
            "Validation authorization fields require authorization_status",
            "def require_authorization_status_coherence",
            "not isinstance(self.authorization_error, str)",
            "Path-matched provider decisions require provider-declared provenance",
        ):
            self.assertIn(response_contract, api_contracts)

        openapi = json.loads(
            (REPOSITORY / "docs/reference/openapi.v1.json").read_text(encoding="utf-8")
        )
        no_feasible = openapi["components"]["schemas"]["NoFeasiblePlanResponse"]
        self.assertIn("model", no_feasible["required"])
        self.assertEqual(
            no_feasible["properties"]["model"]["$ref"],
            "#/components/schemas/PlanModelSubjectResponse",
        )
        authorization_status = openapi["components"]["schemas"]["ValidationResponse"][
            "properties"
        ]["authorization_status"]
        self.assertEqual(
            authorization_status["anyOf"][0]["enum"],
            ["current", "deferred", "blocked"],
        )

    def test_phase6_second_policy_and_evidence_boundary_are_documented(self) -> None:
        def normalized(relative_path: str) -> str:
            text = (REPOSITORY / relative_path).read_text(encoding="utf-8")
            return " ".join(text.split())

        acceptance_leaf = "2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md"
        acceptance_documents = (
            "README.md",
            "ROADMAP.md",
            "SECURITY.md",
            "docs/architecture/artifact-compiler.md",
            "docs/architecture/system.md",
            "docs/getting-started/choose-your-path.md",
            "docs/index.md",
            "docs/reference/capability-matrix.md",
            "docs/reference/plan-schema.md",
            "docs/reference/model-policy-snapshot.md",
            "docs/reference/api.md",
            "docs/reference/evidence-records.md",
            "docs/product/current-capabilities.md",
            "docs/product/claim-language.md",
            "docs/product/ui-ux.md",
            "docs/operations/apple-silicon-pilot.md",
            "docs/operations/index.md",
            "docs/operations/release-gates.md",
            "docs/contributing/changing-contracts.md",
            "docs/contributing/generated-code.md",
            "docs/maintenance/documentation-debt.md",
            "docs/maintenance/documentation-health.md",
        )
        for relative_path in acceptance_documents:
            text = normalized(relative_path)
            self.assertIn(acceptance_leaf, text, relative_path)
            self.assertNotIn("runtime-evidence-open", text, relative_path)
            self.assertNotIn("Phase 6 remains pending", text, relative_path)

        required_claims = {
            "README.md": (
                "`model.qwen2-24l.mlx-qlora`",
                "reviewed dense configuration footprint rather than an artifact allowlist",
                "The two fresh 2026-08-05 MLX-LM runs close the current-source Phase 6 runtime gate",
                "only manifested operator `README.md` and `runbook.md` changed",
            ),
            "ROADMAP.md": (
                "Phase 6 is implemented at the registry, planner, compiler, portable-contract, and test boundaries",
                "`mlx-lm.qlora.single.dense-causal-lm.v1`",
            ),
            "docs/reference/capability-matrix.md": (
                "This is a reviewed configuration footprint, not an artifact allowlist",
                "Uniform four-bit, group size 64, with no module overrides",
                "Two v5/v3 `measured-run-pass` repetitions for the exact 2026-08-05 accepted artifact",
            ),
            "docs/reference/plan-schema.md": (
                "runtime configuration footprint, not an artifact allowlist",
                "two fresh, clean `measured-run-pass` repetitions",
                "does not qualify CUDA or establish safety, model quality, performance",
            ),
            "docs/reference/model-policy-snapshot.md": (
                "`model.qwen2-24l.mlx-qlora`",
                "targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`",
                "Another matching artifact still has to pass its own model-data, measured-preflight, and pilot gates",
            ),
            "docs/reference/api.md": (
                "the Qwen2 policy remains a configuration footprint rather than an artifact allowlist",
                "another matching artifact must pass its own gates",
                "does not qualify CUDA or establish safety, model quality, performance",
            ),
            "docs/reference/evidence-records.md": (
                "It does not broaden or relabel either canonical evidence record",
                "Carrying the historical runtime ID in a current plan preserves its scope",
                "The separate exact-source refresh applies only when its exact plan, bundle, artifact, source, host, runtime, dataset, policy snapshot, and fingerprint bindings match",
            ),
            "docs/reference/cli.md": (
                "`--quantization-group-size INTEGER`",
                "The reviewed dense Qwen2 footprint",
            ),
            "docs/product/claim-language.md": (
                "A configuration-footprint policy is not an artifact allowlist",
                "They close the Phase 6 runtime gate for that exact fixture",
            ),
            "docs/operations/release-gates.md": (
                "`policy.qwen2-24l.mlx-qlora.v1`",
                "`runtime.qwen2-0.5b.mlx-qlora.2026-07-27`",
                "closes the current-source Phase 6 MLX-LM runtime gate",
                "only manifested operator `README.md` and `runbook.md` changed",
            ),
            "docs/contributing/changing-contracts.md": (
                "closes its current-source v5/v3 runtime gate with two fresh, clean",
                "other matching artifacts still require their own gates",
                "does not qualify CUDA or establish safety, model quality, performance",
            ),
            "docs/contributing/generated-code.md": (
                "Generated-code changes that affect any of those bindings require renewed evidence",
                "another matching artifact remains gated",
                "does not qualify CUDA or establish safety, model quality, performance",
            ),
            "docs/maintenance/documentation-debt.md": (
                "### DOC-024: Close Phase 6 runtime evidence for the second model policy",
                "**Status:** Resolved",
                "`14ed44b52a76bb84d8d9db4f2303951aa641339b`",
                "The policy remains a reviewed configuration footprint, not an artifact allowlist",
                "### DOC-025: Refresh Phase 6 evidence at the exact current source",
                "`719255153e3fc7e38e83b5ff826d587e5e58bf80`",
                "`ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`",
            ),
            "docs/maintenance/documentation-health.md": (
                "That exact-source refresh closes the current-source Phase 6 MLX-LM runtime gate for its exact scope",
                "A different matching artifact remains conditional",
                "No qualifying CUDA target-runtime acceptance has been collected",
            ),
        }
        for relative_path, claims in required_claims.items():
            text = normalized(relative_path)
            for claim in claims:
                self.assertIn(claim, text, (relative_path, claim))

        refresh_readme = normalized(
            "docs/operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md"
        )
        for binding in (
            "719255153e3fc7e38e83b5ff826d587e5e58bf80",
            "be99f5664ccb580f2600471f1ae3241a294b1a7e",
            "ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919",
            "measured-run-pass",
        ):
            self.assertIn(binding, refresh_readme)

        baseline_readme = normalized(
            "docs/operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md"
        )
        for claim in (
            "Passed — two clean `measured-run-pass` repetitions",
            "current training-plan v5 and bundle v3 MLX-LM QLoRA ladder twice",
            "Exact pinned artifact, source, host, runtime, dataset, and policy snapshot",
            "Model quality, general Qwen2 compatibility, CUDA acceptance, or production throughput",
        ):
            self.assertIn(claim, baseline_readme)

        baseline_summary = json.loads(
            (
                REPOSITORY
                / "docs/operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/acceptance-summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(baseline_summary["state"], "measured-run-pass")
        self.assertEqual(
            baseline_summary["phase_6_status"], "runtime-evidence-complete"
        )
        self.assertEqual(baseline_summary["completed_clean_repetitions"], 2)
        self.assertEqual(
            baseline_summary["source"]["acceptance_fix_commit"],
            "14ed44b52a76bb84d8d9db4f2303951aa641339b",
        )
        self.assertEqual(
            baseline_summary["model"]["revision"],
            "53a32aee5e9447773fd2b85988395066aef3700a",
        )
        self.assertEqual(baseline_summary["host"]["chip"], "Apple M5 Pro")
        self.assertEqual(baseline_summary["runtime"]["mlx_lm"], "0.31.3")
        self.assertEqual(
            baseline_summary["compiled_input"]["plan_schema_version"],
            "aptus.training-plan.v5",
        )
        self.assertEqual(
            baseline_summary["compiled_input"]["bundle_schema_version"],
            "aptus.bundle.v3",
        )

    def test_exact_source_refresh_packet_is_bound_and_sanitized(self) -> None:
        packet = (
            REPOSITORY
            / "docs/operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh"
        )
        baseline = (
            REPOSITORY / "docs/operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance"
        )
        expected_files = {
            "README.md",
            "SHA256SUMS",
            "acceptance-procedure.json",
            "acceptance-summary.json",
            "bundle-comparison.json",
            "bundle-manifest.json",
            "raw-artifact-digests.json",
            "runs/run-1/run-summary.json",
            "runs/run-2/run-summary.json",
        }
        actual_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)

        checksum_pattern = re.compile(r"^([a-f0-9]{64})  (\./[^\n]+)$")
        checksum_lines = (
            (packet / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        )
        parsed_checksums: list[tuple[str, str]] = []
        for line in checksum_lines:
            match = checksum_pattern.fullmatch(line)
            self.assertIsNotNone(match, line)
            assert match is not None
            parsed_checksums.append((match.group(1), match.group(2)))
        checksum_paths = [relative for _digest, relative in parsed_checksums]
        self.assertEqual(checksum_paths, sorted(checksum_paths))
        self.assertEqual(len(checksum_paths), len(set(checksum_paths)))
        self.assertEqual(
            set(checksum_paths),
            {f"./{relative}" for relative in expected_files - {"SHA256SUMS"}},
        )
        for expected_digest, relative in parsed_checksums:
            target = packet / relative.removeprefix("./")
            self.assertEqual(
                hashlib.sha256(target.read_bytes()).hexdigest(),
                expected_digest,
                relative,
            )

        summary = json.loads(
            (packet / "acceptance-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["state"], "measured-run-pass")
        self.assertEqual(summary["completed_clean_repetitions"], 2)
        self.assertEqual(
            summary["source"]["acceptance_commit"],
            "719255153e3fc7e38e83b5ff826d587e5e58bf80",
        )
        self.assertEqual(
            summary["source"]["acceptance_tree"],
            "be99f5664ccb580f2600471f1ae3241a294b1a7e",
        )
        self.assertEqual(
            summary["compiled_input"]["bundle_fingerprint"],
            "ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919",
        )
        self.assertEqual(
            summary["compiled_input"]["bundle_zip_sha256"],
            "fcad829b4c845c6b5d1e548b293ec1107ccd7a78ea08b63bc7a1b8ca487be9b1",
        )
        self.assertFalse(
            summary["bundle_comparison"]["old_evidence_transferred_to_new_fingerprint"]
        )
        self.assertTrue(
            summary["bundle_comparison"]["new_fingerprint_freshly_qualified"]
        )
        self.assertEqual(len(summary["runs"]), 2)

        for ordinal in (1, 2):
            run = json.loads(
                (packet / f"runs/run-{ordinal}/run-summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(run["ordinal"], ordinal)
            self.assertEqual(run["state"], "measured-run-pass")
            self.assertEqual(
                [job["action"] for job in run["jobs"]],
                ["dependency", "model-data", "preflight", "pilot", "train"],
            )
            self.assertTrue(
                all(
                    job["state"] == "completed" and job["return_code"] == 0
                    for job in run["jobs"]
                )
            )
            self.assertEqual(
                run["jobs"][-1]["artifact_integrity_status"],
                "verified-at-completion",
            )
            self.assertEqual(
                run["jobs"][-1]["completion_attestation_state"],
                "measured-run-pass",
            )
            self.assertTrue(run["full_train"]["source_report_hash_reconstructed"])
            self.assertTrue(run["parent_promotion"]["evidence_sha256_recomputed"])
            self.assertTrue(run["parent_promotion"]["pending_fields_absent"])
            self.assertTrue(run["reload"]["fresh_process_observed"])
            self.assertTrue(run["all_invariants_passed"])

        comparison = json.loads(
            (packet / "bundle-comparison.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [entry["path"] for entry in comparison["changed_paths"]],
            ["README.md", "runbook.md"],
        )
        self.assertEqual(comparison["changed_path_count"], 2)
        self.assertEqual(comparison["unchanged_path_count"], 27)
        self.assertTrue(comparison["runtime_programs_byte_identical"])
        self.assertTrue(comparison["runtime_dependencies_byte_identical"])
        self.assertFalse(comparison["baseline_runtime_evidence_transferred"])
        expected_runtime_hashes = {
            "plan_contract.py": "b69d072a7287da9d6536ed3d6fd85734d97e68ad9ce8b7cb9364b96bdfb92efe",
            "policy_snapshot.py": "aa865c8ec6c3f89c863b22de9bbb9be96f32e1cf59dcf24b9ced2d9da3a94480",
            "preflight.py": "86cb122f355fae3e7aba0afb77e47c1154eb7fd796bf03f6f4e31e56e77d8561",
            "reload.py": "5b2ee41adec0ea443aff7a96918d3116d1670ca776c9dbe26d53b3693046b1b7",
            "run.py": "b18daa1f4eff82dbe25bd338e2c9ca1c9d03566fc215af6b5d7a19d90e7d3029",
            "runtime_lease.py": "a021fcb8b6da10fa5b443a14edea59280fd4041b3df4020fe07a44aee88bb6ad",
            "train.py": "a3e943f707e688821587d9b0216f6404d77b5d526c32ce84cc57f5b826e5e27a",
            "validate.py": "82eaf072abe8575f798b7cea600c111bfb48804f45ef167409a565dc0d72df33",
        }
        self.assertEqual(comparison["runtime_programs"], expected_runtime_hashes)

        baseline_manifest = json.loads(
            (baseline / "bundle-manifest.json").read_text(encoding="utf-8")
        )
        refresh_manifest_path = packet / "bundle-manifest.json"
        refresh_manifest = json.loads(refresh_manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(refresh_manifest_path.read_bytes()).hexdigest(),
            summary["compiled_input"]["bundle_fingerprint"],
        )
        baseline_files = {entry["path"]: entry for entry in baseline_manifest["files"]}
        refresh_files = {entry["path"]: entry for entry in refresh_manifest["files"]}
        self.assertEqual(set(baseline_files), set(refresh_files))
        actual_changed_paths = [
            path
            for path in sorted(refresh_files)
            if baseline_files[path] != refresh_files[path]
        ]
        self.assertEqual(actual_changed_paths, ["README.md", "runbook.md"])

        forbidden_patterns = (
            "/Users/",
            "/private/tmp",
            "/tmp/",
            "owner_pid",
            "process_pid",
            "process_group_id",
            "parent_pid",
            "verifier_pid",
            "log_tail",
        )
        for relative in expected_files - {"SHA256SUMS"}:
            text = (packet / relative).read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                self.assertNotIn(pattern, text, (relative, pattern))

        self.assertEqual(
            hashlib.sha256((baseline / "SHA256SUMS").read_bytes()).hexdigest(),
            summary["baseline"]["packet_sha256s_sha256"],
        )

    def test_private_security_reporting_route_is_concrete(self) -> None:
        address = "aptus-security@proton.me"
        mailto = f"mailto:{address}"
        security = (REPOSITORY / "SECURITY.md").read_text(encoding="utf-8")
        normalized_security = " ".join(security.split())
        issue_config = (REPOSITORY / ".github/ISSUE_TEMPLATE/config.yml").read_text(
            encoding="utf-8"
        )
        debt = (REPOSITORY / "docs/maintenance/documentation-debt.md").read_text(
            encoding="utf-8"
        )
        health = (REPOSITORY / "docs/maintenance/documentation-health.md").read_text(
            encoding="utf-8"
        )
        doc_006 = debt.split("### DOC-006:", 1)[1].split("\n### ", 1)[0]

        self.assertIn(mailto, security)
        self.assertIn(address, issue_config)
        self.assertIn(
            "https://github.com/ogprotege/Aptus/security/policy",
            issue_config,
        )
        self.assertNotIn("security/advisories/new", security)
        self.assertNotIn("security/advisories/new", issue_config)
        self.assertIn("| Version | Security fixes | Status |", security)
        self.assertIn("acknowledgment within three business days", normalized_security)
        self.assertIn("initial assessment within seven", normalized_security)
        self.assertIn("Keep the report private until", normalized_security)
        self.assertIn("**Status:** Resolved", doc_006)
        self.assertIn(address, doc_006)
        self.assertNotIn("**Status:** Open", doc_006)
        self.assertNotIn(
            "still lacks a guaranteed private intake address",
            health,
        )

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

    def test_cli_reference_matches_parser_choices_and_defaults(self) -> None:
        reference = (REPOSITORY / "docs/reference/cli.md").read_text(encoding="utf-8")
        self.assertEqual(
            documented_cli_parser_contract(reference),
            cli_parser_contract(),
            "CLI parser choices or defaults differ from the structured reference",
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
