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
    documents.extend(sorted((REPOSITORY / "docs").rglob("*.md")))
    documents.extend(sorted((REPOSITORY / "examples").glob("*.md")))
    documents.extend(sorted((REPOSITORY / "Reference").glob("*.md")))
    documents.append(REPOSITORY / "desktop/macos/README.md")
    documents.extend(sorted((REPOSITORY / "dev/archive").rglob("*.md")))
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
            "aptus.training-plan.v6",
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
            "aptus.training-plan.v6",
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
            "Every v5, v4, v3, v2, or schema-less plan requires replanning",
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
        self.assertIn("aptus.training-plan.v6", cli)

        error_codes = (REPOSITORY / "docs/reference/error-codes.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("`aptus.training-plan.v6` contract", error_codes)
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
        self.assertIn("aptus.training-plan.v6", system)
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
        self.assertIn("`aptus.training-plan.v6`", health)

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
                "Write a persisted v6 plan JSON without compiling",
                "Every v6 plan persists",
            ),
            "docs/reference/api.md": (
                "required v6 schema",
                "The OpenAPI response requires the v6 schema",
                "Create a new v6 plan from the source facts",
            ),
            "docs/reference/cli.md": (
                "Write a standalone v6 plan",
                "exact v6 domain contract",
            ),
            "docs/reference/error-codes.md": ("create a new v6 plan",),
            "docs/reference/evidence-records.md": (
                "Every candidate in an `aptus.training-plan.v6` plan",
                "The v6 plan ID binds",
            ),
            "docs/reference/plan-schema.md": (
                "The current schema identifier is `aptus.training-plan.v6`",
                "Every v5, v4, v3, v2, or schema-less plan requires replanning",
            ),
            "docs/reference/validation-states.md": (
                "Every v5, v4, v3, v2, or schema-less plan requires replanning",
            ),
            "docs/architecture/artifact-compiler.md": (
                "Valid `aptus.training-plan.v6` payload",
                "The installed Aptus host compiler",
            ),
            "docs/architecture/data-and-identity-flow.md": (
                "`aptus.training-plan.v6` model payload",
                "compare a v6 decision and snapshot digest",
            ),
            "docs/architecture/system.md": (
                "`aptus.training-plan.v6` and `aptus.bundle.v3` cross-bind",
            ),
            "docs/contributing/changing-contracts.md": (
                "The current plan reader accepts only `aptus.training-plan.v6`",
            ),
            "docs/maintenance/documentation-health.md": (
                "The historical Phase 3 implementation added",
                "The current Phase 4 contract uses `aptus.training-plan.v6`",
            ),
            "docs/methodology/overview.md": (
                "contract from `aptus.training-plan.v5` to `aptus.training-plan.v6`",
            ),
            "docs/product/current-capabilities.md": (
                "Persisted `aptus.training-plan.v6` compatibility provenance",
            ),
            "docs/product/ui-ux.md": (
                "creates a new deterministic v6 plan",
                "The `aptus.training-plan.v6` carries one",
            ),
            "docs/methodology/facts-and-provenance.md": (
                "Changing an input requires a new `aptus.training-plan.v6` plan",
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
            self.assertIn("aptus.training-plan.v6", text, relative_path)
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
            "aptus.training-plan.v6",
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
            "The completed CUDA campaign establishes only the exact cells and bounded",
            release_gates,
        )

        capability_matrix = (
            REPOSITORY / "docs/reference/capability-matrix.md"
        ).read_text(encoding="utf-8")
        normalized_capability_matrix = " ".join(capability_matrix.split())
        for evidence_boundary in (
            "2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md",
            "Two fresh, clean Apple Silicon MLX-LM workflows reached `measured-run-pass`",
            "One exact SmolLM2 CUDA LoRA single-device workflow separately reached",
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
            "It supplies current-contract Phase 6 runtime evidence at the exact acceptance source only for the recorded Qwen2.5",
            opening_boundary,
        )
        self.assertIn(
            "2026-08-12-path-beta-cuda-lora-m4/README.md",
            opening_boundary,
        )
        self.assertIn(
            "One earlier exact SmolLM2 CUDA LoRA single-device workflow reached",
            opening_boundary,
        )

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
            "Phase 2B used the exact merged source to publish the independently reviewed",
            " ".join(documentation_debt.split()),
        )
        self.assertIn(
            "2026-08-09-cuda-phase0-recovery-supplement/README.md",
            documentation_debt,
        )
        self.assertIn(
            "The exact CUDA LoRA single-device repeatability anchor is now established",
            " ".join(documentation_health.split()),
        )

        inventory = (
            REPOSITORY / "docs/maintenance/documentation-inventory.md"
        ).read_text(encoding="utf-8")
        normalized_inventory = " ".join(inventory.split())
        repository_documents = set(repository_markdown_documents())
        excluded_documents = {REPOSITORY / ".github/PULL_REQUEST_TEMPLATE.md"}
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
            *(
                path
                for path in (REPOSITORY / "dev/archive").rglob("*.md")
                if path.name != "README.md"
            ),
        }
        governed_documents = repository_documents - excluded_documents
        active_documents = (
            governed_documents - deprecated_documents - archived_documents
        )
        self.assertEqual(len(repository_documents), 143)
        self.assertEqual(len(excluded_documents), 1)
        self.assertEqual(len(governed_documents), 142)
        self.assertEqual(len(active_documents), 113)
        self.assertEqual(len(deprecated_documents), 2)
        self.assertEqual(len(archived_documents), 27)
        self.assertEqual(
            governed_documents,
            active_documents | deprecated_documents | archived_documents,
        )
        self.assertEqual(len(maintained_documentation()), 142)
        self.assertIn("142 are governed", normalized_inventory)
        self.assertIn("142 governed", normalized_inventory)
        self.assertIn("143 tracked Markdown", normalized_inventory)
        self.assertIn("| Active | 113 |", inventory)
        self.assertIn("| Deprecated | 2 |", inventory)
        self.assertIn("| Archived | 27 |", inventory)

    def test_cuda_empirical_campaign_is_canonical_and_bounded(self) -> None:
        campaign_path = REPOSITORY / "docs/operations/cuda-empirical-campaign.md"
        campaign = campaign_path.read_text(encoding="utf-8")
        normalized_campaign = " ".join(campaign.split())
        for required in (
            "Canonical operational plan and completed record for bounded CUDA evidence",
            "Phase 0 — forensic recovery before host changes",
            "Phase 3 — explicit method selection and measurement contracts",
            "exactly five predeclared measured attempts",
            "It cannot execute DDP or LoRA FSDP",
            "Protected raw vault",
            "Attempt-slot ID",
            "Only a native `passed` outcome with `protocol-valid` evidence",
            "Started = Protocol-valid + Capture-invalid",
            "Report tokens per second only from the exact Phase 3 padded",
        ):
            self.assertIn(required, normalized_campaign)

        phase_2_tooling = (
            REPOSITORY / "docs/operations/cuda-campaign-phase2-tooling.md"
        ).read_text(encoding="utf-8")
        normalized_phase_2_tooling = " ".join(phase_2_tooling.split())
        for required in (
            "Phase 2A source tooling and Phase 2B sanitized recovery publication complete and independently reviewed",
            "No Ubuntu command, model download, GPU workload, or new empirical run occurred",
            "It is not Aptus's global ceiling",
            "Phase 2B completion",
            "No Linux connection, Ubuntu-host mutation, model download, GPU workload, or new empirical run occurred during Phase 2B",
            "The earlier stage-review-finalize sequence is not frozen authority",
            "tools/cuda_campaign/admission.py",
            "tools/cuda_campaign/phase4.py",
            "tools/cuda_campaign/outcomes.py",
            "planned-slot context plus all seven activation files",
            "seven frozen output roles",
            "bundle-archive",
            "two fresh live eligibility passes",
            "creates no eligible decision anchor until the final pass succeeds",
            "Failed post-commit verification or parent `fsync` rolls the directory out of the public destination",
        ):
            self.assertIn(required, normalized_phase_2_tooling)
        self.assertNotIn("under red-team remediation", normalized_phase_2_tooling)
        self.assertNotIn("Phase 2A is not complete", normalized_phase_2_tooling)

        phase_6 = campaign.partition("### Phase 6")[2].partition("### Phase 7")[0]
        documented_cuda_single_methods = set(
            re.findall(r"^\| `([^`]+)` \|", phase_6, flags=re.MULTILINE)
        )
        registered_cuda_single_methods = {
            descriptor.method_id
            for descriptor in method_descriptors()
            if descriptor.selectable
            and any(
                binding.training_runtime == TrainingRuntime.TRANSFORMERS_PEFT_CUDA.value
                and binding.compute_backend == Backend.CUDA.value
                and Distribution.SINGLE.value in binding.supported_distributions
                for binding in descriptor.runtime_bindings
            )
        }
        self.assertEqual(
            documented_cuda_single_methods,
            registered_cuda_single_methods,
        )

        linked_documents = (
            REPOSITORY / "ROADMAP.md",
            REPOSITORY / "docs/index.md",
            REPOSITORY / "docs/operations/index.md",
            REPOSITORY / "docs/operations/release-gates.md",
            REPOSITORY / "docs/operations/release-evidence-template.md",
            REPOSITORY / "docs/operations/state-storage-retention.md",
            REPOSITORY / "docs/operations/operator-checklist.md",
            REPOSITORY / "docs/maintenance/documentation-debt.md",
        )
        for document in linked_documents:
            self.assertIn(
                "cuda-empirical-campaign.md",
                document.read_text(encoding="utf-8"),
                document.relative_to(REPOSITORY),
            )

    def test_cuda_campaign_phase1_protocol_is_canonical_and_frozen(self) -> None:
        protocol_path = REPOSITORY / "docs/reference/cuda-campaign-protocol.v1.json"
        protocol_bytes = protocol_path.read_bytes()
        protocol = json.loads(protocol_bytes)
        self.assertEqual(
            protocol_bytes,
            (
                json.dumps(
                    protocol,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        self.assertEqual(protocol["schema_version"], "aptus.cuda-campaign-protocol.v1")
        self.assertEqual(
            set(protocol["record_schema_family"].values()),
            {
                "aptus.experiment-attempt-slot.v1",
                "aptus.experiment-campaign.v1",
                "aptus.experiment-capture-failure.v1",
                "aptus.experiment-claim-boundary.v1",
                "aptus.experiment-comparison-cell.v1",
                "aptus.experiment-comparison-cohort.v1",
                "aptus.experiment-event.v1",
                "aptus.experiment-execution-configuration.v1",
                "aptus.experiment-publication-review.v1",
                "aptus.experiment-raw-manifest.v1",
                "aptus.experiment-raw-seal.v1",
                "aptus.experiment-recovery-supplement.v1",
                "aptus.experiment-run.v1",
                "aptus.experiment-evidence-receipt.v1",
                "aptus.experiment-sanitization-map.v1",
                "aptus.experiment-telemetry-sample.v1",
            },
        )
        identity_prefixes = {
            "campaign": "campaign_",
            "comparison_cohort": "cohort_",
            "comparison_cell": "cell_",
            "attempt_slot": "slot_",
            "execution_configuration": "exec_",
            "experiment_run": "xrun_",
        }
        for identity, prefix in identity_prefixes.items():
            self.assertEqual(protocol["identity_contracts"][identity]["prefix"], prefix)
        self.assertEqual(
            protocol["state_axes"]["slot_status"],
            ["started", "planned-not-started"],
        )
        self.assertEqual(
            protocol["state_axes"]["native_outcome"],
            [
                "passed",
                "refused",
                "failed",
                "cancelled",
                "timed-out",
                "guard-blocked",
                "unknown",
            ],
        )
        self.assertEqual(
            protocol["state_axes"]["evidence_status"],
            ["protocol-valid", "capture-invalid", "not-started"],
        )

        fixture = protocol["fixtures"]["benchmark"]
        fixture_path = REPOSITORY / fixture["path"]
        self.assertEqual(fixture_path.stat().st_size, fixture["byte_size"])
        self.assertEqual(
            hashlib.sha256(fixture_path.read_bytes()).hexdigest(), fixture["sha256"]
        )
        generator = fixture["generator"]
        generator_path = REPOSITORY / generator["path"]
        self.assertEqual(
            hashlib.sha256(generator_path.read_bytes()).hexdigest(),
            generator["sha256"],
        )
        self.assertEqual(
            (fixture["row_count"], fixture["group_count"], fixture["rows_per_group"]),
            (512, 128, 4),
        )
        self.assertEqual(
            fixture["canonical_split"],
            {
                "assignment_sha256": "7e9e747a6e69868d2d542137468cd1baf3d81d7aaac1de29ed14e4dd83b428ed",
                "evaluation_band_rows": [
                    {"rows": 24, "target_content_words": 128},
                    {"rows": 16, "target_content_words": 256},
                    {"rows": 8, "target_content_words": 512},
                    {"rows": 8, "target_content_words": 1024},
                    {"rows": 8, "target_content_words": 2048},
                ],
                "evaluation_groups": 16,
                "evaluation_rows": 64,
                "seed": 424242,
                "train_band_rows": [
                    {"rows": 232, "target_content_words": 128},
                    {"rows": 112, "target_content_words": 256},
                    {"rows": 56, "target_content_words": 512},
                    {"rows": 24, "target_content_words": 1024},
                    {"rows": 24, "target_content_words": 2048},
                ],
                "train_groups": 112,
                "train_rows": 448,
            },
        )
        tokenizer = fixture["tokenizer_verification"]
        self.assertEqual(
            tokenizer["recovered_final_file_inventory"],
            [
                {
                    "byte_size": 3_522_871,
                    "path": "tokenizer.json",
                    "sha256": "bf346d64f6f0fbcefb4c1b6928a98241467dff36c6fbae5fe1785c4ff90667f4",
                },
                {
                    "byte_size": 452,
                    "path": "tokenizer_config.json",
                    "sha256": "9b6f7008bcd69b60572d2e15b28caa540d605df1c08149553296574f66545e53",
                },
            ],
        )
        token_manifest = tokenizer["per_row_manifest"]
        self.assertIsNone(token_manifest["path"])
        self.assertEqual(
            token_manifest["qualification_status"],
            "nonqualifying-phase1-read-only-preview",
        )
        self.assertEqual(token_manifest["byte_size"], 63_234)
        self.assertEqual(
            token_manifest["sha256"],
            "fa8a4c9223e47fa95cb163db871c35159978b92c4ea559b95e8719697c7be9f6",
        )

        models = {
            item["model_id"]: item["revision"]
            for item in protocol["model_catalog"]["models"]
        }
        self.assertEqual(
            models,
            {
                "HuggingFaceTB/SmolLM2-135M-Instruct": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
                "HuggingFaceTB/SmolLM2-360M-Instruct": "a10cc1512eabd3dde888204e902eca88bddb4951",
                "HuggingFaceTB/SmolLM2-1.7B-Instruct": "31b70e2e869a7173562077fd711b654946d38674",
            },
        )
        self.assertEqual(
            [
                item["model_id"]
                for item in protocol["model_catalog"]["breadth_candidates"]
            ],
            [
                "Qwen/Qwen3-0.6B",
                "google/gemma-3-1b-it",
                "mistralai/Mistral-7B-v0.3",
            ],
        )

        matrix = protocol["matrix_contract"]
        anchor = matrix["anchor_execution_configuration"]
        self.assertEqual(
            (
                anchor["compute_precision"],
                anchor["placement"],
                anchor["world_size"],
                anchor["sequence_length"],
                anchor["effective_batch_size"],
                anchor["micro_batch_size"],
                anchor["gradient_accumulation_steps"],
                anchor["optimizer_step_target"],
                anchor["checkpoint_cadence_optimizer_steps"],
            ),
            ("bf16", "single", 1, 256, 8, 4, 2, 128, 64),
        )
        self.assertEqual(
            anchor["adapter"]["target_modules"],
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        self.assertEqual(
            matrix["seed_policy"]["data_order_seed_formula"],
            "1000000 + scheduled_seed",
        )
        phase5 = matrix["phase5_repeatability_anchor"]
        self.assertEqual(phase5["training_seeds"], [101, 211, 307, 401, 503])
        self.assertEqual(phase5["measured_attempt_count"], 5)
        self.assertEqual(phase5["emergency_deadline_seconds"], 1800)
        phase6 = matrix["phase6_same_model_method_matrix"]
        self.assertEqual(
            [block["paired_training_seed"] for block in phase6["exploratory_blocks"]],
            [601, 701, 809],
        )
        self.assertEqual(
            [block["paired_training_seed"] for block in phase6["confirmatory_blocks"]],
            [1009, 2003, 3001, 4001, 5003],
        )
        for block in phase6["exploratory_blocks"] + phase6["confirmatory_blocks"]:
            self.assertEqual(
                block["paired_data_order_seed"],
                1_000_000 + block["paired_training_seed"],
            )
        self.assertEqual(
            matrix["phase7_scale_staircase"]["per_cell_training_seeds"],
            [6101, 6203, 6301],
        )
        phase8 = matrix["phase8_guarded_frontiers"]
        self.assertEqual(
            phase8["sequence_length"]["ladder"], [128, 256, 512, 1024, 2048]
        )
        self.assertEqual(
            phase8["effective_batch_size"]["ladder"], [1, 2, 4, 8, 16, 32, 64]
        )
        self.assertEqual(
            phase8["micro_batch_and_accumulation"]["ladder"],
            [[1, 16], [2, 8], [4, 4], [8, 2], [16, 1]],
        )
        self.assertEqual(
            (phase8["training_seed"], phase8["data_order_seed"]), (8009, 1008009)
        )
        self.assertEqual(phase8["attempt_slot_role"], "phase8-frontier")
        self.assertEqual(
            phase8["managed_action_profile"],
            ["dependency", "model-data", "preflight", "pilot"],
        )
        phase9 = matrix["phase9_endurance"]
        self.assertEqual(phase9["training_seeds"], [9101, 9203, 9301])
        self.assertEqual(phase9["measured_attempt_count"], 3)
        self.assertEqual(
            phase9["pass_conditions"],
            {
                "capture_artifact_or_integrity_defect_allowed": False,
                "complete_required_copy_verification": True,
                "current_successful_off_host_retrieval": True,
                "evidence_status_required": "protocol-valid",
                "native_outcome_required": "passed",
                "required_slot_count": 3,
                "safety_warning_or_stop_allowed": False,
            },
        )
        stability = matrix["common_stability_and_integrity_contract"]
        self.assertEqual(stability["telemetry_coverage_minimum"], 0.99)
        self.assertEqual(stability["telemetry_maximum_gap_seconds"], 2.5)
        self.assertEqual(
            stability[
                "median_absolute_deviation_to_median_training_window_ratio_maximum"
            ],
            0.1,
        )

        safety = protocol["safety_contract"]
        self.assertEqual(
            (
                safety["gpu_thermal"]["warning_temperature_c"],
                safety["gpu_thermal"]["hard_stop_temperature_c"],
                safety["gpu_thermal"]["hard_stop_once_temperature_c"],
            ),
            (78, 84, 89),
        )
        self.assertEqual(safety["vram"]["hard_stop_free_bytes_below"], 2 * 1024**3)
        self.assertEqual(
            safety["host_memory"]["hard_stop_mem_available_bytes_below"],
            8 * 1024**3,
        )
        self.assertEqual(safety["disk"]["hard_stop_free_bytes_below"], 32 * 1024**3)
        self.assertEqual(safety["cooldown"]["required_sample_count"], 120)
        self.assertEqual(safety["cooldown"]["maximum_wait_seconds"], 1800)
        self.assertIn(
            "invalid trainable parameter census", safety["immediate_stop_events"]
        )
        reason_codes = safety["reason_codes"]
        self.assertEqual(len(reason_codes), len(set(reason_codes)))
        self.assertTrue(all(re.fullmatch(r"[A-Z0-9_]+", code) for code in reason_codes))

        telemetry = protocol["telemetry_contract"]
        self.assertEqual(telemetry["sample_interval_seconds"], 1)
        self.assertEqual(telemetry["coverage"]["minimum_qualifying_coverage"], 0.99)
        self.assertEqual(telemetry["gap_policy"]["maximum_qualifying_gap_seconds"], 2.5)
        self.assertEqual(
            telemetry["estimated_gpu_energy"][
                "maximum_adjacent_sample_interval_seconds"
            ],
            2,
        )
        memory_integrity = telemetry["gpu_memory_integrity"]
        self.assertIn(
            "one-half of each used, free, reserved, and total",
            memory_integrity["mismatch_tolerance_rule"],
        )
        self.assertIn(
            "convert used, free, reserved, and total to integer bytes exactly",
            memory_integrity["normalization_rule"],
        )
        phase3 = protocol["phase3_implementation_prerequisites"]
        throughput = protocol["measurement_contract"]["throughput"]
        self.assertTrue(
            throughput[
                "exact_padded_non_padding_and_supervised_token_counters_required_for_token_rate"
            ]
        )
        self.assertTrue(throughput["training_and_evaluation_counters_separate"])
        self.assertTrue(phase3["exact_token_counters"]["defer_allowed"])
        self.assertEqual(
            phase3["exact_token_counters"]["effect_if_deferred"],
            "publish no token-throughput field or claim",
        )
        self.assertEqual(
            phase3["exact_token_counters"]["requirements"],
            [
                "count exact padded input elements presented to the model",
                "count exact non-padding tokens where the attention mask is active",
                "count exact supervised tokens where the label is not -100",
                "separate training and evaluation counters",
                "bind counter semantics and marked monotonic duration to plan, runtime metrics, and sanitizer",
            ],
        )
        self.assertEqual(
            phase3["training_progress_counters"],
            {
                "required_before_phase": 5,
                "requirements": [
                    "separate training and evaluation counters",
                    "count micro-iterations",
                    "count completed non-skipped optimizer steps",
                    "count examples consumed including repeated epoch traversal",
                    "bind counter semantics to plan, runtime metrics, and sanitizer",
                ],
            },
        )
        retention = protocol["retention_contract"]
        self.assertEqual(retention["copy_verification"]["cadence_days"], 90)
        self.assertEqual(retention["full_off_host_retrieval"]["cadence_days"], 180)
        self.assertEqual(
            protocol["sanitizer_contract"]["public_reason"][
                "maximum_unicode_code_points"
            ],
            240,
        )
        recovery = protocol["recovery_supplement_contract"]
        self.assertEqual(
            (
                recovery["expected_digest_manifest"]["logical_digest_count"],
                recovery["expected_digest_manifest"]["unique_digest_count"],
            ),
            (40, 39),
        )
        self.assertEqual(
            protocol["event_ledger_contract"]["final_event"], "seal.started"
        )
        self.assertIn(
            "Phase 3 implementation is merged",
            protocol["phase_order_and_gates"]["host_mutation_gate"],
        )

        human = (REPOSITORY / "docs/reference/cuda-campaign-protocol.md").read_text(
            encoding="utf-8"
        )
        normalized_human = " ".join(human.split())
        for required in (
            "Frozen Phase 1 protocol; implementation pending",
            fixture["sha256"],
            fixture["canonical_split"]["assignment_sha256"],
            "data_order_seed=1000000+scheduled_seed",
            "No frozen fixture row reaches 4,096 tokens",
            "39 unique expected digests",
            "The original August 6 packet remains immutable",
            "Convert each reported used, free, reserved, and total value",
            "Exact padded, non-padding, and supervised token counters may be deferred only",
            "invalid trainable parameter census",
            "applicable uncorrected hardware error",
            "successful current off-host retrieval",
        ):
            self.assertIn(required, normalized_human)
        for private_value in (
            "/Users/",
            "/home/",
            "192.168.",
            "private-hostname.example",
            "aptus-security@proton.me",
        ):
            self.assertNotIn(private_value, human)
            self.assertNotIn(private_value, protocol_bytes.decode("utf-8"))

    def test_post_phase6_documentation_lifecycle_is_governed(self) -> None:
        self.assertEqual(list((REPOSITORY / "dev/active").rglob("*.md")), [])

        archive_index = REPOSITORY / "dev/archive/README.md"
        archived_reviews = sorted(
            path
            for path in (REPOSITORY / "dev/archive").rglob("*.md")
            if path != archive_index
        )
        self.assertEqual(len(archived_reviews), 12)
        for document in archived_reviews:
            metadata = document.read_text(encoding="utf-8")[:1600]
            self.assertIn(
                "**Documentation status:** Archived and superseded review evidence",
                metadata,
                document,
            )
            self.assertIn("**Historical warning:**", metadata, document)

        legacy_directory = REPOSITORY / "docs/audits/aptus-legacy"
        subordinate_legacy_reports = sorted(
            path for path in legacy_directory.glob("*.md") if path.name != "README.md"
        )
        self.assertEqual(len(subordinate_legacy_reports), 9)
        for document in subordinate_legacy_reports:
            metadata = document.read_text(encoding="utf-8")[:1600]
            self.assertIn("**Documentation status:** Archived evidence", metadata)
            self.assertIn("**Historical warning:**", metadata)

        desktop_readme = (REPOSITORY / "desktop/macos/README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "**Documentation status:** Active implementation and build guide",
            desktop_readme,
        )
        desktop_architecture = (
            REPOSITORY / "docs/architecture/macos-desktop.md"
        ).read_text(encoding="utf-8")
        self.assertIn("../../desktop/macos/README.md", desktop_architecture)

        archive_navigation = (REPOSITORY / "docs/archive/index.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("../../dev/archive/README.md", archive_navigation)
        self.assertIn(
            "2026-07-29-documentation-drift-audit/README.md", archive_navigation
        )
        self.assertIn(
            "attempt-01-unreceipted-parent-promotion/README.md",
            archive_navigation,
        )

        inventory = (
            REPOSITORY / "docs/maintenance/documentation-inventory.md"
        ).read_text(encoding="utf-8")
        self.assertIn("MLX bundles additionally generate `reload.py`", inventory)

    def test_apple_reserve_floor_is_explicit_in_operator_references(self) -> None:
        defaults = (REPOSITORY / "docs/reference/configuration-defaults.md").read_text(
            encoding="utf-8"
        )
        api = (REPOSITORY / "docs/reference/api.md").read_text(encoding="utf-8")
        self.assertIn("syntactic default, not the effective Apple", defaults)
        self.assertIn("max(--reserve-gib, 8.0)", defaults)
        self.assertIn(
            "effective 8 GiB floor for unified-memory requests",
            api,
        )

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
                "after PR #41 and the canonical CUDA campaign integration",
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
                "The two fresh 2026-08-05 MLX-LM runs supply current-contract Phase 6 runtime evidence at their exact acceptance source",
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
                "reviewed configuration footprint, not an artifact allowlist",
                "two fresh, clean `measured-run-pass` repetitions",
                "does not qualify CUDA or establish safety, model quality, performance",
            ),
            "docs/reference/model-policy-snapshot.md": (
                "`model.qwen2-24l.mlx-qlora`",
                "targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`",
                "Another matching artifact still has to pass its own model-data, measured-preflight, and pilot gates",
            ),
            "docs/reference/api.md": (
                "The Qwen2 policy remains a configuration footprint rather than an artifact allowlist",
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
                "current-contract evidence at exact source records two fresh v5/v3",
                "only manifested operator `README.md` and `runbook.md` changed",
            ),
            "docs/contributing/changing-contracts.md": (
                "supplies current-contract v5/v3 runtime evidence at its exact acceptance source with two fresh, clean",
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
                "### DOC-025: Refresh Phase 6 evidence at the exact acceptance source",
                "`719255153e3fc7e38e83b5ff826d587e5e58bf80`",
                "`ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`",
            ),
            "docs/maintenance/documentation-health.md": (
                "That refresh supplies current-contract Phase 6 MLX-LM runtime evidence at its exact acceptance source",
                "A different matching artifact remains conditional",
                "The exact CUDA LoRA single-device repeatability anchor is now established",
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

    def test_phase2b_recovery_supplement_is_complete_reviewed_and_sanitized(
        self,
    ) -> None:
        packet = (
            REPOSITORY
            / "docs/operations/evidence/2026-08-09-cuda-phase0-recovery-supplement"
        )
        published = packet / "published"
        expected_published_files = {
            "PUBLICATION-SHA256SUMS",
            "SHA256SUMS",
            "claim-boundary.json",
            "finalization.json",
            "independent-review.json",
            "publication-candidate.json",
            "publication-decision-binding.json",
            "publication-decision.json",
            "recovery-supplement.json",
            "review-bindings.json",
            "sanitization-map.json",
        }
        actual_packet_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(
            actual_packet_files,
            {"README.md"}
            | {f"published/{relative}" for relative in expected_published_files},
        )

        checksum_pattern = re.compile(r"^([a-f0-9]{64})  ([^\n]+)$")

        def verify_checksums(name: str, expected_paths: set[str]) -> None:
            lines = (published / name).read_text(encoding="utf-8").splitlines()
            parsed: list[tuple[str, str]] = []
            for line in lines:
                match = checksum_pattern.fullmatch(line)
                self.assertIsNotNone(match, line)
                assert match is not None
                parsed.append((match.group(1), match.group(2)))
            paths = [relative for _digest, relative in parsed]
            self.assertEqual(paths, sorted(paths))
            self.assertEqual(len(paths), len(set(paths)))
            self.assertEqual(set(paths), expected_paths)
            for expected_digest, relative in parsed:
                self.assertEqual(
                    hashlib.sha256((published / relative).read_bytes()).hexdigest(),
                    expected_digest,
                    relative,
                )

        verify_checksums(
            "PUBLICATION-SHA256SUMS",
            expected_published_files - {"PUBLICATION-SHA256SUMS"},
        )
        finalized_files = {
            "claim-boundary.json",
            "finalization.json",
            "independent-review.json",
            "recovery-supplement.json",
            "review-bindings.json",
            "sanitization-map.json",
        }
        verify_checksums("SHA256SUMS", finalized_files)

        supplement = json.loads(
            (published / "recovery-supplement.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            supplement["summary_counts"],
            {
                "logical_digest_count": 40,
                "not_found": 1,
                "recovered_matching": 39,
                "recovered_mismatched": 0,
            },
        )
        self.assertEqual(len(supplement["items"]), 40)
        not_found = [
            item for item in supplement["items"] if item["disposition"] == "not-found"
        ]
        self.assertEqual(len(not_found), 1)
        self.assertEqual(
            not_found[0]["logical_item_id"],
            "source_and_compilation.raw_model_file_manifest",
        )
        recovered_entries = {
            item["recovered_artifact_entry_id"]
            for item in supplement["items"]
            if item["disposition"] == "recovered-matching"
        }

        self.assertEqual(len(recovered_entries), 38)
        self.assertEqual(
            supplement["additional_search_items"],
            [
                {
                    "disposition": "not-found",
                    "item_id": "python-test-transcript",
                    "reason_code": "ORIGINAL_TRANSCRIPT_NOT_FOUND",
                    "search_scope_codes": [
                        "source-host-boundary",
                        "verified-copy-one",
                        "verified-copy-two",
                    ],
                }
            ],
        )
        self.assertEqual(len(supplement["copy_verification_receipts"]), 2)
        self.assertEqual(
            {
                receipt["failure_domain_id"]
                for receipt in supplement["copy_verification_receipts"]
            },
            {
                "domain_c8dab4e2d1afefd9e2bf69d567a571d1",
                "domain_94097c623c94c0afd983448303f1f905",
            },
        )
        self.assertEqual(supplement["retrieval_receipt"]["result"], "passed")
        self.assertEqual(supplement["retention_receipt"]["result"], "active")
        self.assertEqual(supplement["retention_policy"]["minimum_calendar_months"], 24)
        self.assertEqual(supplement["independent_review"]["status"], "pending")

        review = json.loads(
            (published / "independent-review.json").read_text(encoding="utf-8")
        )
        self.assertEqual(review["result"], "passed")
        self.assertEqual(review["reason_code"], "NONE")
        self.assertEqual(
            review["checks"],
            {
                "claim-boundary-correctness": True,
                "complete-raw-to-public-traceability": True,
                "complete-sorted-unique-sha256sums": True,
                "numeric-recomputation": True,
                "private-value-absence": True,
                "strict-public-schema": True,
            },
        )

        candidate = json.loads(
            (published / "publication-candidate.json").read_text(encoding="utf-8")
        )
        decision = json.loads(
            (published / "publication-decision.json").read_text(encoding="utf-8")
        )
        decision_binding = json.loads(
            (published / "publication-decision-binding.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            candidate["candidate_id"], "candidate_69cc090b8061cc4a086571f2fb9b3f69"
        )
        self.assertEqual(
            candidate["primary_artifact"]["raw_manifest_sha256"],
            "f35b3383fd58263e7964f301dcadd9369e7b19b1fa85a2ce5d09e2348058f8b7",
        )
        self.assertTrue(decision["eligible"])
        self.assertEqual(decision["reason_codes"], [])
        self.assertEqual(
            decision["candidate"]["candidate_id"], candidate["candidate_id"]
        )
        self.assertEqual(
            decision_binding["decision_id"],
            "decision_55b4e5f4f12d497b4144272c2aa5ebe5",
        )
        self.assertEqual(
            decision_binding["decision_raw_manifest_sha256"],
            "4edb3c58a19f93027ed9ab726eb8830edcbbaf991c88ee4a9d465a4432bceb66",
        )

        for relative in expected_published_files:
            contents = (published / relative).read_text(encoding="utf-8")
            for forbidden in ("/Users/", "/Volumes/", "/private/tmp", "/home/"):
                self.assertNotIn(forbidden, contents, (relative, forbidden))

        packet_link = (
            "operations/evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md"
        )
        self.assertIn(packet_link, (REPOSITORY / "docs/index.md").read_text())
        self.assertIn(
            "evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md",
            (REPOSITORY / "docs/operations/index.md").read_text(),
        )
        self.assertIn(
            "docs/operations/evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md",
            (REPOSITORY / "ROADMAP.md").read_text(),
        )

    def test_cuda_phase5_stopping_outcome_is_bound_and_sanitized(self) -> None:
        packet = (
            REPOSITORY
            / "docs/operations/evidence/2026-08-09-cuda-phase5-repeatability-anchor"
        )
        expected_files = {"README.md", "SHA256SUMS", "phase5-outcome.json"}
        actual_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)

        checksum_pattern = re.compile(r"^([a-f0-9]{64})  (\./[^\n]+)$")
        parsed = []
        for line in (packet / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            match = checksum_pattern.fullmatch(line)
            self.assertIsNotNone(match, line)
            assert match is not None
            parsed.append((match.group(1), match.group(2)))
        self.assertEqual(
            [relative for _digest, relative in parsed],
            ["./README.md", "./phase5-outcome.json"],
        )
        for expected_digest, relative in parsed:
            self.assertEqual(
                hashlib.sha256(
                    (packet / relative.removeprefix("./")).read_bytes()
                ).hexdigest(),
                expected_digest,
            )

        outcome = json.loads((packet / "phase5-outcome.json").read_text())
        self.assertEqual(
            outcome["decision"]["phase5_status"], "complete-not-established"
        )
        self.assertFalse(outcome["aggregate"]["repeatability_anchor_established"])
        self.assertFalse(outcome["decision"]["phase6_authorized"])
        self.assertEqual(outcome["aggregate"]["started_measured_slots"], 0)
        self.assertEqual(len(outcome["measured_slots"]), 5)
        self.assertTrue(
            all(
                slot["slot_status"] == "planned-not-started"
                for slot in outcome["measured_slots"]
            )
        )
        public_text = "\n".join(
            (packet / name).read_text(encoding="utf-8")
            for name in ("README.md", "phase5-outcome.json")
        )
        for private_marker in ("/home/", "/Users/", "GPU-", "192.168."):
            self.assertNotIn(private_marker, public_text)

    def test_cuda_phase5_repeatability_anchor_is_bound_and_sanitized(self) -> None:
        packet = (
            REPOSITORY
            / "docs/operations/evidence/2026-08-10-cuda-phase5-repeatability-anchor"
        )
        expected_files = {
            "README.md",
            "SHA256SUMS",
            "independent-review.json",
            "phase5-outcome.json",
            "sanitization-map.json",
        }
        actual_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)

        checksum_pattern = re.compile(r"^([a-f0-9]{64})  (\./[^\n]+)$")
        parsed = []
        for line in (packet / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            match = checksum_pattern.fullmatch(line)
            self.assertIsNotNone(match, line)
            assert match is not None
            parsed.append((match.group(1), match.group(2)))
        self.assertEqual(
            [relative for _digest, relative in parsed],
            [
                "./README.md",
                "./phase5-outcome.json",
                "./sanitization-map.json",
                "./independent-review.json",
            ],
        )
        self.assertEqual(len(parsed), len({relative for _digest, relative in parsed}))
        for expected_digest, relative in parsed:
            self.assertEqual(
                hashlib.sha256(
                    (packet / relative.removeprefix("./")).read_bytes()
                ).hexdigest(),
                expected_digest,
            )

        outcome = json.loads((packet / "phase5-outcome.json").read_text())
        aggregate = outcome["aggregate"]
        self.assertEqual(outcome["decision"]["phase5_status"], "complete-established")
        self.assertTrue(aggregate["repeatability_anchor_established"])
        self.assertTrue(outcome["decision"]["phase6_authorized"])
        self.assertEqual(
            (
                aggregate["required_measured_slots"],
                aggregate["started_measured_slots"],
                aggregate["protocol_valid_native_passes"],
                aggregate["replacements"],
            ),
            (5, 5, 5, 0),
        )
        self.assertEqual(aggregate["completed_non_skipped_optimizer_steps_total"], 640)
        self.assertTrue(aggregate["duration_stability"]["passed"])
        self.assertTrue(aggregate["peak_device_memory_stability"]["passed"])
        self.assertEqual(aggregate["minimum_telemetry_coverage"], 1.0)
        self.assertLessEqual(aggregate["maximum_telemetry_gap_seconds"], 2.5)
        self.assertTrue(aggregate["all_copy_verifications_passed"])
        self.assertTrue(aggregate["all_retrieval_verifications_passed"])
        self.assertEqual(len(outcome["measured_slots"]), 5)
        self.assertTrue(
            all(
                slot["slot_status"] == "started"
                and slot["native_outcome"] == "passed"
                and slot["evidence_status"] == "protocol-valid"
                and slot["completed_non_skipped_optimizer_steps"] == 128
                and slot["copy_verification"] == "passed"
                and slot["retrieval_verification"] == "passed"
                for slot in outcome["measured_slots"]
            )
        )

        review = json.loads((packet / "independent-review.json").read_text())
        sanitization = json.loads((packet / "sanitization-map.json").read_text())
        self.assertEqual(review["review_result"], "passed")
        self.assertTrue(review["role_separation_verified"])
        self.assertTrue(all(review["checks"].values()))
        self.assertEqual(sanitization["public_output"], "phase5-outcome.json")
        self.assertEqual(len(sanitization["field_groups"]), 7)

        public_text = "\n".join(
            (packet / name).read_text(encoding="utf-8") for name in expected_files
        )
        for private_marker in (
            "/home/",
            "/Users/",
            "Sherminator",
            "192.168.",
            "fd21:",
            "wts@",
        ):
            self.assertNotIn(private_marker, public_text)

        packet_link = "evidence/2026-08-10-cuda-phase5-repeatability-anchor/README.md"
        for relative in (
            "docs/operations/index.md",
            "docs/operations/cuda-empirical-campaign.md",
            "docs/reference/capability-matrix.md",
            "docs/product/current-capabilities.md",
        ):
            self.assertIn(packet_link, (REPOSITORY / relative).read_text())

    def test_cuda_phase6_method_matrix_is_bound_and_sanitized(self) -> None:
        packet = (
            REPOSITORY / "docs/operations/evidence/2026-08-10-cuda-phase6-method-matrix"
        )
        expected_files = {
            "README.md",
            "SHA256SUMS",
            "independent-review.json",
            "phase6-outcome.json",
            "sanitization-map.json",
        }
        actual_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)

        checksum_pattern = re.compile(r"^([a-f0-9]{64})  (\./[^\n]+)$")
        parsed = []
        for line in (packet / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            match = checksum_pattern.fullmatch(line)
            self.assertIsNotNone(match, line)
            assert match is not None
            parsed.append((match.group(1), match.group(2)))
        self.assertEqual(
            [relative for _digest, relative in parsed],
            [
                "./README.md",
                "./phase6-outcome.json",
                "./sanitization-map.json",
                "./independent-review.json",
            ],
        )
        for expected_digest, relative in parsed:
            self.assertEqual(
                hashlib.sha256(
                    (packet / relative.removeprefix("./")).read_bytes()
                ).hexdigest(),
                expected_digest,
            )

        outcome = json.loads((packet / "phase6-outcome.json").read_text())
        self.assertEqual(outcome["decision"]["phase6_status"], "complete-no-promotion")
        self.assertFalse(outcome["decision"]["phase7_authorized"])
        self.assertFalse(outcome["decision"]["confirmatory_comparison_authorized"])
        self.assertEqual(outcome["aggregate"]["predeclared_slot_count"], 32)
        self.assertEqual(outcome["aggregate"]["methods_promoted"], [])
        self.assertEqual(outcome["aggregate"]["confirmatory_slots_started"], 0)
        self.assertEqual(outcome["aggregate"]["replacements"], 0)
        self.assertEqual(len(outcome["slot_dispositions"]), 32)
        self.assertEqual(
            [slot["ordinal"] for slot in outcome["slot_dispositions"]],
            list(range(1, 33)),
        )
        self.assertEqual(
            {
                method: sum(
                    slot["method"] == method for slot in outcome["slot_dispositions"]
                )
                for method in ("full", "lora", "int8-lora", "qlora")
            },
            {"full": 8, "lora": 8, "int8-lora": 8, "qlora": 8},
        )
        measurement = outcome["qualifying_measurement"]
        self.assertEqual(measurement["completed_non_skipped_optimizer_steps"], 128)
        self.assertEqual(measurement["native_outcome"], "passed")
        self.assertEqual(measurement["evidence_status"], "protocol-valid")
        self.assertEqual(measurement["copy_verification"], "passed")
        self.assertEqual(measurement["retrieval_verification"], "passed")

        review = json.loads((packet / "independent-review.json").read_text())
        self.assertEqual(review["review_result"], "passed")
        self.assertTrue(review["role_separation_verified"])
        self.assertTrue(all(review["checks"].values()))

        public_text = "\n".join(
            (packet / name).read_text(encoding="utf-8") for name in expected_files
        )
        for private_marker in (
            "/home/",
            "/Users/",
            "Sherminator",
            "192.168.",
            "fd21:",
            "wts@",
            "GPU-",
        ):
            self.assertNotIn(private_marker, public_text)

        packet_link = "evidence/2026-08-10-cuda-phase6-method-matrix/README.md"
        for relative in (
            "docs/operations/index.md",
            "docs/operations/cuda-empirical-campaign.md",
            "docs/reference/capability-matrix.md",
            "docs/product/current-capabilities.md",
        ):
            self.assertIn(packet_link, (REPOSITORY / relative).read_text())

    def test_cuda_phase6_remediation_matrix_is_bound_and_sanitized(self) -> None:
        packet = (
            REPOSITORY
            / "docs/operations/evidence/2026-08-10-cuda-phase6-remediation-matrix"
        )
        expected_files = {
            "README.md",
            "SHA256SUMS",
            "independent-review.json",
            "phase6-outcome.json",
            "sanitization-map.json",
        }
        actual_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)

        checksum_pattern = re.compile(r"^([a-f0-9]{64})  (\./[^\n]+)$")
        parsed = []
        for line in (packet / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            match = checksum_pattern.fullmatch(line)
            self.assertIsNotNone(match, line)
            assert match is not None
            parsed.append((match.group(1), match.group(2)))
        checksum_paths = [relative for _digest, relative in parsed]
        self.assertEqual(checksum_paths, sorted(checksum_paths))
        self.assertEqual(
            set(checksum_paths),
            {f"./{relative}" for relative in expected_files - {"SHA256SUMS"}},
        )
        for expected_digest, relative in parsed:
            self.assertEqual(
                hashlib.sha256(
                    (packet / relative.removeprefix("./")).read_bytes()
                ).hexdigest(),
                expected_digest,
            )

        outcome = json.loads((packet / "phase6-outcome.json").read_text())
        aggregate = outcome["aggregate"]
        self.assertEqual(
            outcome["decision"]["phase6_status"], "complete-no-stable-method"
        )
        self.assertFalse(outcome["decision"]["phase7_authorized"])
        self.assertTrue(outcome["decision"]["confirmatory_comparison_authorized"])
        self.assertEqual(aggregate["predeclared_slot_count"], 32)
        self.assertEqual(aggregate["matrix_slots_started"], 11)
        self.assertEqual(aggregate["matrix_slots_planned_not_started"], 21)
        self.assertEqual(aggregate["methods_exploratory_promoted"], ["full"])
        self.assertEqual(aggregate["stable_confirmatory_methods"], [])
        self.assertEqual(aggregate["replacements"], 0)

        started = outcome["started_slot_dispositions"]
        self.assertEqual(len(started), 11)
        self.assertEqual(sum(slot["native_outcome"] == "passed" for slot in started), 6)
        self.assertEqual(
            sum(slot["native_outcome"] == "cancelled" for slot in started), 5
        )
        self.assertTrue(
            all(
                slot["evidence_status"] == "protocol-valid"
                and slot["capture_reason_code"] == "NONE"
                and slot["seal_verification"] == "passed"
                and slot["telemetry_status"] == "healthy"
                and slot["copy_verification"] == "passed"
                and slot["retrieval_verification"] == "passed"
                for slot in started
            )
        )
        self.assertEqual(outcome["planned_not_started"]["total"], 21)
        self.assertEqual(sum(outcome["custody"].values()), 60)

        decisions = {
            decision["method"]: decision for decision in outcome["method_decisions"]
        }
        self.assertTrue(decisions["full"]["promoted"])
        self.assertFalse(decisions["full"]["stable"])
        self.assertFalse(decisions["lora"]["promoted"])
        self.assertFalse(decisions["int8-lora"]["admitted"])
        self.assertFalse(decisions["qlora"]["admitted"])

        review = json.loads((packet / "independent-review.json").read_text())
        self.assertEqual(review["review_result"], "passed")
        self.assertTrue(review["role_separation_verified"])
        self.assertTrue(all(review["checks"].values()))
        for name, digest in (
            ("README.md", review["inputs"]["readme_sha256"]),
            ("phase6-outcome.json", review["inputs"]["phase6_outcome_sha256"]),
            ("sanitization-map.json", review["inputs"]["sanitization_map_sha256"]),
        ):
            self.assertEqual(
                hashlib.sha256((packet / name).read_bytes()).hexdigest(), digest
            )

        public_text = "\n".join(
            (packet / name).read_text(encoding="utf-8") for name in expected_files
        )
        for private_marker in (
            "/home/",
            "/Users/",
            "Sherminator",
            "192.168.",
            "fd21:",
            "wts@",
        ):
            self.assertNotIn(private_marker, public_text)
        self.assertIsNone(
            re.search(
                r'"(?:cohort|cell|slot|artifact|receipt|xrun|job)_[a-f0-9]+"',
                public_text,
            )
        )

        packet_link = "evidence/2026-08-10-cuda-phase6-remediation-matrix/README.md"
        for relative in (
            "docs/operations/index.md",
            "docs/operations/cuda-empirical-campaign.md",
        ):
            self.assertIn(packet_link, (REPOSITORY / relative).read_text())
        self.assertIn(
            "operations/evidence/2026-08-10-cuda-phase6-remediation-matrix/README.md",
            (REPOSITORY / "docs/index.md").read_text(),
        )

    def test_cuda_phase6_confirmatory_stability_is_bound_and_sanitized(self) -> None:
        packet = (
            REPOSITORY
            / "docs/operations/evidence/2026-08-10-cuda-phase6-confirmatory-stability"
        )
        expected_files = {
            "README.md",
            "SHA256SUMS",
            "independent-review.json",
            "phase6-outcome.json",
            "sanitization-map.json",
        }
        actual_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)

        checksum_pattern = re.compile(r"^([a-f0-9]{64})  (\./[^\n]+)$")
        parsed = []
        for line in (packet / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            match = checksum_pattern.fullmatch(line)
            self.assertIsNotNone(match, line)
            assert match is not None
            parsed.append((match.group(1), match.group(2)))
        checksum_paths = [relative for _digest, relative in parsed]
        self.assertEqual(checksum_paths, sorted(checksum_paths))
        self.assertEqual(
            set(checksum_paths),
            {f"./{relative}" for relative in expected_files - {"SHA256SUMS"}},
        )
        for expected_digest, relative in parsed:
            self.assertEqual(
                hashlib.sha256(
                    (packet / relative.removeprefix("./")).read_bytes()
                ).hexdigest(),
                expected_digest,
            )

        outcome = json.loads((packet / "phase6-outcome.json").read_text())
        aggregate = outcome["aggregate"]
        slots = outcome["measured_slots"]
        self.assertEqual(outcome["decision"]["phase6_status"], "complete-stable-full")
        self.assertTrue(outcome["decision"]["phase7_authorized"])
        self.assertEqual(aggregate["stable_confirmatory_methods"], ["full"])
        self.assertEqual(aggregate["required_confirmatory_slots"], 5)
        self.assertEqual(aggregate["started_confirmatory_slots"], 5)
        self.assertEqual(aggregate["protocol_valid_native_passes"], 5)
        self.assertEqual(aggregate["replacements"], 0)
        self.assertEqual(aggregate["completed_non_skipped_optimizer_steps_total"], 640)
        self.assertTrue(aggregate["duration_stability"]["passed"])
        self.assertLessEqual(
            aggregate["duration_stability"][
                "median_absolute_deviation_to_median_ratio"
            ],
            0.1,
        )
        self.assertLessEqual(
            aggregate["duration_stability"]["maximum_to_minimum_ratio"], 1.2
        )
        self.assertTrue(aggregate["peak_device_memory_stability"]["passed"])
        self.assertGreaterEqual(aggregate["minimum_telemetry_coverage"], 0.99)
        self.assertLessEqual(aggregate["maximum_telemetry_gap_seconds"], 2.5)
        self.assertEqual([slot["ordinal"] for slot in slots], [1, 2, 3, 4, 5])
        self.assertEqual(
            [slot["scheduled_seed"] for slot in slots],
            [1009, 2003, 3001, 4001, 5003],
        )
        self.assertTrue(
            all(
                slot["native_outcome"] == "passed"
                and slot["evidence_status"] == "protocol-valid"
                and slot["completed_non_skipped_optimizer_steps"] == 128
                and slot["telemetry_healthy"]
                and slot["telemetry_coverage"] >= 0.99
                and slot["telemetry_maximum_gap_seconds"] <= 2.5
                and slot["copy_verification"] == "passed"
                and slot["retrieval_verification"] == "passed"
                and slot["reason_code"] == "NONE"
                and slot["capture_reason_code"] == "NONE"
                and not slot["non_none_event_reason_codes"]
                for slot in slots
            )
        )
        self.assertFalse(outcome["prior_nonqualifying_cohort"]["aggregate_inclusion"])
        self.assertEqual(
            outcome["prior_nonqualifying_cohort"][
                "cumulative_slots_reused_in_successful_cohort"
            ],
            0,
        )

        review = json.loads((packet / "independent-review.json").read_text())
        self.assertEqual(review["review_result"], "passed")
        self.assertTrue(review["role_separation_verified"])
        self.assertTrue(all(review["checks"].values()))
        for name, digest in (
            ("README.md", review["inputs"]["readme_sha256"]),
            ("phase6-outcome.json", review["inputs"]["phase6_outcome_sha256"]),
            ("sanitization-map.json", review["inputs"]["sanitization_map_sha256"]),
        ):
            self.assertEqual(
                hashlib.sha256((packet / name).read_bytes()).hexdigest(), digest
            )

        public_text = "\n".join(
            (packet / name).read_text(encoding="utf-8") for name in expected_files
        )
        for private_marker in (
            "/home/",
            "/Users/",
            "Sherminator",
            "192.168.",
            "fd21:",
            "wts@",
            "GPU-",
        ):
            self.assertNotIn(private_marker, public_text)

        packet_link = "evidence/2026-08-10-cuda-phase6-confirmatory-stability/README.md"
        for relative in (
            "docs/operations/index.md",
            "docs/operations/cuda-empirical-campaign.md",
            "docs/reference/capability-matrix.md",
            "docs/product/current-capabilities.md",
        ):
            self.assertIn(packet_link, (REPOSITORY / relative).read_text())
        self.assertIn(
            "operations/evidence/2026-08-10-cuda-phase6-confirmatory-stability/README.md",
            (REPOSITORY / "docs/index.md").read_text(),
        )

    def test_cuda_phase7_scale_staircase_is_bound_and_sanitized(self) -> None:
        packet = (
            REPOSITORY
            / "docs/operations/evidence/2026-08-10-cuda-phase7-scale-staircase"
        )
        expected_files = {
            "README.md",
            "SHA256SUMS",
            "independent-review.json",
            "phase7-outcome.json",
            "sanitization-map.json",
        }
        actual_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)

        checksum_pattern = re.compile(r"^([a-f0-9]{64})  (\./[^\n]+)$")
        parsed = []
        for line in (packet / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            match = checksum_pattern.fullmatch(line)
            self.assertIsNotNone(match, line)
            assert match is not None
            parsed.append((match.group(1), match.group(2)))
        checksum_paths = [relative for _digest, relative in parsed]
        self.assertEqual(checksum_paths, sorted(checksum_paths))
        self.assertEqual(
            set(checksum_paths),
            {f"./{relative}" for relative in expected_files - {"SHA256SUMS"}},
        )
        for expected_digest, relative in parsed:
            self.assertEqual(
                hashlib.sha256(
                    (packet / relative.removeprefix("./")).read_bytes()
                ).hexdigest(),
                expected_digest,
            )

        outcome = json.loads((packet / "phase7-outcome.json").read_text())
        aggregate = outcome["aggregate"]
        self.assertEqual(
            outcome["decision"]["phase7_status"],
            "complete-stopped-at-live-admission",
        )
        self.assertEqual(
            outcome["decision"]["reason_code"], "THERMAL_WARNING_SUSTAINED"
        )
        self.assertFalse(outcome["decision"]["phase8_authorized"])
        self.assertEqual(aggregate["exploratory_slots_planned"], 18)
        self.assertEqual(aggregate["exploratory_slots_started"], 1)
        self.assertEqual(aggregate["exploratory_slots_planned_not_started"], 17)
        self.assertEqual(aggregate["protocol_valid_native_passes"], 1)
        self.assertEqual(aggregate["completed_non_skipped_optimizer_steps_total"], 128)
        self.assertEqual(aggregate["replacements"], 0)
        self.assertEqual(aggregate["stable_phase7_cells"], [])
        self.assertFalse(aggregate["architecture_breadth_activated"])
        self.assertFalse(aggregate["phase8_authorized"])

        slots = outcome["slot_dispositions"]
        self.assertEqual([slot["scheduled_seed"] for slot in slots], [6101, 6203, 6301])
        passing = slots[0]
        self.assertEqual(passing["slot_status"], "started")
        self.assertEqual(passing["native_outcome"], "passed")
        self.assertEqual(passing["evidence_status"], "protocol-valid")
        self.assertEqual(passing["completed_non_skipped_optimizer_steps"], 128)
        self.assertGreaterEqual(passing["telemetry_coverage"], 0.99)
        self.assertLessEqual(passing["telemetry_maximum_gap_seconds"], 2.5)
        self.assertEqual(passing["copy_verification"], "passed")
        self.assertEqual(passing["retrieval_verification"], "passed")
        self.assertEqual(slots[1]["slot_status"], "planned-not-started")
        self.assertIsNone(slots[1]["native_outcome"])
        self.assertEqual(slots[1]["evidence_status"], "not-started")
        self.assertEqual(slots[1]["reason_code"], "THERMAL_WARNING_SUSTAINED")
        self.assertEqual(slots[2]["slot_status"], "planned-not-started")
        self.assertEqual(
            sum(group["count"] for group in outcome["unstarted_groups"]), 15
        )
        self.assertEqual(2 + 15, aggregate["exploratory_slots_planned_not_started"])

        review = json.loads((packet / "independent-review.json").read_text())
        self.assertEqual(review["review_result"], "passed")
        self.assertTrue(review["role_separation_verified"])
        self.assertTrue(all(review["checks"].values()))
        for name, digest in (
            ("README.md", review["inputs"]["readme_sha256"]),
            ("phase7-outcome.json", review["inputs"]["phase7_outcome_sha256"]),
            ("sanitization-map.json", review["inputs"]["sanitization_map_sha256"]),
        ):
            self.assertEqual(
                hashlib.sha256((packet / name).read_bytes()).hexdigest(), digest
            )

        public_text = "\n".join(
            (packet / name).read_text(encoding="utf-8") for name in expected_files
        )
        for private_marker in (
            "/home/",
            "/Users/",
            "Sherminator",
            "192.168.",
            "fd21:",
            "wts@",
            "GPU-",
        ):
            self.assertNotIn(private_marker, public_text)

        packet_link = "evidence/2026-08-10-cuda-phase7-scale-staircase/README.md"
        for relative in (
            "docs/operations/index.md",
            "docs/operations/cuda-empirical-campaign.md",
            "docs/reference/capability-matrix.md",
            "docs/product/current-capabilities.md",
        ):
            self.assertIn(packet_link, (REPOSITORY / relative).read_text())
        self.assertIn(
            "operations/evidence/2026-08-10-cuda-phase7-scale-staircase/README.md",
            (REPOSITORY / "docs/index.md").read_text(),
        )

    def test_cuda_phase7_same_family_stability_is_bound_and_sanitized(
        self,
    ) -> None:
        packet = (
            REPOSITORY
            / "docs/operations/evidence/2026-08-11-cuda-phase7-same-family-stability"
        )
        expected_files = {
            "README.md",
            "SHA256SUMS",
            "independent-review.json",
            "phase7-outcome.json",
            "sanitization-map.json",
        }
        actual_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)

        checksum_pattern = re.compile(r"^([a-f0-9]{64})  (\./[^\n]+)$")
        parsed = []
        for line in (packet / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            match = checksum_pattern.fullmatch(line)
            self.assertIsNotNone(match, line)
            assert match is not None
            parsed.append((match.group(1), match.group(2)))
        checksum_paths = [relative for _digest, relative in parsed]
        self.assertEqual(checksum_paths, sorted(checksum_paths))
        self.assertEqual(
            set(checksum_paths),
            {f"./{relative}" for relative in expected_files - {"SHA256SUMS"}},
        )
        for expected_digest, relative in parsed:
            self.assertEqual(
                hashlib.sha256(
                    (packet / relative.removeprefix("./")).read_bytes()
                ).hexdigest(),
                expected_digest,
            )

        outcome = json.loads((packet / "phase7-outcome.json").read_text())
        aggregate = outcome["aggregate"]
        self.assertEqual(
            outcome["decision"]["phase7_same_family_status"],
            "complete-reviewed",
        )
        self.assertEqual(
            outcome["decision"]["architecture_breadth_status"],
            "requires-separate-reviewed-protocol-amendment",
        )
        self.assertFalse(outcome["decision"]["phase8_authorized"])
        self.assertFalse(outcome["decision"]["historical_cohort_resumed"])
        self.assertEqual(aggregate["exploratory_slots_planned"], 18)
        self.assertEqual(aggregate["exploratory_slots_started"], 9)
        self.assertEqual(aggregate["exploratory_slots_planned_not_started"], 9)
        self.assertEqual(aggregate["protocol_valid_native_passes"], 9)
        self.assertEqual(aggregate["conditioning_protocol_valid_native_passes"], 3)
        self.assertEqual(aggregate["replacements"], 0)
        self.assertEqual(
            aggregate["stable_phase7_cells"],
            ["135m:lora", "135m:full", "360m:lora"],
        )
        self.assertFalse(aggregate["architecture_breadth_activated"])
        self.assertFalse(aggregate["phase8_authorized"])
        self.assertEqual(
            sum(group["count"] for group in outcome["unstarted_groups"]), 9
        )

        stable_cells = outcome["stable_cells"]
        self.assertEqual(len(stable_cells), 3)
        for cell in stable_cells:
            self.assertTrue(cell["stable"])
            self.assertEqual(len(cell["slots"]), 3)
            self.assertLessEqual(cell["aggregates"]["duration_mad_to_median"], 0.1)
            self.assertLessEqual(cell["aggregates"]["duration_maximum_to_minimum"], 1.2)
            self.assertLessEqual(
                cell["aggregates"]["peak_device_memory_range_bytes"],
                cell["aggregates"]["peak_device_memory_range_limit_bytes"],
            )
            self.assertTrue(
                all(
                    slot["completed_non_skipped_optimizer_steps"] == 128
                    and slot["telemetry_coverage"] >= 0.99
                    and slot["telemetry_maximum_gap_seconds"] <= 2.5
                    for slot in cell["slots"]
                )
            )

        custody = outcome["custody"]
        self.assertEqual(custody["deeply_verified_sealed_artifacts"], 12)
        self.assertEqual(custody["verified_off_host_copies"], 12)
        self.assertEqual(custody["verified_fresh_restorations"], 12)

        review = json.loads((packet / "independent-review.json").read_text())
        self.assertEqual(review["review_result"], "passed")
        self.assertTrue(review["role_separation_verified"])
        self.assertTrue(all(review["checks"].values()))
        for name, digest in (
            ("README.md", review["inputs"]["readme_sha256"]),
            ("phase7-outcome.json", review["inputs"]["phase7_outcome_sha256"]),
            ("sanitization-map.json", review["inputs"]["sanitization_map_sha256"]),
        ):
            self.assertEqual(
                hashlib.sha256((packet / name).read_bytes()).hexdigest(), digest
            )

        public_text = "\n".join(
            (packet / name).read_text(encoding="utf-8") for name in expected_files
        )
        for private_marker in (
            "/home/",
            "/Users/",
            "Sherminator",
            "192.168.",
            "fd21:",
            "wts@",
            "GPU-",
        ):
            self.assertNotIn(private_marker, public_text)

        packet_link = "evidence/2026-08-11-cuda-phase7-same-family-stability/README.md"
        for relative in (
            "docs/operations/index.md",
            "docs/operations/cuda-empirical-campaign.md",
            "docs/reference/capability-matrix.md",
            "docs/product/current-capabilities.md",
        ):
            self.assertIn(packet_link, (REPOSITORY / relative).read_text())
        self.assertIn(
            "operations/evidence/2026-08-11-cuda-phase7-same-family-stability/README.md",
            (REPOSITORY / "docs/index.md").read_text(),
        )

    def test_cuda_phase7_breadth_amendment_is_bound_and_sanitized(self) -> None:
        packet = (
            REPOSITORY
            / "docs/operations/evidence/2026-08-11-cuda-phase7-breadth-amendment"
        )
        expected_files = {
            "README.md",
            "SHA256SUMS",
            "amendment.json",
            "independent-review.json",
            "sanitization-map.json",
        }
        actual_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)

        checksum_pattern = re.compile(r"^([a-f0-9]{64})  (\./[^\n]+)$")
        parsed = []
        for line in (packet / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            match = checksum_pattern.fullmatch(line)
            self.assertIsNotNone(match, line)
            assert match is not None
            parsed.append((match.group(1), match.group(2)))
        checksum_paths = [relative for _digest, relative in parsed]
        self.assertEqual(checksum_paths, sorted(checksum_paths))
        self.assertEqual(
            set(checksum_paths),
            {f"./{relative}" for relative in expected_files - {"SHA256SUMS"}},
        )
        for expected_digest, relative in parsed:
            self.assertEqual(
                hashlib.sha256(
                    (packet / relative.removeprefix("./")).read_bytes()
                ).hexdigest(),
                expected_digest,
            )

        amendment = json.loads((packet / "amendment.json").read_text())
        self.assertEqual(amendment["decision"]["review_status"], "passed")
        self.assertEqual(
            amendment["decision"]["admitted_cells_after_merge"],
            ["qwen3-0p6b:lora"],
        )
        self.assertEqual(
            amendment["decision"]["planned_exploratory_slots_after_merge"], 3
        )
        self.assertFalse(amendment["decision"]["attempt_ledger_created_before_review"])
        self.assertFalse(amendment["decision"]["training_started_before_review"])
        self.assertFalse(amendment["decision"]["replacement_allowed"])
        self.assertFalse(amendment["decision"]["phase8_authorized"])
        self.assertEqual(
            amendment["frozen_contract"]["method_order"],
            ["lora", "full", "int8-lora", "qlora"],
        )
        self.assertEqual(
            amendment["frozen_contract"]["training_seeds"], [6101, 6203, 6301]
        )

        models = {item["label"]: item for item in amendment["models"]}
        self.assertEqual(
            {
                label: (item["model_id"], item["revision"])
                for label, item in models.items()
            },
            {
                "qwen3-0p6b": (
                    "Qwen/Qwen3-0.6B",
                    "c1899de289a04d12100db370d81485cdf75e47ca",
                ),
                "gemma3-1b-it": (
                    "google/gemma-3-1b-it",
                    "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
                ),
                "mistral-7b-v0p3": (
                    "mistralai/Mistral-7B-v0.3",
                    "caa1feb0e54d415e2df31207e5f4e273e33509b1",
                ),
            },
        )
        qwen = models["qwen3-0p6b"]
        self.assertEqual(qwen["artifact"]["exact_execution_artifact_bytes"], 1519182365)
        self.assertEqual(qwen["artifact"]["exact_parameter_count"], 751632384)
        self.assertEqual(len(qwen["artifact"]["files"]), 7)
        self.assertEqual(
            sum(item["size_bytes"] for item in qwen["artifact"]["files"]),
            qwen["artifact"]["exact_execution_artifact_bytes"],
        )
        self.assertEqual(
            set(qwen["artifact"]["target_module_tensor_counts"].values()), {28}
        )
        self.assertEqual(
            [
                item["method"]
                for item in qwen["method_dispositions"]
                if item["admitted"]
            ],
            ["lora"],
        )
        self.assertEqual(
            models["gemma3-1b-it"]["disposition"], "excluded-license-not-accepted"
        )
        self.assertEqual(
            models["mistral-7b-v0p3"]["disposition"], "planner-not-admitted"
        )
        self.assertTrue(
            all(
                not method["admitted"]
                for label in ("gemma3-1b-it", "mistral-7b-v0p3")
                for method in models[label]["method_dispositions"]
            )
        )

        review = json.loads((packet / "independent-review.json").read_text())
        self.assertEqual(review["review_result"], "passed")
        self.assertTrue(review["role_separation_verified"])
        self.assertTrue(all(review["checks"].values()))
        for name, digest in (
            ("README.md", review["inputs"]["readme_sha256"]),
            ("amendment.json", review["inputs"]["amendment_sha256"]),
            ("sanitization-map.json", review["inputs"]["sanitization_map_sha256"]),
        ):
            self.assertEqual(
                hashlib.sha256((packet / name).read_bytes()).hexdigest(), digest
            )

        public_text = "\n".join(
            (packet / name).read_text(encoding="utf-8") for name in expected_files
        )
        for private_marker in (
            "/home/",
            "/Users/",
            "Sherminator",
            "192.168.",
            "fd21:",
            "wts@",
            "GPU-",
        ):
            self.assertNotIn(private_marker, public_text)

        packet_link = "evidence/2026-08-11-cuda-phase7-breadth-amendment/README.md"
        for relative in (
            "docs/operations/index.md",
            "docs/operations/cuda-empirical-campaign.md",
            "docs/reference/capability-matrix.md",
            "docs/product/current-capabilities.md",
        ):
            self.assertIn(packet_link, (REPOSITORY / relative).read_text())
        self.assertIn(
            "operations/evidence/2026-08-11-cuda-phase7-breadth-amendment/README.md",
            (REPOSITORY / "docs/index.md").read_text(),
        )

        protocol = json.loads(
            (REPOSITORY / "docs/reference/cuda-campaign-protocol.v1.json").read_text()
        )
        breadth = protocol["matrix_contract"]["phase7_scale_staircase"][
            "architecture_breadth_amendment"
        ]
        self.assertEqual(breadth["review_status"], "passed")
        self.assertFalse(breadth["phase8_authorized"])
        self.assertEqual(
            [
                (item["model_id"], item["revision"], item["admitted_methods"])
                for item in breadth["models"]
            ],
            [
                (
                    "Qwen/Qwen3-0.6B",
                    "c1899de289a04d12100db370d81485cdf75e47ca",
                    ["lora"],
                ),
                (
                    "google/gemma-3-1b-it",
                    "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
                    [],
                ),
                (
                    "mistralai/Mistral-7B-v0.3",
                    "caa1feb0e54d415e2df31207e5f4e273e33509b1",
                    [],
                ),
            ],
        )

    def test_cuda_phase7_breadth_stability_is_bound_and_sanitized(self) -> None:
        packet = (
            REPOSITORY
            / "docs/operations/evidence/2026-08-11-cuda-phase7-breadth-stability"
        )
        expected_files = {
            "README.md",
            "SHA256SUMS",
            "independent-review.json",
            "phase7-outcome.json",
            "sanitization-map.json",
        }
        actual_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)

        checksum_pattern = re.compile(r"^([a-f0-9]{64})  (\./[^\n]+)$")
        parsed = []
        for line in (packet / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            match = checksum_pattern.fullmatch(line)
            self.assertIsNotNone(match, line)
            assert match is not None
            parsed.append((match.group(1), match.group(2)))
        self.assertEqual(
            [relative for _digest, relative in parsed],
            sorted(relative for _digest, relative in parsed),
        )
        self.assertEqual(
            {relative for _digest, relative in parsed},
            {f"./{relative}" for relative in expected_files - {"SHA256SUMS"}},
        )
        for expected_digest, relative in parsed:
            self.assertEqual(
                hashlib.sha256(
                    (packet / relative.removeprefix("./")).read_bytes()
                ).hexdigest(),
                expected_digest,
            )

        outcome = json.loads((packet / "phase7-outcome.json").read_text())
        self.assertEqual(outcome["decision"]["phase7_status"], "complete-reviewed")
        self.assertEqual(
            outcome["decision"]["phase7_architecture_breadth_status"],
            "complete-reviewed",
        )
        self.assertFalse(outcome["decision"]["phase8_authorized"])
        self.assertFalse(outcome["decision"]["replacement_allowed"])
        self.assertFalse(outcome["decision"]["replacement_created"])
        self.assertFalse(outcome["decision"]["stopped_cohorts_resumed"])
        self.assertEqual(
            outcome["model_freeze"]["exact_unique_loaded_parameter_count"],
            596049920,
        )
        self.assertEqual(
            outcome["model_freeze"]["serialized_state_dict_tensor_elements"],
            751632384,
        )
        self.assertEqual(outcome["configuration"]["training_seeds"], [6101, 6203, 6301])
        stable = outcome["stable_cell"]
        self.assertTrue(stable["stable"])
        self.assertTrue(all(stable["checks"].values()))
        self.assertEqual(
            [slot["scheduled_seed"] for slot in stable["slots"]],
            [6101, 6203, 6301],
        )
        self.assertTrue(
            all(
                slot["completed_non_skipped_optimizer_steps"] == 128
                for slot in stable["slots"]
            )
        )
        self.assertEqual(stable["aggregates"]["peak_device_memory_range_bytes"], 0)
        self.assertEqual(
            outcome["custody"],
            {
                "deeply_verified_sealed_artifacts": 4,
                "protected_conditioning_artifacts": 1,
                "protected_exploratory_artifacts": 3,
                "verified_fresh_restorations": 4,
                "verified_off_host_copies": 4,
            },
        )

        review = json.loads((packet / "independent-review.json").read_text())
        self.assertEqual(review["review_result"], "passed")
        self.assertTrue(review["role_separation_verified"])
        self.assertTrue(all(review["checks"].values()))
        self.assertFalse(review["phase8_authorized"])
        for name, digest in (
            ("README.md", review["inputs"]["readme_sha256"]),
            ("phase7-outcome.json", review["inputs"]["phase7_outcome_sha256"]),
            ("sanitization-map.json", review["inputs"]["sanitization_map_sha256"]),
        ):
            self.assertEqual(
                hashlib.sha256((packet / name).read_bytes()).hexdigest(), digest
            )

        public_text = "\n".join(
            (packet / name).read_text(encoding="utf-8") for name in expected_files
        )
        for private_marker in (
            "/home/",
            "/Users/",
            "Sherminator",
            "192.168.",
            "fd21:",
            "wts@",
            "GPU-",
        ):
            self.assertNotIn(private_marker, public_text)

        packet_link = "evidence/2026-08-11-cuda-phase7-breadth-stability/README.md"
        for relative in (
            "docs/operations/index.md",
            "docs/operations/cuda-empirical-campaign.md",
            "docs/reference/capability-matrix.md",
            "docs/reference/cuda-campaign-protocol.md",
            "docs/product/current-capabilities.md",
        ):
            self.assertIn(packet_link, (REPOSITORY / relative).read_text())
        self.assertIn(
            "operations/evidence/2026-08-11-cuda-phase7-breadth-stability/README.md",
            (REPOSITORY / "docs/index.md").read_text(),
        )

    def test_cuda_phase8_guarded_frontier_is_bound_and_sanitized(self) -> None:
        packet = (
            REPOSITORY
            / "docs/operations/evidence/2026-08-11-cuda-phase8-guarded-frontier"
        )
        expected_files = {
            "README.md",
            "SHA256SUMS",
            "independent-review.json",
            "phase8-outcome.json",
            "sanitization-map.json",
        }
        actual_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)

        checksum_pattern = re.compile(r"^([a-f0-9]{64})  (\./[^\n]+)$")
        parsed = []
        for line in (packet / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            match = checksum_pattern.fullmatch(line)
            self.assertIsNotNone(match, line)
            assert match is not None
            parsed.append((match.group(1), match.group(2)))
        self.assertEqual(
            [relative for _digest, relative in parsed],
            sorted(relative for _digest, relative in parsed),
        )
        self.assertEqual(
            {relative for _digest, relative in parsed},
            {f"./{relative}" for relative in expected_files - {"SHA256SUMS"}},
        )
        for expected_digest, relative in parsed:
            self.assertEqual(
                hashlib.sha256(
                    (packet / relative.removeprefix("./")).read_bytes()
                ).hexdigest(),
                expected_digest,
            )

        outcome = json.loads((packet / "phase8-outcome.json").read_text())
        self.assertEqual(outcome["aggregate"]["phase8_status"], "complete-reviewed")
        self.assertEqual(outcome["aggregate"]["predeclared_slots"], 17)
        self.assertEqual(outcome["aggregate"]["started_slots"], 16)
        self.assertEqual(outcome["aggregate"]["native_protocol_valid_passes"], 14)
        self.assertEqual(outcome["aggregate"]["bounded_pilot_cuda_oom_nonpasses"], 2)
        self.assertEqual(outcome["aggregate"]["planned_not_started_slots"], 1)
        self.assertFalse(outcome["aggregate"]["full_training_executed"])
        self.assertFalse(outcome["decision"]["phase9_authorized"])
        self.assertFalse(outcome["decision"]["phase9_activation_performed"])
        self.assertFalse(outcome["decision"]["replacement_allowed"])
        self.assertFalse(outcome["decision"]["replacement_created"])
        self.assertEqual(
            outcome["managed_profile"]["actions"],
            ["dependencies", "model-data", "measured-preflight", "bounded-pilot"],
        )
        self.assertFalse(
            outcome["managed_profile"]["confirmed_training_action_present"]
        )

        axes = outcome["axis_results"]
        self.assertEqual(axes["sequence_length"]["endpoint"]["sequence_length"], 1024)
        self.assertEqual(axes["sequence_length"]["endpoint"]["reason_code"], "CUDA_OOM")
        self.assertEqual(axes["sequence_length"]["planned_not_started"], [2048])
        self.assertEqual(
            axes["effective_batch_size"]["endpoint_kind"],
            "right-censored-highest-passing-rung",
        )
        self.assertEqual(
            axes["effective_batch_size"]["endpoint"]["effective_batch_size"], 64
        )
        self.assertEqual(
            axes["micro_batch_and_accumulation"]["endpoint"]["micro_batch_size"],
            16,
        )
        self.assertEqual(
            axes["micro_batch_and_accumulation"]["endpoint"]["reason_code"],
            "CUDA_OOM",
        )

        selection = outcome["headroom_selection"]
        selected = selection["selected_candidate"]
        self.assertEqual(
            selected["point"],
            {
                "effective_batch_size": 32,
                "gradient_accumulation_steps": 8,
                "micro_batch_size": 4,
                "sequence_length": 256,
            },
        )
        self.assertEqual(selected["rank_score_sequence_times_effective_batch"], 8192)
        self.assertEqual(
            [
                row["rank_score_sequence_times_effective_batch"]
                for row in selection["eligible_axis_candidates"]
            ],
            [8192, 4096, 2048],
        )
        self.assertGreaterEqual(
            selected["observed"]["minimum_free_vram_bytes"],
            selection["thresholds"]["minimum_free_vram_bytes"],
        )
        self.assertTrue(
            all(
                row["minimum_free_vram_bytes"]
                < selection["thresholds"]["minimum_free_vram_bytes"]
                for row in selection["rejected_predecessors"]
            )
        )
        self.assertEqual(
            outcome["custody"],
            {
                "deeply_verified_sealed_artifacts": 16,
                "manifested_bytes_per_layer": 42133048,
                "manifested_files_per_layer": 604,
                "verified_fresh_restorations": 16,
                "verified_off_host_copies": 16,
            },
        )

        review = json.loads((packet / "independent-review.json").read_text())
        self.assertEqual(review["review_result"], "passed")
        self.assertEqual(review["phase8_status"], "complete-reviewed")
        self.assertFalse(review["phase9_authorized"])
        self.assertTrue(review["role_separation_verified"])
        self.assertTrue(all(review["checks"].values()))
        for name, digest in (
            ("README.md", review["inputs"]["readme_sha256"]),
            ("phase8-outcome.json", review["inputs"]["phase8_outcome_sha256"]),
            ("sanitization-map.json", review["inputs"]["sanitization_map_sha256"]),
        ):
            self.assertEqual(
                hashlib.sha256((packet / name).read_bytes()).hexdigest(), digest
            )

        public_text = "\n".join(
            (packet / name).read_text(encoding="utf-8") for name in expected_files
        )
        for private_marker in (
            "/home/",
            "/Users/",
            "Sherminator",
            "192.168.",
            "fd21:",
            "wts@",
            "GPU-",
        ):
            self.assertNotIn(private_marker, public_text)

        packet_link = "evidence/2026-08-11-cuda-phase8-guarded-frontier/README.md"
        for relative in (
            "docs/operations/index.md",
            "docs/operations/cuda-empirical-campaign.md",
            "docs/reference/capability-matrix.md",
            "docs/product/current-capabilities.md",
        ):
            self.assertIn(packet_link, (REPOSITORY / relative).read_text())
        self.assertIn(
            "operations/evidence/2026-08-11-cuda-phase8-guarded-frontier/README.md",
            (REPOSITORY / "docs/index.md").read_text(),
        )

    def test_cuda_phase9_endurance_is_bound_and_sanitized(self) -> None:
        packet = (
            REPOSITORY / "docs/operations/evidence/2026-08-11-cuda-phase9-endurance"
        )
        expected_files = {
            "README.md",
            "SHA256SUMS",
            "independent-review.json",
            "phase9-outcome.json",
            "sanitization-map.json",
        }
        actual_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)

        checksum_pattern = re.compile(r"^([a-f0-9]{64})  (\./[^\n]+)$")
        parsed = []
        for line in (packet / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            match = checksum_pattern.fullmatch(line)
            self.assertIsNotNone(match, line)
            assert match is not None
            parsed.append((match.group(1), match.group(2)))
        self.assertEqual(
            [relative for _digest, relative in parsed],
            sorted(relative for _digest, relative in parsed),
        )
        self.assertEqual(
            {relative for _digest, relative in parsed},
            {f"./{relative}" for relative in expected_files - {"SHA256SUMS"}},
        )
        for expected_digest, relative in parsed:
            self.assertEqual(
                hashlib.sha256(
                    (packet / relative.removeprefix("./")).read_bytes()
                ).hexdigest(),
                expected_digest,
            )

        outcome = json.loads((packet / "phase9-outcome.json").read_text())
        self.assertEqual(outcome["aggregate"]["phase9_status"], "complete-reviewed")
        self.assertEqual(outcome["aggregate"]["measured_slots"], 3)
        self.assertEqual(outcome["aggregate"]["native_protocol_valid_passes"], 3)
        self.assertEqual(outcome["aggregate"]["replacements"], 0)
        self.assertFalse(outcome["aggregate"]["cloud_or_external_gpu_used"])
        self.assertEqual(outcome["aggregate"]["job_control_exercises_passed"], 8)
        self.assertEqual(outcome["aggregate"]["job_control_exercises_expected"], 8)
        self.assertFalse(outcome["decision"]["phase10_authorized"])
        self.assertFalse(outcome["decision"]["phase10_performed"])
        self.assertFalse(outcome["decision"]["replacement_allowed"])
        self.assertFalse(outcome["decision"]["replacement_created"])
        self.assertEqual(
            outcome["frozen_configuration"],
            {
                "checkpoint_cadence_optimizer_steps": 64,
                "dataset_fixture_sha256": "6d90599e949bf2698b940e0c159e1fa24f3dc0c162005546bd270fc761aac7f2",
                "effective_batch_size": 32,
                "emergency_deadline_seconds": 5400,
                "gradient_accumulation_steps": 8,
                "method": "lora",
                "micro_batch_size": 4,
                "model_id": "Qwen/Qwen3-0.6B",
                "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                "optimizer_step_target": 300,
                "precision": "bf16",
                "sequence_length": 256,
            },
        )
        self.assertEqual(len(outcome["slots"]), 3)
        self.assertTrue(
            all(
                slot["optimizer_steps"] == 300
                and slot["micro_iterations"] == 2400
                and slot["examples_consumed"] == 9600
                and slot["native_outcome"] == "passed"
                and slot["evidence_status"] == "protocol-valid"
                and slot["custody_verification"] == "passed"
                and slot["telemetry_coverage"] == 1.0
                and slot["replacement_created"] is False
                for slot in outcome["slots"]
            )
        )
        self.assertEqual(
            outcome["totals"]["completed_non_skipped_optimizer_steps"], 900
        )
        self.assertEqual(outcome["totals"]["micro_iterations"], 7200)
        self.assertEqual(outcome["totals"]["examples_consumed"], 28800)
        self.assertTrue(outcome["metric_interpretation"]["aggregate_rates_only"])
        self.assertFalse(outcome["metric_interpretation"]["cross_run_drift_claim_made"])
        self.assertFalse(
            outcome["metric_interpretation"]["phase5_or_phase6_stability_ratio_applied"]
        )
        self.assertEqual(
            outcome["metric_interpretation"]["final_export_verification_level"],
            "structural-file-tree",
        )
        self.assertEqual(
            outcome["custody"],
            {
                "deeply_verified_sealed_artifacts": 3,
                "manifested_bytes_per_layer": 17569834,
                "manifested_files_per_layer": 129,
                "verified_fresh_restorations": 3,
                "verified_off_host_copies": 3,
            },
        )
        self.assertTrue(outcome["job_controls"]["all_passed"])
        self.assertFalse(outcome["job_controls"]["gpu_training_invoked"])

        review = json.loads((packet / "independent-review.json").read_text())
        self.assertEqual(review["review_result"], "passed")
        self.assertEqual(review["phase9_status"], "complete-reviewed")
        self.assertFalse(review["phase10_authorized"])
        self.assertTrue(all(review["checks"].values()))
        self.assertEqual(
            hashlib.sha256((packet / "phase9-outcome.json").read_bytes()).hexdigest(),
            review["inputs"]["phase9_outcome_sha256"],
        )

        public_text = "\n".join(
            (packet / name).read_text(encoding="utf-8") for name in expected_files
        )
        for private_marker in (
            "/home/",
            "/Users/",
            "Sherminator",
            "192.168.",
            "fd21:",
            "wts@",
            "GPU-",
            "artifact_",
            "receipt_",
            "operation_",
        ):
            self.assertNotIn(private_marker, public_text)

        packet_link = "evidence/2026-08-11-cuda-phase9-endurance/README.md"
        for relative in (
            "docs/operations/index.md",
            "docs/operations/cuda-empirical-campaign.md",
            "docs/reference/capability-matrix.md",
            "docs/product/current-capabilities.md",
        ):
            self.assertIn(packet_link, (REPOSITORY / relative).read_text())
        self.assertIn(
            "operations/evidence/2026-08-11-cuda-phase9-endurance/README.md",
            (REPOSITORY / "docs/index.md").read_text(),
        )

    def test_cuda_phase10_certification_is_bound_reviewed_and_sanitized(
        self,
    ) -> None:
        packet = (
            REPOSITORY
            / "docs/operations/evidence/2026-08-11-cuda-phase10-certification"
        )
        expected_files = {
            "README.md",
            "SHA256SUMS",
            "independent-review.json",
            "phase10-certification.json",
            "phase10-decision.json",
            "sanitization-map.json",
        }
        actual_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)

        checksum_pattern = re.compile(r"^([a-f0-9]{64})  (\./[^\n]+)$")
        parsed = []
        for line in (packet / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            match = checksum_pattern.fullmatch(line)
            self.assertIsNotNone(match, line)
            assert match is not None
            parsed.append((match.group(1), match.group(2)))
        self.assertEqual(
            [relative for _digest, relative in parsed],
            sorted(relative for _digest, relative in parsed),
        )
        self.assertEqual(
            {relative for _digest, relative in parsed},
            {f"./{relative}" for relative in expected_files - {"SHA256SUMS"}},
        )
        for expected_digest, relative in parsed:
            self.assertEqual(
                hashlib.sha256(
                    (packet / relative.removeprefix("./")).read_bytes()
                ).hexdigest(),
                expected_digest,
            )

        certification = json.loads(
            (packet / "phase10-certification.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            certification["bindings"]["phase10_source_commit"],
            "3c56b37799be16d4698b6e75e76c690b4ba8e818",
        )
        self.assertEqual(
            certification["bindings"]["phase10_source_tree"],
            "76d38631026fe55c0015a0608eae6d45500e4f8d",
        )
        self.assertEqual(
            certification["bindings"]["current_protocol_sha256"],
            "005de18e60d8b707986dee6eef4b7199e7e1fecba385ab591ff708671b858335",
        )
        self.assertEqual(
            certification["ledger_totals"],
            {
                "evidence_statuses": {"capture-invalid": 1, "protocol-valid": 57},
                "native_outcomes": {
                    "cancelled": 7,
                    "failed": 3,
                    "passed": 47,
                    "unknown": 1,
                },
                "planned": 149,
                "planned_not_started": 91,
                "qualifying_passes": 47,
                "replacement_runs": 0,
                "started": 58,
            },
        )
        self.assertEqual(len(certification["qualifying_stable_cells"]), 6)
        self.assertEqual(len(certification["phase8_frontier_points"]), 17)
        started_frontier = [
            point
            for point in certification["phase8_frontier_points"]
            if point["slot_status"] == "started"
        ]
        self.assertEqual(len(started_frontier), 16)
        self.assertTrue(
            all(point["full_training_executed"] is False for point in started_frontier)
        )
        self.assertEqual(
            sum(point["native_outcome"] == "passed" for point in started_frontier),
            14,
        )
        self.assertEqual(
            sum(point["reason_code"] == "CUDA_OOM" for point in started_frontier),
            2,
        )
        endurance = certification["phase9_endurance"]
        self.assertEqual(
            endurance["totals"]["completed_non_skipped_optimizer_steps"], 900
        )
        self.assertEqual(endurance["job_controls"]["exercise_count"], 8)
        self.assertTrue(endurance["job_controls"]["all_passed"])
        self.assertFalse(endurance["job_controls"]["gpu_training_invoked"])
        self.assertEqual(
            certification["release_gate_mapping"]["aptus_v0_2_release_readiness"],
            "not-established",
        )
        self.assertFalse(certification["phase10_decision"]["new_training_performed"])
        self.assertFalse(
            certification["phase10_decision"][
                "cloud_or_external_gpu_acquisition_performed"
            ]
        )

        review = json.loads(
            (packet / "independent-review.json").read_text(encoding="utf-8")
        )
        self.assertEqual(review["status"], "complete-reviewed")
        self.assertTrue(all(review["checks"].values()))
        self.assertEqual(
            review["counts"],
            {
                "deeply_verified_protected_artifacts": 68,
                "verified_prior_public_packets": 13,
                "verified_stable_cells": 6,
            },
        )
        self.assertEqual(
            review["review_input_bindings"]["phase10_certification_sha256"],
            hashlib.sha256(
                (packet / "phase10-certification.json").read_bytes()
            ).hexdigest(),
        )

        decision = json.loads(
            (packet / "phase10-decision.json").read_text(encoding="utf-8")
        )
        self.assertEqual(decision["status"], "complete-reviewed")
        self.assertTrue(decision["decision"]["campaign_certification_complete"])
        self.assertFalse(decision["decision"]["release_ready"])
        self.assertFalse(decision["decision"]["phase11_authorized"])
        self.assertEqual(
            decision["bindings"]["independent_review_sha256"],
            hashlib.sha256(
                (packet / "independent-review.json").read_bytes()
            ).hexdigest(),
        )

        public_text = "\n".join(
            (packet / name).read_text(encoding="utf-8") for name in expected_files
        )
        for private_marker in (
            "/home/",
            "/Users/",
            "Sherminator",
            "192.168.",
            "fd21:",
            "wts@",
            "GPU-",
            "artifact_",
            "receipt_",
            "operation_",
        ):
            self.assertNotIn(private_marker, public_text)

        packet_link = "evidence/2026-08-11-cuda-phase10-certification/README.md"
        for relative in (
            "docs/operations/index.md",
            "docs/operations/cuda-empirical-campaign.md",
            "docs/operations/release-gates.md",
            "docs/reference/capability-matrix.md",
            "docs/product/current-capabilities.md",
        ):
            self.assertIn(packet_link, (REPOSITORY / relative).read_text())
        self.assertIn(
            "operations/evidence/2026-08-11-cuda-phase10-certification/README.md",
            (REPOSITORY / "docs/index.md").read_text(),
        )

    def test_cuda_phase7_breadth_parameter_correction_is_bound_and_sanitized(
        self,
    ) -> None:
        packet = (
            REPOSITORY / "docs/operations/evidence/"
            "2026-08-11-cuda-phase7-breadth-parameter-correction"
        )
        expected_files = {
            "README.md",
            "SHA256SUMS",
            "correction.json",
            "independent-review.json",
            "sanitization-map.json",
        }
        actual_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)

        checksum_pattern = re.compile(r"^([a-f0-9]{64})  (\./[^\n]+)$")
        parsed = []
        for line in (packet / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            match = checksum_pattern.fullmatch(line)
            self.assertIsNotNone(match, line)
            assert match is not None
            parsed.append((match.group(1), match.group(2)))
        self.assertEqual(
            [relative for _digest, relative in parsed],
            sorted(relative for _digest, relative in parsed),
        )
        self.assertEqual(
            {relative for _digest, relative in parsed},
            {f"./{relative}" for relative in expected_files - {"SHA256SUMS"}},
        )
        for expected_digest, relative in parsed:
            self.assertEqual(
                hashlib.sha256(
                    (packet / relative.removeprefix("./")).read_bytes()
                ).hexdigest(),
                expected_digest,
            )

        correction = json.loads((packet / "correction.json").read_text())
        model = correction["model"]
        self.assertEqual(model["serialized_state_dict_tensor_element_count"], 751632384)
        self.assertEqual(model["unique_loaded_model_parameter_count"], 596049920)
        self.assertTrue(model["tie_word_embeddings"])
        self.assertTrue(model["input_output_share_storage"])
        self.assertEqual(
            model["serialized_state_dict_tensor_element_count"]
            - model["unique_loaded_model_parameter_count"],
            model["input_embedding_elements"],
        )
        stopped = correction["stopped_conditioning"]
        self.assertEqual(stopped["failed_action"], "model-data")
        self.assertEqual(stopped["reason_code"], "PROCESS_EXIT_NONZERO")
        self.assertFalse(stopped["pilot_started"])
        self.assertFalse(stopped["training_started"])
        self.assertEqual(
            [
                item["method"]
                for item in correction["corrected_method_dispositions"]
                if item["admitted"]
            ],
            ["lora"],
        )
        decision = correction["decision"]
        self.assertFalse(decision["stopped_cohort_resumed"])
        self.assertFalse(decision["replacement_slot_created"])
        self.assertEqual(decision["exploratory_slots_started"], 0)
        self.assertTrue(decision["new_cohort_requires_review_after_merge"])
        self.assertFalse(decision["phase8_authorized"])

        review = json.loads((packet / "independent-review.json").read_text())
        self.assertEqual(review["decision"]["status"], "passed")
        self.assertFalse(review["decision"]["stopped_cohort_resumable"])
        self.assertTrue(all(review["checks"].values()))

        public_text = "\n".join(
            (packet / name).read_text(encoding="utf-8") for name in expected_files
        )
        for private_marker in (
            "/home/",
            "/Users/",
            "Sherminator",
            "192.168.",
            "fd21:",
            "wts@",
            "GPU-",
        ):
            self.assertNotIn(private_marker, public_text)

        packet_link = (
            "evidence/2026-08-11-cuda-phase7-breadth-parameter-correction/README.md"
        )
        for relative in (
            "docs/operations/index.md",
            "docs/operations/cuda-empirical-campaign.md",
            "docs/reference/capability-matrix.md",
            "docs/product/current-capabilities.md",
        ):
            self.assertIn(packet_link, (REPOSITORY / relative).read_text())
        self.assertIn(
            "operations/evidence/"
            "2026-08-11-cuda-phase7-breadth-parameter-correction/README.md",
            (REPOSITORY / "docs/index.md").read_text(),
        )

        protocol = json.loads(
            (REPOSITORY / "docs/reference/cuda-campaign-protocol.v1.json").read_text()
        )
        breadth = protocol["matrix_contract"]["phase7_scale_staircase"][
            "architecture_breadth_amendment"
        ]
        self.assertFalse(breadth["first_cohort_resumable"])
        self.assertEqual(
            breadth["first_cohort_status"],
            "stopped-model-data-parameter-semantics-mismatch",
        )
        qwen = breadth["models"][0]
        self.assertEqual(qwen["serialized_state_dict_tensor_element_count"], 751632384)
        self.assertEqual(qwen["unique_loaded_model_parameter_count"], 596049920)
        self.assertEqual(
            qwen["parameter_count_semantics"], "unique_loaded_model_parameters"
        )

    def test_cuda_phase5_retention_addendum_is_bound_and_sanitized(self) -> None:
        relative = (
            "docs/operations/evidence/"
            "2026-08-10-cuda-phase5-repeatability-retention.json"
        )
        addendum_path = REPOSITORY / relative
        addendum = json.loads(addendum_path.read_text())

        self.assertEqual(addendum["status"], "active-qualified")
        self.assertEqual(
            addendum["retention_policy"]["policy_id"],
            "cuda-v02-public-claim-evidence-24m-v1",
        )
        self.assertEqual(
            addendum["retention_policy"]["retain_not_before_utc"],
            "2028-08-11T00:00:00+00:00",
        )
        self.assertTrue(addendum["custody"]["claim_qualified"])
        self.assertTrue(
            addendum["custody"]["durable_private_off_experiment_host_copy_established"]
        )
        self.assertEqual(len(addendum["artifacts"]), 5)
        self.assertEqual(
            [artifact["ordinal"] for artifact in addendum["artifacts"]],
            [1, 2, 3, 4, 5],
        )
        self.assertTrue(
            all(
                artifact["retention_receipt"]["kind"] == "retention"
                and artifact["retention_receipt"]["result"] == "active"
                and re.fullmatch(
                    r"receipt_[a-f0-9]{32}",
                    artifact["retention_receipt"]["receipt_id"],
                )
                and re.fullmatch(
                    r"[a-f0-9]{64}", artifact["retention_receipt"]["sha256"]
                )
                for artifact in addendum["artifacts"]
            )
        )
        self.assertEqual(set(addendum["procedural_review"]["checks"].values()), {True})

        public_text = addendum_path.read_text()
        for private_marker in (
            "/home/",
            "/Users/",
            "Sherminator",
            "192.168.",
            "fd21:",
            "wts@",
        ):
            self.assertNotIn(private_marker, public_text)
        self.assertIn(
            f"evidence/{addendum_path.name}",
            (REPOSITORY / "docs/operations/index.md").read_text(),
        )

    def test_cuda_lora_single_acceptance_packet_is_bound_and_sanitized(
        self,
    ) -> None:
        packet = (
            REPOSITORY
            / "docs/operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance"
        )
        expected_files = {
            "README.md",
            "SHA256SUMS",
            "acceptance-procedure.json",
            "acceptance-summary.json",
            "bundle-manifest.json",
            "clean-plan.json",
            "host-hardware.json",
            "inspection-receipt.json",
            "model-files.sha256",
            "model-policy-snapshot.v1.json",
            "provider-inspection.json",
            "python-packages.txt",
            "raw-artifact-digests.json",
            "runtime-environment.json",
            "runs/run-1/run-summary.json",
        }
        actual_files = {
            path.relative_to(packet).as_posix()
            for path in packet.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)
        packet_snapshot = "".join(
            f"{relative}:{(packet / relative).stat().st_size}:"
            f"{hashlib.sha256((packet / relative).read_bytes()).hexdigest()}\n"
            for relative in sorted(expected_files)
        ).encode("ascii")
        self.assertEqual(
            hashlib.sha256(packet_snapshot).hexdigest(),
            "f329226630a933c520beb818b398c73b69a944ce7cc8dca7c022c9add5646023",
        )

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
        procedure = json.loads(
            (packet / "acceptance-procedure.json").read_text(encoding="utf-8")
        )
        manifest_path = packet / "bundle-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        plan_path = packet / "clean-plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        policy_path = packet / "model-policy-snapshot.v1.json"
        provider_path = packet / "provider-inspection.json"
        provider = json.loads(provider_path.read_text(encoding="utf-8"))
        receipt = json.loads(
            (packet / "inspection-receipt.json").read_text(encoding="utf-8")
        )
        host = json.loads((packet / "host-hardware.json").read_text(encoding="utf-8"))
        runtime = json.loads(
            (packet / "runtime-environment.json").read_text(encoding="utf-8")
        )
        run = json.loads(
            (packet / "runs/run-1/run-summary.json").read_text(encoding="utf-8")
        )
        raw_digests = json.loads(
            (packet / "raw-artifact-digests.json").read_text(encoding="utf-8")
        )

        source_commit = "c12c4d8db0037a2c278a2ad95a0a2cbda4387eed"
        source_tree = "ad482883cfb6ad2b8ac72f7b7d1009c918e5c345"
        bundle_fingerprint = (
            "296fb7b710f60345a590748f053eb15f9b5b4f4b3fec539ae3a705e31d6a640b"
        )
        embedded_plan_sha256 = (
            "b13ed14b416c18e796f64fd3fc41c50466daed6fc69b0c37b4b943ed274f4ad4"
        )
        policy_sha256 = (
            "c2ae989c8b68df6e984dc7c8670397e791ff30e1f5ce82129e25c1c2b93268d8"
        )
        model_revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
        dataset_sha256 = (
            "bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44"
        )

        self.assertEqual(summary["state"], "measured-run-pass")
        self.assertEqual(summary["source"]["acceptance_commit"], source_commit)
        self.assertEqual(summary["source"]["acceptance_tree"], source_tree)
        self.assertEqual(summary["source"]["python_tests_passed"], 550)
        self.assertEqual(summary["repetition"]["qualifying_execution_count"], 1)
        self.assertFalse(summary["repetition"]["repeatability_claimed"])
        self.assertEqual(summary["model"]["revision"], model_revision)
        self.assertEqual(summary["dataset"]["sha256"], dataset_sha256)
        self.assertEqual(summary["dataset"]["rows"], 4)

        self.assertEqual(
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            bundle_fingerprint,
        )
        self.assertEqual(
            hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            embedded_plan_sha256,
        )
        self.assertEqual(
            hashlib.sha256(policy_path.read_bytes()).hexdigest(), policy_sha256
        )
        self.assertEqual(
            hashlib.sha256(provider_path.read_bytes()).hexdigest(),
            summary["model"]["provider_inspection_sha256"],
        )
        self.assertEqual(manifest["schema_version"], "aptus.bundle.v3")
        self.assertEqual(manifest["plan_sha256"], embedded_plan_sha256)
        self.assertEqual(manifest["policy_snapshot_sha256"], policy_sha256)
        self.assertEqual(manifest["plan_id"], summary["compiled_input"]["plan_id"])
        self.assertEqual(
            manifest["candidate_id"], summary["compiled_input"]["candidate_id"]
        )
        self.assertEqual(plan["schema_version"], "aptus.training-plan.v5")
        self.assertEqual(plan["plan_id"], manifest["plan_id"])
        self.assertEqual(plan["dataset"]["source_path"], "data/dataset.jsonl")
        self.assertEqual(plan["model"]["revision"], model_revision)
        self.assertEqual(plan["recommended"]["method"], "lora")
        self.assertEqual(plan["recommended"]["distribution"], "single")
        self.assertEqual(plan["recommended"]["world_size"], 1)
        self.assertEqual(plan["recommended"]["device_indices"], [0])
        self.assertEqual(provider["inspection_receipt"], receipt)
        self.assertEqual(receipt["resolved_revision"], model_revision)

        self.assertEqual(host["host"]["operating_system"], "Ubuntu 24.04.4 LTS")
        self.assertEqual(host["gpu"]["name"], "NVIDIA GeForce RTX 3050")
        self.assertEqual(host["gpu"]["compute_capability"], "8.6")
        self.assertEqual(host["gpu"]["torch_visible_total_memory_bytes"], 8220573696)
        self.assertEqual(host["handoff"]["active_cuda_compute_process_count"], 0)
        self.assertFalse(host["handoff"]["aptus_gpu_lease_present"])
        self.assertFalse(host["privacy"]["hostname_included"])
        self.assertFalse(host["privacy"]["network_identifiers_included"])
        self.assertFalse(host["privacy"]["gpu_uuid_included"])
        self.assertFalse(host["privacy"]["process_identifiers_included"])
        self.assertEqual(
            runtime["environment_binding_sha256"],
            summary["runtime"]["environment_binding_sha256"],
        )
        self.assertEqual(runtime["imports"]["torch"], "2.13.0+cu130")
        self.assertEqual(runtime["cuda"]["torch_cuda_runtime"], "13.0")
        self.assertTrue(runtime["checks"]["pip_check_passed"])
        self.assertTrue(runtime["checks"]["real_cuda_tensor_probe_passed"])
        self.assertFalse(runtime["inventories"]["bitsandbytes_installed"])

        packages_path = packet / "python-packages.txt"
        packages = packages_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(packages), 62)
        for direct_pin in (
            "torch==2.13.0",
            "transformers==5.14.1",
            "accelerate==1.14.0",
            "safetensors==0.8.0",
            "peft==0.19.1",
            "aptus==0.2.0",
        ):
            self.assertIn(direct_pin, packages)
        self.assertFalse(any(line.startswith("bitsandbytes==") for line in packages))
        self.assertEqual(
            hashlib.sha256(packages_path.read_bytes()).hexdigest(),
            runtime["inventories"]["python_packages_sha256"],
        )

        model_manifest_path = packet / "model-files.sha256"
        model_manifest = model_manifest_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(model_manifest), 25)
        self.assertTrue(
            all(re.fullmatch(r"[a-f0-9]{64}  [^\n]+", line) for line in model_manifest)
        )
        self.assertEqual(
            hashlib.sha256(model_manifest_path.read_bytes()).hexdigest(),
            summary["model"]["model_file_manifest_sha256"],
        )
        self.assertTrue(summary["model"]["upstream_generated_paths_sanitized"])
        self.assertEqual(
            summary["model"]["raw_model_file_manifest_sha256"],
            raw_digests["source_and_compilation"]["raw_model_file_manifest"],
        )
        self.assertEqual(
            [line.split("  ", 1)[1] for line in model_manifest if "runs/" in line],
            [
                "runs/upstream-tensorboard-event-01",
                "runs/upstream-tensorboard-event-02",
            ],
        )

        expected_actions = ["dependency", "model-data", "preflight", "pilot", "train"]
        self.assertEqual([job["action"] for job in run["jobs"]], expected_actions)
        self.assertTrue(
            all(
                job["state"] == "completed"
                and job["return_code"] == 0
                and job["error"] is None
                for job in run["jobs"]
            )
        )
        self.assertEqual(
            [job["job_id"] for job in run["jobs"]],
            summary["qualifying_workflow"]["job_ids"],
        )
        self.assertEqual(run["bindings"]["bundle_fingerprint"], bundle_fingerprint)
        self.assertEqual(run["bindings"]["policy_snapshot_sha256"], policy_sha256)
        self.assertEqual(run["bindings"]["plan_id"], manifest["plan_id"])
        self.assertEqual(run["bindings"]["candidate_id"], manifest["candidate_id"])
        self.assertEqual(run["bindings"]["model_revision"], model_revision)
        self.assertEqual(run["bindings"]["dataset_sha256"], dataset_sha256)
        self.assertEqual(
            run["bindings"]["environment_sha256"],
            runtime["environment_binding_sha256"],
        )
        for job in run["jobs"]:
            self.assertEqual(job["artifact_fingerprint"], bundle_fingerprint)
            self.assertEqual(
                job["authorized_model_policy_snapshot_sha256"], policy_sha256
            )
            self.assertEqual(job["runtime_interpreter"], "accepted-runtime-python")
        binding_verification = run["job_binding_verification"]
        self.assertTrue(
            binding_verification["all_job_records_share_bundle_fingerprint"]
        )
        self.assertTrue(
            binding_verification["all_job_records_share_authorized_policy_snapshot"]
        )
        self.assertTrue(
            binding_verification["all_job_commands_use_accepted_runtime_interpreter"]
        )
        accepted_environment = binding_verification["accepted_environment_binding"]
        self.assertEqual(
            accepted_environment["sha256"], run["bindings"]["environment_sha256"]
        )
        self.assertEqual(accepted_environment["source"], "terminal-validation-report")
        self.assertFalse(accepted_environment["persisted_on_each_job_record"])
        self.assertFalse(binding_verification["raw_commands_committed"])

        capacity_checks = run["capacity_checks"]
        validation_capacity = capacity_checks["validation_report_snapshot"]
        parent_capacity = capacity_checks["train_job_parent_prelaunch"]
        self.assertEqual(validation_capacity["source"], "terminal-validation-report")
        self.assertEqual(parent_capacity["source"], "persisted-train-job-record")
        for required_key in (
            "required_free_cuda_bytes",
            "required_host_ram_bytes",
            "required_training_output_disk_bytes",
            "required_checkpoint_disk_bytes",
            "required_final_export_disk_bytes",
        ):
            self.assertEqual(
                validation_capacity[required_key], parent_capacity[required_key]
            )
        self.assertNotEqual(
            validation_capacity["free_cuda_bytes"], parent_capacity["free_cuda_bytes"]
        )

        self.assertTrue(run["pilot"]["phase_two_resumed"])
        self.assertTrue(run["pilot"]["checkpoint_continuation_observed"])
        self.assertEqual(run["pilot"]["phase_one"]["global_step"], 1)
        self.assertEqual(run["pilot"]["phase_two"]["resumed_from_checkpoint_step"], 1)
        self.assertEqual(run["pilot"]["phase_two"]["global_step"], 2)
        self.assertTrue(run["pilot"]["census_equal_between_phases"])
        self.assertEqual(
            run["pilot"]["trainable_parameter_census"]["descriptor_sha256"],
            run["full_training"]["trainable_parameter_census"]["descriptor_sha256"],
        )
        self.assertEqual(run["full_training"]["global_step"], 3)
        self.assertEqual(run["full_training"]["training_row_count"], 3)
        self.assertEqual(run["full_training"]["evaluation_row_count"], 1)
        self.assertEqual(
            run["full_training"]["trainable_parameter_census"][
                "trainable_parameter_count"
            ],
            4884480,
        )
        self.assertEqual(
            run["full_training"]["finite_guard_counts"]["non_skipped_optimizer_steps"],
            3,
        )

        export = run["final_export"]
        self.assertEqual(export["schema_version"], "aptus.final-export.v1")
        self.assertEqual(export["verification_level"], "structural-file-tree")
        self.assertEqual(export["method"], "lora")
        self.assertEqual(export["weight_files"], ["adapter_model.safetensors"])
        self.assertEqual(
            export["total_bytes"], sum(entry["size_bytes"] for entry in export["files"])
        )
        adapter = next(
            entry
            for entry in export["files"]
            if entry["path"] == "adapter_model.safetensors"
        )
        self.assertEqual(
            adapter["sha256"],
            "fd3eb151acf70ab072eb8a60186df782370fa182b74dd92f8630591ba7a9dba5",
        )
        terminal = run["terminal_validation"]
        self.assertEqual(terminal["state"], "measured-run-pass")
        self.assertEqual(
            terminal["artifact_integrity_status"], "verified-at-completion"
        )
        self.assertFalse(terminal["pending_or_active_fields_present"])
        self.assertTrue(terminal["parent_promotion"]["evidence_hash_recomputed"])
        self.assertEqual(
            terminal["parent_promotion"]["evidence_sha256"],
            "59562af2c758df7b985bb6d1cc5b8e3eca7f4fbdf06f86e86f45891a19244f66",
        )

        rehearsal = summary["nonqualifying_attempts"][0]
        self.assertEqual(rehearsal["qualification"], "non-qualifying")
        self.assertEqual(rehearsal["completed_gate_jobs"], 4)
        self.assertFalse(rehearsal["train_job_created"])
        self.assertFalse(rehearsal["full_train_launched"])
        self.assertFalse(rehearsal["runtime_evidence_carried_forward"])
        self.assertNotEqual(
            rehearsal["environment_binding_sha256"],
            summary["runtime"]["environment_binding_sha256"],
        )
        self.assertIsNone(
            raw_digests["nonqualifying_rehearsal"]["full_train_job_record"]
        )
        self.assertEqual(len(raw_digests["qualifying_job_records"]), 5)
        self.assertFalse(raw_digests["retention"]["raw_artifacts_committed"])
        self.assertEqual(procedure["qualifying_action_order"], expected_actions)

        forbidden_patterns = (
            "/Users/",
            "/home/",
            "/root/",
            "/private/tmp",
            "/tmp/",
            "192.168.",
            "owner_pid",
            "process_pid",
            "process_group_id",
            "lease_token",
            "private_key",
        )
        for relative in expected_files - {"SHA256SUMS"}:
            text = (packet / relative).read_text(encoding="utf-8").lower()
            for pattern in forbidden_patterns:
                self.assertNotIn(pattern.lower(), text, (relative, pattern))
            for pattern in (
                r"\bip-\d{1,3}(?:-\d{1,3}){3}\b",
                r"events\.out\.tfevents\.\d+\.[^.\s]+\.\d+\.\d+",
            ):
                self.assertIsNone(
                    re.search(pattern, text, flags=re.IGNORECASE),
                    (relative, pattern),
                )

        operations_index = (REPOSITORY / "docs/operations/index.md").read_text(
            encoding="utf-8"
        )
        packet_prefix = "evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/"
        for relative in expected_files:
            self.assertIn(f"{packet_prefix}{relative}", operations_index, relative)
        self.assertIn(
            "does **not** contain the original\nverbose Python test stdout/stderr",
            operations_index,
        )
        self.assertIn(
            "It does not record a digest or durable external\n"
            "location for the Python test transcript itself.",
            operations_index,
        )

        evidence_leaf = "2026-08-06-smollm2-cuda-lora-single-acceptance/README.md"
        for relative in (
            "README.md",
            "SECURITY.md",
            "docs/index.md",
            "docs/getting-started/quickstart.md",
            "docs/operations/apple-silicon-pilot.md",
            "docs/operations/index.md",
            "docs/operations/release-gates.md",
            "docs/reference/capability-matrix.md",
            "docs/product/current-capabilities.md",
            "docs/product/claim-language.md",
            "docs/product/ui-ux.md",
            "docs/maintenance/documentation-debt.md",
            "docs/maintenance/documentation-health.md",
        ):
            text = (REPOSITORY / relative).read_text(encoding="utf-8")
            self.assertIn(evidence_leaf, text, relative)

        packet_readme = " ".join(
            (packet / "README.md").read_text(encoding="utf-8").split()
        )
        self.assertIn("one exact CUDA LoRA single-device workflow", packet_readme)
        self.assertIn("does not establish repeatability", packet_readme)

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
        self.assertNotIn(
            "Publish a concrete private security-reporting route.",
            health,
        )

    def test_maintained_client_contract_closeout_is_documented(self) -> None:
        debt = (REPOSITORY / "docs/maintenance/documentation-debt.md").read_text(
            encoding="utf-8"
        )
        health = (REPOSITORY / "docs/maintenance/documentation-health.md").read_text(
            encoding="utf-8"
        )
        doc_026 = debt.split("### DOC-026:", 1)[1].split("\n## ", 1)[0]
        normalized_doc_026 = " ".join(doc_026.split())

        self.assertIn("**Status:** Resolved", doc_026)
        self.assertIn("all six runtime-inventory fields", normalized_doc_026)
        self.assertIn("all four native HTTP routes", normalized_doc_026)
        self.assertIn(
            "unknown extra response properties stay forward-compatible",
            normalized_doc_026,
        )
        for stale_claim in (
            "maintained React normalization and Swift decoding boundaries still require",
            "Other React normalization code",
            "Close the remaining maintained React normalization and Swift decoder parity",
        ):
            self.assertNotIn(stale_claim, health)

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
            if document.is_relative_to(REPOSITORY / "dev/archive"):
                continue
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

    def test_phase_ten_milestone_is_consistent_across_active_surfaces(self) -> None:
        milestone_documents = (
            "README.md",
            "ROADMAP.md",
            "SECURITY.md",
            "docs/index.md",
            "docs/product/current-capabilities.md",
            "docs/product/claim-language.md",
            "docs/product/vision.md",
            "docs/reference/capability-matrix.md",
            "docs/operations/index.md",
            "docs/operations/cuda-empirical-campaign.md",
            "docs/operations/release-gates.md",
            "docs/getting-started/choose-your-path.md",
            "docs/maintenance/documentation-health.md",
        )
        for relative in milestone_documents:
            text = (REPOSITORY / relative).read_text(encoding="utf-8")
            self.assertIn("Phase 10", text, f"{relative}: missing milestone")

        ledger_documents = (
            "README.md",
            "ROADMAP.md",
            "docs/index.md",
            "docs/product/current-capabilities.md",
            "docs/product/claim-language.md",
            "docs/reference/capability-matrix.md",
            "docs/operations/index.md",
            "docs/operations/cuda-empirical-campaign.md",
            "docs/operations/release-gates.md",
            "docs/getting-started/choose-your-path.md",
            "docs/maintenance/documentation-health.md",
        )
        for relative in ledger_documents:
            text = (REPOSITORY / relative).read_text(encoding="utf-8")
            for claim in ("149", "58", "91", "47"):
                self.assertIn(claim, text, f"{relative}: missing ledger {claim}")

        active_status_documents = (
            "README.md",
            "ROADMAP.md",
            "docs/product/current-capabilities.md",
            "docs/operations/index.md",
        )
        stale_claims = (
            "Review whether to authorize a new frozen Phase 6 cohort",
            "Phase 7 is not authorized.",
            "only the exact SmolLM2 LoRA single-device record is runtime-qualified",
            "only the exact separately recorded LoRA single-device target-host workflow",
        )
        for relative in active_status_documents:
            text = (REPOSITORY / relative).read_text(encoding="utf-8")
            for claim in stale_claims:
                self.assertNotIn(claim, text, f"{relative}: stale milestone claim")

        campaign = (
            REPOSITORY / "docs/operations/cuda-empirical-campaign.md"
        ).read_text(encoding="utf-8")
        self.assertIn("there is no Phase 11", campaign)
        self.assertIn("performed no training", campaign)

        claim_language = (REPOSITORY / "docs/product/claim-language.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("not permission to run full training merely to", claim_language)
        self.assertIn("provoke an OOM", claim_language)


if __name__ == "__main__":
    unittest.main()
