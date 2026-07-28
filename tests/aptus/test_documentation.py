from __future__ import annotations

import argparse
import json
import re
import unittest
from collections import deque
from pathlib import Path
from urllib.parse import unquote

from aptus.domain import Method
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
