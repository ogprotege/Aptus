from __future__ import annotations

from typing import Any, Iterable


KNOWN_DOMAIN_ASSETS = {
    "SystemArchitecture.md",
    "Integration_Architecture.md",
    "src/hypertuner/task-configs.ts",
    "src/method-constraints.ts",
    "src/model-database.ts",
    "src/model-database-update.ts",
    "src/formulas/target_modules.ts",
    "src/formulas/batch_size_v2.ts",
    "src/formulas/rank_v2.ts",
    "src/formulas/weight_decay_v2.ts",
    "src/formulas/learning_rate_2.ts",
    "src/python/config.py",
    "src/python/core_optimizer.py",
    "src/python/dataset_analyzer.py",
    "src/python/export_model.py",
    "src/python/model_registry.py",
    "src/python/register_dataset.py",
    "src/python/register_model.py",
    "src/python/resource_scanner.py",
    "src/python/script_generator_v2.py",
    "src/python/train.py",
    "src/hypertuner/training/lora_trainer.py",
    "src/hypertuner/evaluation/lora_evaluator.py",
    "docs/reft_methods_guide.md",
}

KNOWN_REFERENCE_ASSETS = {
    "src/optimizer.ts",
    "src/hypertuner/methodSelector.ts",
    "src/output/command_line.ts",
    "src/output/config_file.ts",
    "src/python/dora_decomposer.py",
    "src/python/flexora_optimizer.py",
    "src/python/reft_adapter.py",
    "src/python/reft_enhanced.py",
    "src/python/reft_setup.py",
    "api/FastAPI/main.py",
    "hypertune_cli.py",
}

KNOWN_FAKE_OR_STUBBED = {
    "src/python/tune_service.py",
}

FABRICATED_DATA_PATHS = {
    "HyperTune-NEW_stuff_05-16-25/manual-models.json",
}


def _canonical_rank(path: str) -> tuple[int, int, str]:
    penalty = 0
    if path in KNOWN_DOMAIN_ASSETS:
        penalty -= 1000
    elif path in KNOWN_REFERENCE_ASSETS:
        penalty -= 500
    if " 2" in path:
        penalty += 100
    if path.startswith("src/python 2/"):
        penalty += 100
    if path.startswith("tests/integration 2/"):
        penalty += 100
    if path.startswith("HyperTune-NEW_stuff_"):
        penalty += 40
    if path.startswith("Random/"):
        penalty += 30
    if path.startswith("DeploymentOptions/"):
        penalty += 10
    if path.endswith(".py") and "Guide" in path:
        penalty += 50
    return penalty, len(path), path


def _duplicate_index(
    clusters: Iterable[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for cluster in clusters:
        paths = cluster["paths"]
        canonical = min(paths, key=_canonical_rank)
        for path in paths:
            index[path] = {
                "cluster_id": cluster["cluster_id"],
                "canonical_path": canonical,
            }
    return index


def _static_index(static_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["path"]: item
        for item in static_report.get("files", [])
        if "path" in item
    }


def _base_decision(path: str, kind: str, extension: str) -> dict[str, Any]:
    if path.endswith(".DS_Store") or path == ".DS_Store":
        return {
            "disposition": "DISCARD",
            "operational_viability": 0,
            "knowledge_value": 0,
            "confidence": "high",
            "rationale": "Operating-system metadata with no product or historical value.",
            "warnings": [],
            "evidence_ids": ["EVIDENCE:OS_JUNK"],
        }

    if kind == "empty":
        return {
            "disposition": "DISCARD",
            "operational_viability": 0,
            "knowledge_value": 0,
            "confidence": "high",
            "rationale": "The artifact is byte-empty and contains no recoverable implementation or knowledge.",
            "warnings": [],
            "evidence_ids": ["EVIDENCE:EMPTY_FILE"],
        }

    if path.startswith("PyReft-Repo/"):
        return {
            "disposition": "ARCHIVE",
            "operational_viability": 2,
            "knowledge_value": 4,
            "confidence": "high",
            "rationale": "Vendored third-party research code is useful as reference but must not enter Aptus product code without verified provenance and license obligations.",
            "warnings": ["DO_NOT_SHIP_PROVENANCE"],
            "evidence_ids": ["EVIDENCE:VENDORED_RESEARCH_CODE"],
        }

    if path in FABRICATED_DATA_PATHS:
        return {
            "disposition": "ARCHIVE",
            "operational_viability": 0,
            "knowledge_value": 1,
            "confidence": "high",
            "rationale": "Historical model data includes speculative closed-model internals and is unsuitable as a product data source.",
            "warnings": ["DO_NOT_SHIP_FABRICATED_DATA"],
            "evidence_ids": ["EVIDENCE:UNVERIFIED_CLOSED_MODEL_DATA"],
        }

    if path in KNOWN_FAKE_OR_STUBBED:
        return {
            "disposition": "DISCARD",
            "operational_viability": 0,
            "knowledge_value": 0,
            "confidence": "high",
            "rationale": "The implementation returns hard-coded success metrics rather than performing the advertised operation.",
            "warnings": ["DO_NOT_SHIP_FAKE_METRICS"],
            "evidence_ids": ["EVIDENCE:HARDCODED_FAKE_RESULTS"],
        }

    if path in KNOWN_DOMAIN_ASSETS:
        return {
            "disposition": "ADAPT",
            "operational_viability": 2,
            "knowledge_value": 4,
            "confidence": "medium",
            "rationale": "This artifact contains Aptus-relevant domain logic or a useful implementation seam, but requires validation and integration behind new typed boundaries.",
            "warnings": [],
            "evidence_ids": ["EVIDENCE:KNOWN_DOMAIN_ASSET"],
        }

    if path in KNOWN_REFERENCE_ASSETS:
        return {
            "disposition": "ADAPT",
            "operational_viability": 1,
            "knowledge_value": 3,
            "confidence": "medium",
            "rationale": "The artifact preserves a valuable algorithm or interface shape, but its current implementation is incomplete, unsafe, or contract-incompatible.",
            "warnings": [],
            "evidence_ids": ["EVIDENCE:IMPLEMENTATION_REFERENCE"],
        }

    if path.startswith("Legal Docs/"):
        return {
            "disposition": "DISCARD",
            "operational_viability": 0,
            "knowledge_value": 0,
            "confidence": "high",
            "rationale": "Unfilled generic legal template; it is neither valid Aptus legal advice nor unique product knowledge.",
            "warnings": ["DO_NOT_SHIP_PLACEHOLDER_LEGAL"],
            "evidence_ids": ["EVIDENCE:PLACEHOLDER_TEMPLATE"],
        }

    if path.startswith(("DeploymentOptions/", "deploy/")) or extension in {
        ".toml",
        ".yaml",
        ".yml",
    }:
        return {
            "disposition": "ARCHIVE",
            "operational_viability": 1,
            "knowledge_value": 1,
            "confidence": "medium",
            "rationale": "Historical deployment intent is preserved, but configuration drift prevents direct reuse.",
            "warnings": [],
            "evidence_ids": ["EVIDENCE:LEGACY_DEPLOYMENT_CONFIG"],
        }

    if path.startswith("tests/"):
        return {
            "disposition": "ARCHIVE",
            "operational_viability": 1,
            "knowledge_value": 2,
            "confidence": "medium",
            "rationale": "Legacy test intent may inform future specifications, but the suite is not credible as current verification.",
            "warnings": [],
            "evidence_ids": ["EVIDENCE:LEGACY_TEST_SPEC"],
        }

    if extension in {".py", ".js", ".ts", ".tsx"}:
        return {
            "disposition": "ARCHIVE",
            "operational_viability": 1,
            "knowledge_value": 2,
            "confidence": "medium",
            "rationale": "Unique legacy source is retained for reference, but no evidence currently supports direct reuse.",
            "warnings": [],
            "evidence_ids": ["EVIDENCE:UNPROVEN_LEGACY_SOURCE"],
        }

    return {
        "disposition": "ARCHIVE",
        "operational_viability": 0,
        "knowledge_value": 2,
        "confidence": "medium",
        "rationale": "Unique historical artifact retained until the Aptus extraction decision is complete.",
        "warnings": [],
        "evidence_ids": ["EVIDENCE:HISTORICAL_ARTIFACT"],
    }


def classify_artifacts(
    records: Iterable[dict[str, Any]],
    duplicate_clusters: Iterable[dict[str, Any]],
    static_report: dict[str, Any],
) -> list[dict[str, Any]]:
    duplicates = _duplicate_index(duplicate_clusters)
    static = _static_index(static_report)
    results = []

    for record in sorted(records, key=lambda item: item["path"]):
        path = record["path"]
        duplicate = duplicates.get(path)
        if duplicate and duplicate["canonical_path"] != path:
            decision = {
                "disposition": "DISCARD",
                "operational_viability": 0,
                "knowledge_value": 0,
                "confidence": "high",
                "rationale": "Byte-identical duplicate of the preserved canonical artifact.",
                "warnings": [],
                "evidence_ids": ["EVIDENCE:SHA256_IDENTICAL_DUPLICATE"],
            }
        else:
            decision = _base_decision(
                path,
                record.get("kind", "unknown"),
                record.get("extension", ""),
            )

        static_item = static.get(path)
        if static_item and static_item.get("parse_status") == "failed":
            decision["operational_viability"] = 0
            decision["warnings"] = [
                *decision["warnings"],
                "DO_NOT_SHIP_PARSE_FAILURE",
            ]
            decision["evidence_ids"] = [
                *decision["evidence_ids"],
                "EVIDENCE:STATIC_PARSE_FAILURE",
            ]

        disposition = decision["disposition"]
        if disposition == "ADAPT":
            proposed_destination = "future Aptus design candidate"
        elif disposition == "ARCHIVE":
            proposed_destination = "immutable legacy archive"
        else:
            proposed_destination = None

        results.append(
            {
                "path": path,
                "disposition": disposition,
                "operational_viability": decision["operational_viability"],
                "knowledge_value": decision["knowledge_value"],
                "confidence": decision["confidence"],
                "rationale": decision["rationale"],
                "warnings": sorted(set(decision["warnings"])),
                "evidence_ids": sorted(set(decision["evidence_ids"])),
                "duplicate_cluster_id": (
                    duplicate["cluster_id"] if duplicate else None
                ),
                "canonical_path": (
                    duplicate["canonical_path"] if duplicate else path
                ),
                "proposed_destination": proposed_destination,
            }
        )

    return results
