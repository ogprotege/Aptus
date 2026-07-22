from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .classification import classify_artifacts
from .cli import generate_inventory_bundle
from .inventory import write_json, write_jsonl
from .references import analyze_references
from .security import scan_secrets


GENERATED_FILENAMES = (
    "baseline-manifest.json",
    "inventory.jsonl",
    "duplicate-clusters.json",
    "version-families.json",
    "reference-map.json",
    "secret-scan.json",
    "classification.jsonl",
    "classification-summary.json",
    "generated-bundle-manifest.json",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_with_rollback(staging: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    backup = staging / ".previous"
    backup.mkdir()
    existed = set()

    for filename in GENERATED_FILENAMES:
        target = output / filename
        if target.exists():
            shutil.copy2(target, backup / filename)
            existed.add(filename)

    published = []
    try:
        for filename in GENERATED_FILENAMES:
            os.replace(staging / filename, output / filename)
            published.append(filename)
    except Exception:
        for filename in published:
            target = output / filename
            if filename in existed:
                os.replace(backup / filename, target)
            else:
                target.unlink(missing_ok=True)
        raise


def generate_evidence_bundle(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(f"Legacy source is not a directory: {source}")
    output = output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Audit output must be outside the legacy source tree.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".aptus-audit-generation-",
        dir=output.parent,
    ) as temp_dir:
        staging = Path(temp_dir)
        inventory_summary = generate_inventory_bundle(source, staging)
        records = _read_jsonl(staging / "inventory.jsonl")
        duplicate_clusters = _read_json(staging / "duplicate-clusters.json")["clusters"]

        reference_report = analyze_references(source)
        write_json(staging / "reference-map.json", reference_report)

        secret_findings = scan_secrets(source)
        write_json(
            staging / "secret-scan.json",
            {
                "finding_count": len(secret_findings),
                "findings": secret_findings,
            },
        )

        decisions = classify_artifacts(
            records,
            duplicate_clusters,
            reference_report,
        )
        write_jsonl(staging / "classification.jsonl", decisions)

        dispositions = Counter(item["disposition"] for item in decisions)
        warnings = Counter(
            warning for item in decisions for warning in item["warnings"]
        )
        summary = {
            "artifact_count": len(decisions),
            "baseline_manifest_sha256": inventory_summary["manifest_sha256"],
            "dispositions": dict(sorted(dispositions.items())),
            "warning_counts": dict(sorted(warnings.items())),
            "adapt_candidates": [
                item["path"] for item in decisions if item["disposition"] == "ADAPT"
            ],
        }
        write_json(staging / "classification-summary.json", summary)

        bundle_files = [
            filename
            for filename in GENERATED_FILENAMES
            if filename != "generated-bundle-manifest.json"
        ]
        write_json(
            staging / "generated-bundle-manifest.json",
            {
                "source_root": str(source),
                "source_manifest_sha256": inventory_summary["manifest_sha256"],
                "files": {
                    filename: _sha256(staging / filename) for filename in bundle_files
                },
            },
        )
        _publish_with_rollback(staging, output)
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the complete static Aptus legacy evidence bundle."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    summary = generate_evidence_bundle(arguments.source, arguments.output)
    print(
        f"Generated evidence for {summary['artifact_count']} artifacts; "
        f"manifest {summary['baseline_manifest_sha256']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
