from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .inventory import (
    build_duplicate_clusters,
    build_version_families,
    inventory_tree,
    write_json,
    write_jsonl,
)


def _manifest_digest(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update((record["sha256"] or "").encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["size_bytes"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def generate_inventory_bundle(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(f"Legacy source is not a directory: {source}")
    output = output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Audit output must be outside the legacy source tree.")
    records = inventory_tree(source)
    clusters = build_duplicate_clusters(records)
    version_families = build_version_families(records)

    write_jsonl(output / "inventory.jsonl", records)
    write_json(
        output / "duplicate-clusters.json",
        {
            "cluster_count": len(clusters),
            "clusters": clusters,
        },
    )
    write_json(
        output / "version-families.json",
        {
            "family_count": len(version_families),
            "families": version_families,
        },
    )

    duplicate_file_count = sum(len(cluster["paths"]) for cluster in clusters)
    summary = {
        "source_root": str(source),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "manifest_sha256": _manifest_digest(records),
        "duplicate_cluster_count": len(clusters),
        "duplicate_file_count": duplicate_file_count,
        "version_family_count": len(version_families),
        "kind_counts": {
            kind: sum(record["kind"] == kind for record in records)
            for kind in ("text", "binary", "empty", "symlink")
        },
    }
    write_json(output / "baseline-manifest.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory the legacy source tree for the Aptus forensic audit."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    summary = generate_inventory_bundle(arguments.source, arguments.output)
    print(
        "Inventoried "
        f"{summary['file_count']} files; "
        f"manifest {summary['manifest_sha256']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
