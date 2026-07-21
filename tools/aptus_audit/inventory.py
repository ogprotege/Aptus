from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import stat
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_kind(path: Path, size_bytes: int) -> str:
    if size_bytes == 0:
        return "empty"

    sample = path.read_bytes()[:8192]
    if b"\x00" in sample:
        return "binary"

    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return "binary"
    return "text"


def inventory_tree(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    records: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if not path.is_symlink() and not path.is_file():
            continue

        relative_path = path.relative_to(root).as_posix()
        metadata = path.lstat()
        symlink_target = os.readlink(path) if path.is_symlink() else None
        size_bytes = metadata.st_size
        kind = "symlink" if path.is_symlink() else _file_kind(path, size_bytes)
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

        records.append(
            {
                "path": relative_path,
                "absolute_path": str(path),
                "name": path.name,
                "extension": "".join(path.suffixes),
                "size_bytes": size_bytes,
                "mode": stat.filemode(metadata.st_mode),
                "modified_ns": metadata.st_mtime_ns,
                "kind": kind,
                "mime_type": mime_type,
                "sha256": None if path.is_symlink() else _sha256(path),
                "symlink_target": symlink_target,
            }
        )

    return records


def build_duplicate_clusters(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("sha256"):
            grouped[record["sha256"]].append(record)

    clusters = []
    for digest, members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        paths = sorted(member["path"] for member in members)
        clusters.append(
            {
                "cluster_id": f"sha256:{digest}",
                "sha256": digest,
                "size_bytes": members[0]["size_bytes"],
                "paths": paths,
            }
        )
    return clusters


def _normalized_version_path(path: str) -> str:
    normalized_parts = []
    for component in path.split("/"):
        component_path = Path(component)
        suffix = "".join(component_path.suffixes)
        stem = component[: -len(suffix)] if suffix else component
        stem = re.sub(r" 2$", "", stem)
        stem = re.sub(r"[_-]v?2$", "", stem, flags=re.IGNORECASE)
        normalized_parts.append(f"{stem}{suffix}")
    return "/".join(normalized_parts)


def build_version_families(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        path = record["path"]
        grouped[_normalized_version_path(path)].append(path)

    families = []
    for normalized_path, paths in sorted(grouped.items()):
        if len(paths) < 2:
            continue
        families.append(
            {
                "family_id": f"normalized:{normalized_path}",
                "normalized_path": normalized_path,
                "paths": sorted(paths),
            }
        )
    return families


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True))
            destination.write("\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
