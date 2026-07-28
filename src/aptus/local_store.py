from __future__ import annotations

import json
import os
import stat
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def private_directory(path: Path) -> Path:
    """Create or verify a user-private local state directory."""

    target = path.expanduser()
    if target.is_symlink():
        raise PermissionError(f"Aptus state directories cannot be symlinks: {target}")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.is_symlink() or not target.is_dir():
        raise PermissionError(f"Aptus state path must be a directory: {target}")
    target.chmod(0o700)
    metadata = target.stat()
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise PermissionError(
            f"Aptus state directory is owned by another user: {target}"
        )
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o700:
        raise PermissionError(f"Aptus state directory must use mode 0700: {target}")
    return target.resolve()


def atomic_write_json(
    path: Path,
    value: Mapping[str, Any] | dict[str, Any],
    *,
    mode: int | None = None,
) -> None:
    """Atomically replace a JSON object without following a target symlink."""

    if path.is_symlink():
        raise PermissionError(f"Aptus will not replace a JSON symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        if mode is not None:
            os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if mode is not None:
            path.chmod(mode)
    finally:
        temporary.unlink(missing_ok=True)


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def quarantine_file(path: Path, quarantine_root: Path, *, reason: str) -> Path:
    """Move an unreadable state record aside and preserve a reason receipt."""

    root = private_directory(quarantine_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = root / f"{timestamp}-{uuid.uuid4().hex[:8]}-{path.name}"
    os.replace(path, destination)
    if not destination.is_symlink():
        destination.chmod(0o600)
    atomic_write_json(
        destination.with_suffix(destination.suffix + ".reason.json"),
        {
            "schema_version": "aptus.quarantine-receipt.v1",
            "original_name": path.name,
            "quarantined_at": utc_now(),
            "reason": reason,
        },
        mode=0o600,
    )
    return destination
