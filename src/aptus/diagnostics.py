from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping

from . import __version__
from .catalog import STACK_VERSIONS
from .domain import to_primitive
from .local_store import utc_now
from .profiling import probe_apple_platform
from .runtime_env import TRAINING_RUNTIMES, runtime_environment_key, runtime_inventory


DIAGNOSTIC_SCHEMA_VERSION = "aptus.diagnostics.v1"
DOCTOR_SCHEMA_VERSION = "aptus.environment-doctor.v1"


_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:[^\s,:;\)\]\}\"']+/?)+")


def _path_fingerprint(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"[absolute-path:{digest}]"


def _redact_home(value: str) -> str:
    home = str(Path.home())
    if value == home:
        return "$HOME"
    if value.startswith(home + os.sep):
        return "$HOME" + value[len(home) :]
    if os.path.isabs(value):
        return _path_fingerprint(value)
    return _ABSOLUTE_PATH.sub(lambda match: _path_fingerprint(match.group(0)), value)


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_home(value)
    if isinstance(value, Mapping):
        return {_redact_home(str(key)): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _safe_json_object(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _diagnostic_runtime_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Keep compatibility evidence while removing interpreter path values."""

    interpreters: list[dict[str, Any]] = []
    raw_interpreters = inventory.get("interpreters")
    if isinstance(raw_interpreters, list):
        for raw in raw_interpreters:
            if not isinstance(raw, Mapping):
                continue
            path = raw.get("path")
            raw_runtimes = raw.get("runtimes")
            runtimes: dict[str, dict[str, Any]] = {}
            if isinstance(raw_runtimes, Mapping):
                for runtime_id, raw_details in raw_runtimes.items():
                    if not isinstance(raw_details, Mapping):
                        continue
                    runtimes[str(runtime_id)] = {
                        key: raw_details[key]
                        for key in (
                            "available",
                            "compatible",
                            "versions",
                            "expected_versions",
                            "mps_built",
                            "mps_available",
                            "cuda_available",
                        )
                        if key in raw_details
                    }
                    runtimes[str(runtime_id)]["reason_present"] = any(
                        isinstance(raw_details.get(key), str)
                        and bool(str(raw_details[key]).strip())
                        for key in ("reason", "compatibility_reason")
                    )
            interpreters.append(
                {
                    "path_fingerprint": (
                        hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
                        if isinstance(path, str)
                        else None
                    ),
                    "source": raw.get("source"),
                    "python_version": raw.get("python_version"),
                    "runtimes": runtimes,
                    "probe_error_present": isinstance(raw.get("error"), str)
                    and bool(str(raw["error"]).strip()),
                }
            )

    def counts(field: str) -> dict[str, int]:
        raw = inventory.get(field)
        if not isinstance(raw, Mapping):
            return {}
        return {
            str(runtime_id): len(paths) if isinstance(paths, list) else 0
            for runtime_id, paths in raw.items()
        }

    return {
        "schema_version": inventory.get("schema_version"),
        "interpreters": interpreters,
        "available_counts": counts("available"),
        "compatible_counts": counts("compatible"),
        "configuration": inventory.get("configuration", {}),
        "paths_included": False,
    }


def _state_summary(state_dir: Path) -> dict[str, Any]:
    target = state_dir.expanduser().absolute()
    result: dict[str, Any] = {
        "path": _redact_home(str(target)),
        "exists": target.exists(),
        "is_symlink": target.is_symlink(),
        "private_permissions": None,
        "jobs": {"records": 0, "unreadable": 0, "by_state": {}, "by_action": {}},
        "projects": {"records": 0, "revisions": 0},
        "quarantine_records": 0,
    }
    if target.is_symlink() or not target.exists() or not target.is_dir():
        return result

    try:
        metadata = target.stat()
        result["private_permissions"] = (
            not hasattr(os, "getuid") or metadata.st_uid == os.getuid()
        ) and (os.name != "posix" or stat.S_IMODE(metadata.st_mode) == stat.S_IRWXU)
    except OSError:
        return result

    jobs = result["jobs"]
    assert isinstance(jobs, dict)
    jobs_root = target / "jobs"
    if jobs_root.is_dir() and not jobs_root.is_symlink():
        for path in sorted(jobs_root.glob("*.json")):
            value = _safe_json_object(path)
            if value is None:
                jobs["unreadable"] += 1
                continue
            jobs["records"] += 1
            for field, bucket in (("state", "by_state"), ("action", "by_action")):
                item = value.get(field)
                if isinstance(item, str) and item:
                    counts = jobs[bucket]
                    counts[item] = counts.get(item, 0) + 1

    projects_root = target / "projects"
    projects = result["projects"]
    assert isinstance(projects, dict)
    if projects_root.is_dir() and not projects_root.is_symlink():
        for project_root in sorted(projects_root.glob("project_*")):
            if project_root.is_symlink() or not project_root.is_dir():
                continue
            if _safe_json_object(project_root / "project.json") is not None:
                projects["records"] += 1
            revisions = project_root / "revisions"
            if revisions.is_dir() and not revisions.is_symlink():
                projects["revisions"] += sum(
                    1
                    for item in revisions.glob("revision_*.json")
                    if item.is_file() and not item.is_symlink()
                )

    quarantine_root = target / "quarantine"
    if quarantine_root.is_dir() and not quarantine_root.is_symlink():
        result["quarantine_records"] = sum(
            1
            for item in quarantine_root.rglob("*")
            if item.is_file() and not item.is_symlink()
        )
    return result


def _disk_summary(state_dir: Path) -> dict[str, int] | dict[str, str]:
    target = state_dir.expanduser().absolute()
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        usage = shutil.disk_usage(target)
    except OSError as error:
        return {"error": type(error).__name__}
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def build_doctor_report(
    state_dir: Path,
    *,
    inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    measured_inventory = (
        dict(inventory) if inventory is not None else runtime_inventory()
    )
    compatible_value = measured_inventory.get("compatible", {})
    compatible = compatible_value if isinstance(compatible_value, Mapping) else {}
    preferred_runtime = (
        "mlx-lm"
        if platform.system() == "Darwin" and platform.machine().lower() == "arm64"
        else None
    )
    if preferred_runtime is not None:
        preferred = compatible.get(preferred_runtime)
        ready = isinstance(preferred, list) and bool(preferred)
    else:
        ready = any(
            isinstance(compatible.get(runtime_id), list)
            and bool(compatible[runtime_id])
            for runtime_id in TRAINING_RUNTIMES
        )
    configured = {
        runtime_id: bool(
            os.environ.get(runtime_environment_key(runtime_id), "").strip()
        )
        for runtime_id in TRAINING_RUNTIMES
    }
    next_steps: list[str] = []
    if preferred_runtime == "mlx-lm" and not ready:
        next_steps = [
            "Create a Python 3.12 virtual environment outside any compiled bundle.",
            (
                "Install the reviewed direct pins: "
                f"mlx=={STACK_VERSIONS['mlx']} and mlx-lm=={STACK_VERSIONS['mlx-lm']}."
            ),
            "Run this doctor again, then explicitly choose the passing interpreter in Aptus.",
        ]
    elif not ready:
        next_steps = [
            "Create an isolated runtime outside any compiled bundle.",
            "Install the exact direct requirements for the selected Aptus candidate.",
            "Run this doctor again and explicitly select a passing interpreter.",
        ]
    return _sanitize(
        {
            "schema_version": DOCTOR_SCHEMA_VERSION,
            "aptus_version": __version__,
            "observed_at": utc_now(),
            "status": "ready" if ready else "action-required",
            "preferred_runtime": preferred_runtime,
            "configured_runtime_keys": configured,
            "runtime_inventory": _diagnostic_runtime_inventory(measured_inventory),
            "state": _state_summary(state_dir),
            "next_steps": next_steps,
            "installation_performed": False,
        }
    )


def build_diagnostics(
    state_dir: Path,
    *,
    inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    doctor = build_doctor_report(state_dir, inventory=inventory)
    try:
        apple_platform: dict[str, Any] | None = to_primitive(probe_apple_platform())
        apple_platform_error = None
    except (OSError, RuntimeError, ValueError) as error:
        apple_platform = None
        apple_platform_error = type(error).__name__
    return _sanitize(
        {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "collected_at": utc_now(),
            "aptus_version": __version__,
            "host": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "frozen": bool(getattr(sys, "frozen", False)),
                "logical_cpu_count": os.cpu_count(),
            },
            "apple_platform": apple_platform,
            "apple_platform_error": apple_platform_error,
            "disk": _disk_summary(state_dir),
            "doctor": doctor,
            "privacy": {
                "logs_included": False,
                "dataset_or_model_content_included": False,
                "project_names_included": False,
                "home_path_redacted": True,
                "environment_values_included": False,
            },
        }
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info


def create_diagnostic_archive(
    state_dir: Path,
    output: Path,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> Path:
    target = output.expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Diagnostic archive already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        dict(diagnostics) if diagnostics is not None else build_diagnostics(state_dir)
    )
    diagnostic_bytes = _json_bytes(payload)
    readme_bytes = (
        "Aptus diagnostic bundle\n\n"
        "This archive contains bounded host, runtime, and local-state counts. "
        "It excludes logs, dataset and model content, project names, environment "
        "values, and unredacted home paths. Review diagnostics.json before sharing.\n"
    ).encode("utf-8")
    manifest = {
        "schema_version": "aptus.diagnostic-archive.v1",
        "files": {
            "README.txt": hashlib.sha256(readme_bytes).hexdigest(),
            "diagnostics.json": hashlib.sha256(diagnostic_bytes).hexdigest(),
        },
    }
    descriptor: int | None = None
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w+b") as stream:
            descriptor = None
            with zipfile.ZipFile(stream, mode="w") as archive:
                archive.writestr(_zip_info("README.txt"), readme_bytes)
                archive.writestr(_zip_info("diagnostics.json"), diagnostic_bytes)
                archive.writestr(_zip_info("manifest.json"), _json_bytes(manifest))
        target.chmod(0o600)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        target.unlink(missing_ok=True)
        raise
    return target
