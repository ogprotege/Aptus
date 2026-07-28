from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .catalog import STACK_VERSIONS


TRAINING_RUNTIMES = (
    "mlx-lm",
    "pytorch-mps",
    "transformers-peft-cuda",
)

_RUNTIME_ENVIRONMENT_KEYS = {
    "mlx-lm": "APTUS_MLX_PYTHON",
    "pytorch-mps": "APTUS_PYTORCH_PYTHON",
    "transformers-peft-cuda": "APTUS_CUDA_PYTHON",
}


def runtime_compatibility(
    runtime_id: str,
    details: Mapping[str, object],
) -> tuple[bool, str | None]:
    """Return whether an imported runtime satisfies Aptus' executable contract."""

    if details.get("available") is not True:
        reason = details.get("reason")
        return False, str(reason) if isinstance(reason, str) and reason else None
    if runtime_id != "mlx-lm":
        return True, None
    versions = details.get("versions")
    expected = {package: STACK_VERSIONS[package] for package in ("mlx", "mlx-lm")}
    found = (
        {package: versions.get(package) for package in expected}
        if isinstance(versions, Mapping)
        else {package: None for package in expected}
    )
    if found == expected:
        return True, None
    return (
        False,
        "The MLX-LM imports passed, but the dependency contract is incompatible: "
        f"expected {expected}, found {found}.",
    )


def runtime_environment_key(runtime_id: str) -> str:
    try:
        return _RUNTIME_ENVIRONMENT_KEYS[runtime_id]
    except KeyError as error:
        raise ValueError(f"Unknown Aptus training runtime: {runtime_id}") from error


def validate_runtime_configuration(
    runtime_id: str,
    interpreter_path: Path,
    *,
    timeout: float = 8.0,
) -> RuntimeInterpreter:
    key = runtime_environment_key(runtime_id)
    probe = probe_runtime_interpreter(
        interpreter_path,
        source=f"configured:{key}",
        timeout=timeout,
    )
    details = probe.runtimes.get(runtime_id, {})
    if probe.error is not None or details.get("available") is not True:
        reason = probe.error or details.get("reason") or "runtime probe did not pass"
        raise RuntimeError(
            f"The selected interpreter is not ready for {runtime_id}: {reason}"
        )
    compatible, compatibility_reason = runtime_compatibility(runtime_id, details)
    if not compatible:
        raise RuntimeError(
            "The selected interpreter has an incompatible runtime contract: "
            f"{compatibility_reason or 'the exact dependency requirements did not pass.'}"
        )
    return probe


_PROBE_PROGRAM = r"""
import importlib.metadata
import json
import platform
import sys

def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

result = {
    "python_version": platform.python_version(),
    "runtimes": {},
}

try:
    import mlx
    import mlx_lm
    result["runtimes"]["mlx-lm"] = {
        "available": True,
        "versions": {
            "mlx": distribution_version("mlx"),
            "mlx-lm": distribution_version("mlx-lm"),
        },
    }
except Exception as error:
    result["runtimes"]["mlx-lm"] = {
        "available": False,
        "reason": f"{type(error).__name__}: {error}",
    }

try:
    import torch
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    result["runtimes"]["pytorch-mps"] = {
        "available": mps_built and mps_available,
        "versions": {"torch": distribution_version("torch")},
        "mps_built": mps_built,
        "mps_available": mps_available,
        "reason": None if mps_built and mps_available else "PyTorch MPS is not built and available in this interpreter.",
    }
    cuda_available = bool(torch.cuda.is_available())
    result["runtimes"]["transformers-peft-cuda"] = {
        "available": cuda_available,
        "versions": {
            "torch": distribution_version("torch"),
            "transformers": distribution_version("transformers"),
            "peft": distribution_version("peft"),
            "accelerate": distribution_version("accelerate"),
        },
        "cuda_available": cuda_available,
        "reason": None if cuda_available else "PyTorch CUDA is not available in this interpreter.",
    }
except Exception as error:
    detail = f"{type(error).__name__}: {error}"
    result["runtimes"]["pytorch-mps"] = {"available": False, "reason": detail}
    result["runtimes"]["transformers-peft-cuda"] = {"available": False, "reason": detail}

print(json.dumps(result, sort_keys=True))
"""


@dataclass(frozen=True)
class RuntimeInterpreter:
    path: str
    source: str
    python_version: str | None
    runtimes: Mapping[str, Mapping[str, object]]
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _candidate_interpreters(
    environment: Mapping[str, str],
    explicit: Iterable[tuple[str, str]] = (),
) -> tuple[tuple[Path, str], ...]:
    candidates: list[tuple[str, str]] = list(explicit)
    for runtime_id, key in _RUNTIME_ENVIRONMENT_KEYS.items():
        value = environment.get(key, "").strip()
        if value:
            candidates.append((value, f"environment:{key}:{runtime_id}"))

    if not getattr(sys, "frozen", False):
        candidates.append((sys.executable, "current-process"))
    discovered = shutil.which("python3")
    if discovered:
        candidates.append((discovered, "path:python3"))
    candidates.extend(
        (
            ("/opt/homebrew/bin/python3", "known-path:homebrew"),
            ("/usr/local/bin/python3", "known-path:usr-local"),
        )
    )

    result: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for raw, source in candidates:
        path = Path(os.path.abspath(Path(raw).expanduser()))
        try:
            exists = path.exists()
        except OSError:
            continue
        if (
            not exists
            or path in seen
            or not path.is_file()
            or not os.access(path, os.X_OK)
        ):
            continue
        # Keep the selected command path. A virtual environment commonly exposes
        # ``bin/python`` as a symlink to its base interpreter. Resolving that
        # symlink discards the virtual environment's package context when the
        # recorded path is launched later.
        seen.add(path)
        result.append((path, source))
    return tuple(result)


def probe_runtime_interpreter(
    path: Path,
    *,
    source: str,
    timeout: float = 8.0,
) -> RuntimeInterpreter:
    selected = Path(os.path.abspath(path.expanduser()))
    if (
        not selected.exists()
        or not selected.is_file()
        or not os.access(selected, os.X_OK)
    ):
        raise ValueError(f"Runtime interpreter is not executable: {selected}")
    try:
        completed = subprocess.run(
            [str(selected), "-c", _PROBE_PROGRAM],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return RuntimeInterpreter(
            path=str(selected),
            source=source,
            python_version=None,
            runtimes={},
            error=f"Runtime probe failed: {error}",
        )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        return RuntimeInterpreter(
            path=str(selected),
            source=source,
            python_version=None,
            runtimes={},
            error=f"Runtime probe exited {completed.returncode}: {detail[-500:]}",
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return RuntimeInterpreter(
            path=str(selected),
            source=source,
            python_version=None,
            runtimes={},
            error="Runtime probe returned invalid JSON.",
        )
    runtimes = payload.get("runtimes") if isinstance(payload, dict) else None
    if not isinstance(runtimes, dict):
        return RuntimeInterpreter(
            path=str(selected),
            source=source,
            python_version=None,
            runtimes={},
            error="Runtime probe returned an invalid contract.",
        )
    normalized = {
        runtime_id: details
        for runtime_id, details in runtimes.items()
        if runtime_id in TRAINING_RUNTIMES and isinstance(details, dict)
    }
    return RuntimeInterpreter(
        path=str(selected),
        source=source,
        python_version=(
            payload.get("python_version")
            if isinstance(payload.get("python_version"), str)
            else None
        ),
        runtimes=normalized,
    )


def discover_runtime_interpreters(
    *,
    environment: Mapping[str, str] | None = None,
    explicit: Iterable[tuple[str, str]] = (),
    timeout: float = 8.0,
) -> tuple[RuntimeInterpreter, ...]:
    active_environment = os.environ if environment is None else environment
    return tuple(
        probe_runtime_interpreter(path, source=source, timeout=timeout)
        for path, source in _candidate_interpreters(active_environment, explicit)
    )


def resolve_runtime_interpreter(
    runtime_id: str,
    *,
    interpreters: Iterable[RuntimeInterpreter] | None = None,
    environment: Mapping[str, str] | None = None,
) -> RuntimeInterpreter:
    if runtime_id not in TRAINING_RUNTIMES:
        raise ValueError(f"Unknown Aptus training runtime: {runtime_id}")
    active_environment = os.environ if environment is None else environment
    values = tuple(
        interpreters
        if interpreters is not None
        else discover_runtime_interpreters(environment=active_environment)
    )
    key = _RUNTIME_ENVIRONMENT_KEYS[runtime_id]
    configured = active_environment.get(key, "").strip()
    if configured:
        try:
            configured_path = Path(os.path.abspath(Path(configured).expanduser()))
            if not configured_path.exists():
                raise FileNotFoundError(configured_path)
        except OSError as error:
            raise RuntimeError(
                f"{key} does not resolve to an interpreter: {error}"
            ) from error
        configured_probe = next(
            (item for item in values if Path(item.path) == configured_path),
            None,
        )
        if configured_probe is None:
            raise RuntimeError(f"{key} was configured but could not be probed.")
        details = configured_probe.runtimes.get(runtime_id, {})
        compatible, reason = runtime_compatibility(runtime_id, details)
        if configured_probe.error is not None or not compatible:
            raise RuntimeError(
                f"{key} is not compatible with the current Aptus runtime contract: "
                f"{configured_probe.error or reason or 'runtime probe did not pass'}"
            )
        return configured_probe

    available = next(
        (
            item
            for item in values
            if item.error is None
            and runtime_compatibility(runtime_id, item.runtimes.get(runtime_id, {}))[0]
        ),
        None,
    )
    if available is not None:
        return available
    raise RuntimeError(
        f"No available interpreter was found for {runtime_id}. "
        f"Set {_RUNTIME_ENVIRONMENT_KEYS[runtime_id]} to the exact Python executable for that runtime."
    )


def runtime_inventory(
    *,
    interpreters: Iterable[RuntimeInterpreter] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    values = tuple(
        interpreters
        if interpreters is not None
        else discover_runtime_interpreters(environment=environment)
    )
    available: dict[str, list[str]] = {
        runtime_id: [] for runtime_id in TRAINING_RUNTIMES
    }
    compatible: dict[str, list[str]] = {
        runtime_id: [] for runtime_id in TRAINING_RUNTIMES
    }
    serialized_interpreters: list[dict[str, object]] = []
    for interpreter in values:
        serialized = interpreter.to_dict()
        serialized_runtimes: dict[str, dict[str, object]] = {}
        for runtime_id in TRAINING_RUNTIMES:
            raw_details = interpreter.runtimes.get(runtime_id)
            if raw_details is None:
                continue
            details = dict(raw_details)
            is_compatible, compatibility_reason = runtime_compatibility(
                runtime_id, details
            )
            details["compatible"] = is_compatible
            if runtime_id == "mlx-lm":
                details["expected_versions"] = {
                    package: STACK_VERSIONS[package] for package in ("mlx", "mlx-lm")
                }
            if compatibility_reason is not None:
                details["compatibility_reason"] = compatibility_reason
            serialized_runtimes[runtime_id] = details
            if details.get("available") is True:
                available[runtime_id].append(interpreter.path)
            if is_compatible:
                compatible[runtime_id].append(interpreter.path)
        serialized["runtimes"] = serialized_runtimes
        serialized_interpreters.append(serialized)
    return {
        "schema_version": "aptus.runtime-inventory.v1",
        "interpreters": serialized_interpreters,
        "available": available,
        "compatible": compatible,
        "configuration": dict(_RUNTIME_ENVIRONMENT_KEYS),
    }
