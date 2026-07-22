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
    if runtime_id == "mlx-lm":
        versions = details.get("versions")
        expected = {package: STACK_VERSIONS[package] for package in ("mlx", "mlx-lm")}
        if not isinstance(versions, Mapping) or any(
            versions.get(package) != version for package, version in expected.items()
        ):
            found = {
                package: versions.get(package)
                if isinstance(versions, Mapping)
                else None
                for package in expected
            }
            raise RuntimeError(
                "The selected MLX interpreter has incompatible dependency versions: "
                f"expected {expected}, found {found}."
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
        path = Path(raw).expanduser()
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        if (
            resolved in seen
            or not resolved.is_file()
            or not os.access(resolved, os.X_OK)
        ):
            continue
        seen.add(resolved)
        result.append((resolved, source))
    return tuple(result)


def probe_runtime_interpreter(
    path: Path,
    *,
    source: str,
    timeout: float = 8.0,
) -> RuntimeInterpreter:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"Runtime interpreter is not executable: {resolved}")
    try:
        completed = subprocess.run(
            [str(resolved), "-c", _PROBE_PROGRAM],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return RuntimeInterpreter(
            path=str(resolved),
            source=source,
            python_version=None,
            runtimes={},
            error=f"Runtime probe failed: {error}",
        )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        return RuntimeInterpreter(
            path=str(resolved),
            source=source,
            python_version=None,
            runtimes={},
            error=f"Runtime probe exited {completed.returncode}: {detail[-500:]}",
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return RuntimeInterpreter(
            path=str(resolved),
            source=source,
            python_version=None,
            runtimes={},
            error="Runtime probe returned invalid JSON.",
        )
    runtimes = payload.get("runtimes") if isinstance(payload, dict) else None
    if not isinstance(runtimes, dict):
        return RuntimeInterpreter(
            path=str(resolved),
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
        path=str(resolved),
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
            configured_path = Path(configured).expanduser().resolve(strict=True)
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
        return configured_probe

    available = next(
        (
            item
            for item in values
            if item.error is None
            and item.runtimes.get(runtime_id, {}).get("available") is True
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
    for interpreter in values:
        for runtime_id in TRAINING_RUNTIMES:
            if interpreter.runtimes.get(runtime_id, {}).get("available") is True:
                available[runtime_id].append(interpreter.path)
    return {
        "schema_version": "aptus.runtime-inventory.v1",
        "interpreters": [item.to_dict() for item in values],
        "available": available,
        "configuration": dict(_RUNTIME_ENVIRONMENT_KEYS),
    }
