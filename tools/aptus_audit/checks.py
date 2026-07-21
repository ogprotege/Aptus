from __future__ import annotations

import hashlib
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _preview(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[output truncated by Aptus audit runner]\n"


def run_check(
    *,
    check_id: str,
    command: Sequence[str],
    cwd: Path,
    timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
    inherit_proxy: bool = False,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    started_clock = time.monotonic()
    allowed_keys = {"PATH", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT"}
    if inherit_proxy:
        allowed_keys.update(
            {
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "NO_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
                "no_proxy",
            }
        )
    inherited_environment_keys = sorted(
        key for key in os.environ if key in allowed_keys
    )
    allowed_environment = {
        key: value for key, value in os.environ.items() if key in allowed_keys
    }
    if environment:
        allowed_environment.update(environment)

    stdout = ""
    stderr = ""
    exit_code: int | None = None
    timed_out = False

    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=allowed_environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout or ""
        stderr = error.stderr or ""
    except OSError as error:
        stderr = f"{type(error).__name__}: {error}"

    stdout = _as_text(stdout)
    stderr = _as_text(stderr)
    duration_ms = round((time.monotonic() - started_clock) * 1000)
    if timed_out:
        status = "timed_out"
    elif exit_code == 0:
        status = "passed"
    else:
        status = "failed"

    return {
        "check_id": check_id,
        "safety_class": "sandboxed-dynamic",
        "command": list(command),
        "cwd": str(cwd.resolve()),
        "started_at_utc": started_at.isoformat(),
        "duration_ms": duration_ms,
        "timeout_seconds": timeout_seconds,
        "inherited_environment_keys": inherited_environment_keys,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "status": status,
        "stdout_sha256": _digest(stdout),
        "stderr_sha256": _digest(stderr),
        "stdout_preview": _preview(stdout),
        "stderr_preview": _preview(stderr),
    }
