from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows only.
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX only.
    msvcrt = None


LEASE_ENV = "APTUS_GPU_LEASE_TOKEN"
_THREAD_LOCK = threading.RLock()
_WORLD_WRITABLE_TMP = frozenset(
    {
        Path("/tmp"),
        Path("/private/tmp"),
        Path("/var/tmp"),
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_secure_runtime_dir(path: Path) -> bool:
    if not path.is_absolute():
        return False
    try:
        stat_result = path.lstat()
    except OSError:
        return False
    if path.is_symlink() or not path.is_dir():
        return False
    if hasattr(os, "getuid") and stat_result.st_uid != os.getuid():
        return False
    if os.name == "posix" and stat_result.st_mode & 0o077:
        return False
    try:
        resolved = path.resolve()
    except OSError:
        return False
    forbidden = {candidate.resolve() for candidate in _WORLD_WRITABLE_TMP}
    return resolved not in forbidden and forbidden.isdisjoint(resolved.parents)


def default_lease_parent() -> Path:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if runtime_dir:
        candidate = Path(runtime_dir)
        if _is_secure_runtime_dir(candidate):
            return candidate / "aptus"
    if os.name == "posix":
        return Path.home() / ".aptus" / "run"
    return Path(tempfile.gettempdir()) / "aptus-run"


def default_lease_root(parent: Path | None = None) -> Path:
    identity = (
        str(os.getuid())
        if hasattr(os, "getuid")
        else os.environ.get("USERNAME", "default")
    )
    safe_identity = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return (parent or default_lease_parent()) / f"aptus-gpu-lease-{safe_identity}"


def _lease_paths(parent: Path | None = None) -> tuple[Path, Path, Path]:
    root = default_lease_root(parent)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root_stat = root.lstat()
    if root.is_symlink() or not root.is_dir():
        raise PermissionError(f"Aptus lease root is not a secure directory: {root}")
    if hasattr(os, "getuid") and root_stat.st_uid != os.getuid():
        raise PermissionError(f"Aptus lease root is owned by another user: {root}")
    if os.name == "posix" and root_stat.st_mode & 0o077:
        root.chmod(0o700)
    return root, root / "lease.json", root / ".lease.lock"


@contextmanager
def _lease_lock(parent: Path | None = None) -> Iterator[tuple[Path, Path]]:
    root, lease_path, lock_path = _lease_paths(parent)
    with _THREAD_LOCK, lock_path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - Windows only.
            lock_file.seek(0)
            if not lock_file.read(1):
                lock_file.write("\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield root, lease_path
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows only.
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _process_identity(value: Any) -> str | None:
    if not isinstance(value, int) or value <= 0:
        return None
    if sys.platform.startswith("linux"):
        try:
            fields = (
                Path(f"/proc/{value}/stat")
                .read_text(encoding="utf-8")
                .rsplit(")", 1)[1]
                .split()
            )
        except (OSError, IndexError):
            pass
        else:
            if len(fields) > 19:
                return f"linux-start-ticks:{fields[19]}"
    if os.name == "posix":
        try:
            completed = subprocess.run(
                ["ps", "-o", "lstart=", "-p", str(value)],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        started = completed.stdout.strip()
        return f"{sys.platform}-started:{started}" if started else None
    if os.name == "nt":  # pragma: no cover - Windows only.
        try:
            import ctypes
            from ctypes import wintypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, value)
            if not handle:
                return None
            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            try:
                succeeded = ctypes.windll.kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                )
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
            if succeeded:
                ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
                return f"windows-created:{ticks}"
        except (AttributeError, OSError):
            return None
    return None


def _pid_alive(value: Any, expected_identity: Any = None) -> bool:
    if not isinstance(value, int) or value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if isinstance(expected_identity, str):
        actual_identity = _process_identity(value)
        if actual_identity is not None and actual_identity != expected_identity:
            return False
    return True


def _process_group_alive(value: Any) -> bool:
    if os.name != "posix" or not isinstance(value, int) or value <= 0:
        return False
    try:
        os.killpg(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_posix_group(
    process_group_id: int, leader: subprocess.Popen[Any] | None = None
) -> None:
    if not _process_group_alive(process_group_id):
        return
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5
    while _process_group_alive(process_group_id) and time.monotonic() < deadline:
        if leader is not None:
            leader.poll()
        time.sleep(0.05)
    if _process_group_alive(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 2
        while _process_group_alive(process_group_id) and time.monotonic() < deadline:
            if leader is not None:
                leader.poll()
            time.sleep(0.05)
    if _process_group_alive(process_group_id):
        raise RuntimeError(
            f"Portable Aptus process group {process_group_id} remained live after SIGKILL."
        )


def _lease_live(value: dict[str, Any]) -> bool:
    return (
        _pid_alive(value.get("owner_pid"), value.get("owner_process_identity"))
        or _pid_alive(value.get("process_pid"), value.get("process_identity"))
        or _process_group_alive(value.get("process_group_id"))
    )


def _read_lease(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"Aptus lease path is unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Aptus GPU lease is unreadable: {error}") from error
    required = {"job_id", "state_root", "owner_pid", "created_at", "lease_token"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError("Aptus GPU lease has an invalid contract.")
    return value


def _write_lease(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _active_error(value: dict[str, Any]) -> RuntimeError:
    return RuntimeError(
        "Aptus already has an active GPU action "
        f"{value.get('job_id')} from {value.get('state_root')}. "
        "Only one Aptus GPU action may run for this local user and host."
    )


@contextmanager
def portable_execution_lease(
    bundle_dir: Path,
    *,
    action: str,
    _lease_parent: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Create or borrow the JobService-compatible per-user host lease."""

    bundle = bundle_dir.resolve()
    inherited_token = os.environ.get(LEASE_ENV)
    created = False
    borrowed_portable = False
    previous_process: tuple[Any, Any, Any] | None = None
    token: str
    with _lease_lock(_lease_parent) as (_root, lease_path):
        existing = _read_lease(lease_path)
        if (
            existing is not None
            and inherited_token
            and existing.get("lease_token") == inherited_token
            and _lease_live(existing)
        ):
            token = inherited_token
            if existing.get("owner_kind") == "portable":
                previous_process = (
                    existing.get("process_pid"),
                    existing.get("process_identity"),
                    existing.get("process_group_id"),
                )
                existing.update(
                    process_pid=os.getpid(),
                    process_identity=_process_identity(os.getpid()),
                    process_group_id=None,
                    borrowed_at=_now(),
                )
                _write_lease(lease_path, existing)
                borrowed_portable = True
        else:
            if existing is not None:
                if _lease_live(existing):
                    raise _active_error(existing)
                lease_path.unlink(missing_ok=True)
            token = "portable_" + uuid.uuid4().hex
            identity = _process_identity(os.getpid())
            existing = {
                "schema_version": "aptus.gpu-lease.v1",
                "job_id": token,
                "lease_token": token,
                "state_root": str(bundle),
                "bundle_dir": str(bundle),
                "action": action,
                "owner_kind": "portable",
                "owner_pid": os.getpid(),
                "owner_process_identity": identity,
                "process_pid": os.getpid(),
                "process_identity": identity,
                "process_group_id": None,
                "created_at": _now(),
            }
            _write_lease(lease_path, existing)
            os.environ[LEASE_ENV] = token
            created = True
    try:
        yield existing
    finally:
        with _lease_lock(_lease_parent) as (_root, lease_path):
            current = _read_lease(lease_path)
            if current is not None and current.get("lease_token") == token:
                if borrowed_portable and previous_process is not None:
                    current.update(
                        process_pid=previous_process[0],
                        process_identity=previous_process[1],
                        process_group_id=previous_process[2],
                    )
                    _write_lease(lease_path, current)
                elif created:
                    child_pid = current.get("process_pid")
                    child_live = child_pid != os.getpid() and _pid_alive(
                        child_pid, current.get("process_identity")
                    )
                    child_live = child_live or _process_group_alive(
                        current.get("process_group_id")
                    )
                    if not child_live:
                        lease_path.unlink(missing_ok=True)
        if created:
            if inherited_token is None:
                os.environ.pop(LEASE_ENV, None)
            else:
                os.environ[LEASE_ENV] = inherited_token


def require_execution_lease(*, _lease_parent: Path | None = None) -> dict[str, Any]:
    token = os.environ.get(LEASE_ENV)
    if not token:
        raise RuntimeError(
            "Direct train.py execution is disabled. Use validate.py or run.py so Aptus can hold the host-global GPU lease."
        )
    with _lease_lock(_lease_parent) as (_root, lease_path):
        value = _read_lease(lease_path)
        if value is None or value.get("lease_token") != token or not _lease_live(value):
            raise RuntimeError(
                "The Aptus GPU lease token is absent, stale, or changed."
            )
        return value


def run_with_lease(
    command: Sequence[str],
    *,
    cwd: Path,
    _lease_parent: Path | None = None,
) -> subprocess.CompletedProcess[None]:
    """Run one child while binding portable crash recovery to its PID."""

    lease = require_execution_lease(_lease_parent=_lease_parent)
    token = os.environ[LEASE_ENV]
    portable = lease.get("owner_kind") == "portable"
    if portable and os.name != "posix":  # pragma: no cover - Windows only.
        raise RuntimeError(
            "Direct portable execution is fail-closed on Windows in Aptus v0.2. Use the managed JobService path."
        )
    process = subprocess.Popen(
        list(command), cwd=cwd, start_new_session=portable and os.name == "posix"
    )
    previous_process: tuple[Any, Any, Any] | None = None
    try:
        if portable:
            with _lease_lock(_lease_parent) as (_root, lease_path):
                current = _read_lease(lease_path)
                if current is None or current.get("lease_token") != token:
                    raise RuntimeError(
                        "The Aptus GPU lease changed during child launch."
                    )
                previous_process = (
                    current.get("process_pid"),
                    current.get("process_identity"),
                    current.get("process_group_id"),
                )
                current.update(
                    process_pid=process.pid,
                    process_identity=_process_identity(process.pid),
                    process_group_id=process.pid,
                    started_at=_now(),
                )
                _write_lease(lease_path, current)
        return_code = process.wait()
        if portable and _process_group_alive(process.pid):
            _terminate_posix_group(process.pid, process)
            raise RuntimeError(
                "The portable Aptus launcher exited while descendants remained live."
            )
    except BaseException:
        if portable:
            _terminate_posix_group(process.pid, process)
            if process.poll() is None:
                process.wait(timeout=2)
        elif process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        raise
    finally:
        if portable and process.poll() is not None:
            with _lease_lock(_lease_parent) as (_root, lease_path):
                current = _read_lease(lease_path)
                if current is not None and current.get("lease_token") == token:
                    restored = previous_process or (
                        current.get("owner_pid"),
                        current.get("owner_process_identity"),
                        None,
                    )
                    current.update(
                        process_pid=restored[0],
                        process_identity=restored[1],
                        process_group_id=restored[2],
                    )
                    _write_lease(lease_path, current)
    return subprocess.CompletedProcess(list(command), return_code)
