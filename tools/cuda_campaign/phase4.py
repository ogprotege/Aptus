"""Sealed Phase-4 source and idle-baseline authority.

This contract deliberately does not reuse the frozen experiment raw-manifest
``record_kind`` namespace.  Phase 4 produces a small, independently sealed
three-file artifact whose exact source tree, host, telemetry configuration,
and 600-sample idle window must still match when a qualifying harness starts.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .contracts import (
    SCHEMA_VERSIONS,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    compact_canonical_json_bytes,
    sha256_bytes,
    utc_now,
    validate_record,
)
from .monitoring import (
    JOURNAL_BOOT_AUTHORITY_SCHEMA,
    MINIMUM_QUALIFYING_COVERAGE,
    SAMPLE_INTERVAL_SECONDS,
    LinuxNvidiaJournalEventProvider,
    ProbeFailure,
    TrustedExecutable,
    detect_nvidia_thermal_limit_authority,
    resolve_trusted_nvidia_smi,
    validate_idle_baseline,
    validate_telemetry_sample,
)


PHASE4_SOURCE_FREEZE_SCHEMA = "aptus.cuda-campaign-phase4-source-freeze.v1"
PHASE4_SOURCE_FREEZE_SEAL_SCHEMA = "aptus.cuda-campaign-phase4-source-freeze-seal.v1"
PHASE4_SOURCE_FREEZE_NAME = "phase4-source-freeze.json"
PHASE4_IDLE_SAMPLES_NAME = "idle-baseline-samples.jsonl"
PHASE4_SOURCE_FREEZE_SEAL_NAME = "PHASE4-SEALED.json"
PHASE4_BASELINE_SAMPLE_COUNT = 600

_RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?\+00:00$")

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_MACHINE_ID = re.compile(r"^[0-9a-f]{32}$")
_HOST_ID = re.compile(r"^host_[0-9a-f]{32}$")
_PRODUCTION_TOOL = {
    "name": "aptus-cuda-campaign-phase4-freeze",
    "version": "v1",
}
_NONPRODUCTION_TEST_TOOL = {
    "name": "aptus-cuda-campaign-phase4-test-fixture",
    "version": "v1-nonproduction",
}
_SOURCE_FIELDS = frozenset({"commit", "tree"})
_HOST_OBSERVATION_FIELDS = frozenset(
    {
        "gpu_index",
        "gpu_memory_total_bytes",
        "gpu_name",
        "gpu_thermal_limits",
        "gpu_thermal_limits_status",
        "gpu_thermal_limits_support_binding",
        "gpu_uuid_sha256",
        "host_memory_total_bytes",
        "kernel_release",
        "logical_cpu_count",
        "machine_id_sha256",
        "nvidia_driver_version",
        "nvidia_smi_binding_sha256",
    }
)
_FREEZE_FIELDS = frozenset(
    {
        "schema_version",
        "producer",
        "campaign_id",
        "comparison_cohort_id",
        "comparison_cell_id",
        "campaign_sha256",
        "comparison_cohort_sha256",
        "comparison_cell_sha256",
        "source_binding",
        "host_binding_sha256",
        "environment_binding_sha256",
        "model_binding_sha256",
        "dataset_and_split_binding_sha256",
        "method",
        "retention_policy_id",
        "current_host_observation",
        "current_host_binding_sha256",
        "current_boot_authority",
        "telemetry_configuration",
        "telemetry_configuration_sha256",
        "idle_baseline_experiment_run_id",
        "idle_baseline_samples_sha256",
        "idle_baseline_sample_count",
        "idle_baseline_summary",
        "created_at_utc",
    }
)
_JOURNAL_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "boot_id_sha256",
        "journalctl_binding_sha256",
        "initial_cursor_sha256",
        "final_cursor_sha256",
        "initial_projection",
        "final_projection",
    }
)
_CLEAN_JOURNAL_PROJECTION = {
    "xid_errors": [],
    "reset_detected": False,
    "device_lost": False,
    "hardware_error": False,
}
_SEAL_FIELDS = frozenset(
    {
        "schema_version",
        "source_freeze_sha256",
        "source_freeze_size_bytes",
        "idle_baseline_samples_sha256",
        "idle_baseline_samples_size_bytes",
        "sealed_at_utc",
    }
)
_CONFIGURATION_FIELDS = frozenset(
    {
        "configuration_sha256",
        "format_version",
        "lifecycle",
        "profile",
        "provenance",
        "safety_limits",
        "sampling",
        "thermal_policy",
    }
)
_SUPPORT_BINDINGS = frozenset(
    {
        "cpu_temperature",
        "gpu_thermal_limits",
        "hardware_events",
        "nvidia_smi_binary",
        "nvme_temperature",
        "xid_projection",
    }
)


class Phase4SourceFreezeError(ValueError):
    """A Phase-4 source/baseline artifact is absent, mutable, or misbound."""


def _normalized_utc_timestamp(value: Any, label: str) -> str:
    if type(value) is not str or _RFC3339_UTC.fullmatch(value) is None:
        raise Phase4SourceFreezeError(
            f"{label} must be normalized RFC 3339 UTC with +00:00."
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise Phase4SourceFreezeError(
            f"{label} is not a real calendar timestamp."
        ) from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise Phase4SourceFreezeError(f"{label} must use UTC.")
    return value


@dataclass(frozen=True)
class TrustedPhase4Boundary:
    """Facts derived from trusted Git, machine, and NVIDIA boundaries."""

    source_binding: Mapping[str, str]
    host_observation: Mapping[str, Any]
    journal_boot_authority: Mapping[str, Any]
    nvidia_smi: TrustedExecutable | None = None

    def __post_init__(self) -> None:
        source = _validate_source_binding(self.source_binding)
        host = _validate_host_observation(self.host_observation)
        journal = _validate_journal_boot_authority(self.journal_boot_authority)
        object.__setattr__(self, "source_binding", MappingProxyType(source))
        object.__setattr__(self, "host_observation", MappingProxyType(host))
        object.__setattr__(self, "journal_boot_authority", MappingProxyType(journal))


@dataclass(frozen=True)
class Phase4SourceFreezeVerification:
    directory: Path
    source_freeze: Mapping[str, Any]
    seal: Mapping[str, Any]
    baseline_binding: Mapping[str, Any]
    source_freeze_sha256: str
    seal_sha256: str
    samples_sha256: str


def _validate_source_binding(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_FIELDS:
        raise Phase4SourceFreezeError(
            "Phase-4 source binding must contain exact commit and tree fields."
        )
    if any(
        not isinstance(value[name], str) or _GIT_OBJECT.fullmatch(value[name]) is None
        for name in _SOURCE_FIELDS
    ):
        raise Phase4SourceFreezeError("Phase-4 Git object identity is invalid.")
    return {"commit": value["commit"], "tree": value["tree"]}


def _validate_host_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _HOST_OBSERVATION_FIELDS:
        raise Phase4SourceFreezeError("Current-host observation fields are not exact.")
    if (
        isinstance(value["gpu_index"], bool)
        or not isinstance(value["gpu_index"], int)
        or value["gpu_index"] < 0
        or isinstance(value["gpu_memory_total_bytes"], bool)
        or not isinstance(value["gpu_memory_total_bytes"], int)
        or value["gpu_memory_total_bytes"] <= 0
        or isinstance(value["host_memory_total_bytes"], bool)
        or not isinstance(value["host_memory_total_bytes"], int)
        or value["host_memory_total_bytes"] <= 0
        or isinstance(value["logical_cpu_count"], bool)
        or not isinstance(value["logical_cpu_count"], int)
        or value["logical_cpu_count"] <= 0
    ):
        raise Phase4SourceFreezeError("Current-host GPU numeric identity is invalid.")
    for name in (
        "gpu_name",
        "kernel_release",
        "nvidia_driver_version",
        "gpu_thermal_limits_status",
        "gpu_thermal_limits_support_binding",
    ):
        if (
            not isinstance(value[name], str)
            or not value[name]
            or value[name].strip() != value[name]
        ):
            raise Phase4SourceFreezeError("Current-host text identity is invalid.")
    thermal_status = value["gpu_thermal_limits_status"]
    thermal_limits = value["gpu_thermal_limits"]
    if (
        thermal_status not in {"supported", "unsupported"}
        or (
            thermal_status == "supported"
            and (
                type(thermal_limits) is not dict
                or set(thermal_limits)
                != {
                    "maximum_operating_temperature_c",
                    "slowdown_temperature_c",
                    "shutdown_temperature_c",
                    "target_temperature_c",
                }
            )
        )
        or (thermal_status == "unsupported" and thermal_limits is not None)
    ):
        raise Phase4SourceFreezeError("Current-host thermal-limit identity is invalid.")
    for name in (
        "gpu_uuid_sha256",
        "machine_id_sha256",
        "nvidia_smi_binding_sha256",
    ):
        if not isinstance(value[name], str) or _DIGEST.fullmatch(value[name]) is None:
            raise Phase4SourceFreezeError("Current-host digest identity is invalid.")
    return dict(value)


def _validate_journal_boot_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _JOURNAL_AUTHORITY_FIELDS:
        raise Phase4SourceFreezeError("Current-boot journal authority is not exact.")
    if value["schema_version"] != JOURNAL_BOOT_AUTHORITY_SCHEMA:
        raise Phase4SourceFreezeError("Current-boot journal schema is unsupported.")
    for name in (
        "boot_id_sha256",
        "journalctl_binding_sha256",
        "initial_cursor_sha256",
        "final_cursor_sha256",
    ):
        if not isinstance(value[name], str) or _DIGEST.fullmatch(value[name]) is None:
            raise Phase4SourceFreezeError("Current-boot journal digest is invalid.")
    for name in ("initial_projection", "final_projection"):
        if type(value[name]) is not dict or value[name] != _CLEAN_JOURNAL_PROJECTION:
            raise Phase4SourceFreezeError(
                "Current-boot journal contains an NVIDIA hardware event."
            )
    return json.loads(compact_canonical_json_bytes(value))


def _journal_support_bindings(authority: Mapping[str, Any]) -> dict[str, str]:
    suffix = (
        f"boot-sha256:{authority['boot_id_sha256']}:"
        f"journalctl-sha256:{authority['journalctl_binding_sha256']}"
    )
    return {
        "hardware_events": f"journalctl-current-boot-cursor-v1:{suffix}",
        "xid_projection": f"journalctl-nvrm-xid-v1:{suffix}",
    }


def _file_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_pinned_at(
    root_descriptor: int,
    name: str,
    *,
    exact_mode: int | None = None,
    require_current_owner: bool = False,
    maximum_bytes: int,
) -> bytes:
    """Read a Phase-4 member relative to one pinned directory descriptor."""

    if Path(name).name != name or name in {"", ".", ".."}:
        raise Phase4SourceFreezeError("A Phase-4 member name is invalid.")
    try:
        before = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
    except OSError as error:
        raise Phase4SourceFreezeError("A Phase-4 member is unavailable.") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > maximum_bytes
    ):
        raise Phase4SourceFreezeError(
            "Phase-4 members must be regular and non-hardlinked."
        )
    if (
        exact_mode is not None
        and os.name == "posix"
        and stat.S_IMODE(before.st_mode) != exact_mode
    ):
        raise Phase4SourceFreezeError("A Phase-4 member has an unsafe mode.")
    if require_current_owner and hasattr(os, "getuid") and before.st_uid != os.getuid():
        raise Phase4SourceFreezeError("A Phase-4 member has the wrong owner.")
    fingerprint = _file_fingerprint(before)
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
    except OSError as error:
        raise Phase4SourceFreezeError(
            "A Phase-4 member cannot be opened safely."
        ) from error
    payload = bytearray()
    try:
        if _file_fingerprint(os.fstat(descriptor)) != fingerprint:
            raise Phase4SourceFreezeError("A Phase-4 member changed during open.")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            payload.extend(block)
            if len(payload) > maximum_bytes:
                raise Phase4SourceFreezeError("A Phase-4 member is unbounded.")
        if _file_fingerprint(os.fstat(descriptor)) != fingerprint:
            raise Phase4SourceFreezeError("A Phase-4 member changed during read.")
    finally:
        os.close(descriptor)
    try:
        after = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
    except OSError as error:
        raise Phase4SourceFreezeError("A Phase-4 member disappeared.") from error
    if _file_fingerprint(after) != fingerprint:
        raise Phase4SourceFreezeError("A Phase-4 member path changed during read.")
    return bytes(payload)


def _read_pinned(
    path: Path,
    *,
    exact_mode: int | None = None,
    require_current_owner: bool = False,
    require_root_owner: bool = False,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> bytes:
    """Read one regular, nonlinked file through an identity-pinned descriptor."""

    try:
        before = path.lstat()
    except OSError as error:
        raise Phase4SourceFreezeError(
            "A Phase-4 source file is unavailable."
        ) from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > maximum_bytes
    ):
        raise Phase4SourceFreezeError(
            "Phase-4 source files must be regular and non-hardlinked."
        )
    if (
        exact_mode is not None
        and os.name == "posix"
        and stat.S_IMODE(before.st_mode) != exact_mode
    ):
        raise Phase4SourceFreezeError("A Phase-4 source file has an unsafe mode.")
    if require_current_owner and hasattr(os, "getuid") and before.st_uid != os.getuid():
        raise Phase4SourceFreezeError("A Phase-4 source file has the wrong owner.")
    if require_root_owner and before.st_uid != 0:
        raise Phase4SourceFreezeError("A trusted host identity file is not root-owned.")
    fingerprint = _file_fingerprint(before)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise Phase4SourceFreezeError(
            "A Phase-4 source file cannot be opened safely."
        ) from error
    payload = bytearray()
    try:
        opened = os.fstat(descriptor)
        if _file_fingerprint(opened) != fingerprint:
            raise Phase4SourceFreezeError("A Phase-4 source file changed during open.")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            payload.extend(block)
            if len(payload) > maximum_bytes:
                raise Phase4SourceFreezeError("A Phase-4 source file is unbounded.")
        if _file_fingerprint(os.fstat(descriptor)) != fingerprint:
            raise Phase4SourceFreezeError("A Phase-4 source file changed during read.")
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as error:
        raise Phase4SourceFreezeError("A Phase-4 source path disappeared.") from error
    if _file_fingerprint(after) != fingerprint:
        raise Phase4SourceFreezeError("A Phase-4 source path changed during read.")
    return bytes(payload)


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Phase-4 artifact write made no progress.")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _private_root(path: Path, *, create: bool) -> Path:
    target = path.expanduser()
    if create:
        if target.exists() or target.is_symlink():
            raise FileExistsError("Phase-4 source-freeze destination already exists.")
        parent = target.parent.resolve(strict=True)
        os.mkdir(parent / target.name, 0o700)
        target = parent / target.name
        target.chmod(0o700)
        _fsync_directory(parent)
    if target.is_symlink():
        raise Phase4SourceFreezeError("Phase-4 source-freeze root is a symlink.")
    root = target.resolve(strict=True)
    metadata = root.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
        or (os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o700)
    ):
        raise Phase4SourceFreezeError("Phase-4 source-freeze root is not private.")
    return root


def _trusted_executable(name: str) -> str:
    candidate = shutil.which(name)
    if candidate is None:
        raise Phase4SourceFreezeError(f"Trusted {name} executable is unavailable.")
    try:
        path = Path(candidate).resolve(strict=True)
        metadata = path.stat()
    except OSError as error:
        raise Phase4SourceFreezeError(
            f"Trusted {name} executable is unavailable."
        ) from error
    if (
        not path.is_absolute()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o022
        or not metadata.st_mode & 0o111
        or (os.name == "posix" and metadata.st_uid != 0)
    ):
        raise Phase4SourceFreezeError(f"Trusted {name} executable is unsafe.")
    return str(path)


def _run_bounded(
    command: Sequence[str], *, environment: Mapping[str, str] | None = None
) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=False,
            timeout=5,
            env=dict(environment) if environment is not None else None,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Phase4SourceFreezeError("A trusted Phase-4 probe failed.") from error
    if completed.returncode != 0 or len(completed.stdout) > 1024 * 1024:
        raise Phase4SourceFreezeError("A trusted Phase-4 probe did not pass.")
    try:
        return completed.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise Phase4SourceFreezeError(
            "A trusted Phase-4 probe returned invalid text."
        ) from error


def _derive_source_binding(repository_root: Path) -> dict[str, str]:
    if repository_root.is_symlink():
        raise Phase4SourceFreezeError("Source repository root cannot be a symlink.")
    root = repository_root.resolve(strict=True)
    metadata = root.stat()
    if not stat.S_ISDIR(metadata.st_mode) or (
        hasattr(os, "getuid") and metadata.st_uid != os.getuid()
    ):
        raise Phase4SourceFreezeError("Source repository ownership is not trusted.")
    git = _trusted_executable("git")
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/nonexistent-aptus-phase4",
        "LC_ALL": "C",
        "PATH": str(Path(git).parent),
    }
    prefix = [git, "-c", "core.hooksPath=/dev/null", "-C", str(root)]
    observed_root = _run_bounded(
        prefix + ["rev-parse", "--show-toplevel"], environment=environment
    ).strip()
    if Path(observed_root).resolve(strict=True) != root:
        raise Phase4SourceFreezeError("Git source root differs from the declared root.")
    commit = _run_bounded(
        prefix + ["rev-parse", "--verify", "HEAD^{commit}"], environment=environment
    ).strip()
    tree = _run_bounded(
        prefix + ["rev-parse", "--verify", "HEAD^{tree}"], environment=environment
    ).strip()
    dirty = _run_bounded(
        prefix + ["status", "--porcelain=v1", "--untracked-files=all"],
        environment=environment,
    )
    if dirty:
        raise Phase4SourceFreezeError("Phase-4 source repository is not clean.")
    return _validate_source_binding({"commit": commit, "tree": tree})


def _machine_id_sha256() -> str:
    payload = _read_pinned(
        Path("/etc/machine-id"),
        require_root_owner=True,
        maximum_bytes=4096,
    )
    try:
        machine_id = payload.decode("ascii", errors="strict").strip().lower()
    except UnicodeDecodeError as error:
        raise Phase4SourceFreezeError("Machine identity is unreadable.") from error
    if _MACHINE_ID.fullmatch(machine_id) is None:
        raise Phase4SourceFreezeError("Machine identity is invalid.")
    return sha256_bytes(machine_id.encode("ascii"))


def _trusted_host_facts() -> dict[str, Any]:
    payload = _read_pinned(
        Path("/proc/meminfo"),
        require_root_owner=True,
        maximum_bytes=1024 * 1024,
    )
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise Phase4SourceFreezeError("Host memory identity is unreadable.") from error
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", text, re.MULTILINE)
    logical_cpu_count = os.cpu_count()
    try:
        kernel_release = os.uname().release
    except AttributeError as error:
        raise Phase4SourceFreezeError("Kernel identity is unavailable.") from error
    if (
        match is None
        or logical_cpu_count is None
        or logical_cpu_count <= 0
        or not kernel_release
    ):
        raise Phase4SourceFreezeError("Required current-host facts are unavailable.")
    return {
        "host_memory_total_bytes": int(match.group(1)) * 1024,
        "kernel_release": kernel_release,
        "logical_cpu_count": logical_cpu_count,
    }


def _derive_host_observation(
    *, nvidia_smi_path: str | None, gpu_index: int
) -> tuple[dict[str, Any], TrustedExecutable]:
    if os.name != "posix" or not os.path.exists("/proc"):
        raise Phase4SourceFreezeError(
            "Qualifying Phase-4 host verification requires Linux."
        )
    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
        raise Phase4SourceFreezeError("GPU index is invalid.")
    trusted = resolve_trusted_nvidia_smi(nvidia_smi_path)
    thermal = detect_nvidia_thermal_limit_authority(trusted, gpu_index=gpu_index)
    output = _run_bounded(
        (
            trusted.verify(),
            f"--id={gpu_index}",
            "--query-gpu=uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        )
    )
    rows = list(csv.reader(output.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise Phase4SourceFreezeError("NVIDIA host identity query is ambiguous.")
    gpu_uuid, gpu_name, memory_text, driver = (item.strip() for item in rows[0])
    try:
        memory_mib = int(memory_text)
    except ValueError as error:
        raise Phase4SourceFreezeError("NVIDIA memory identity is invalid.") from error
    if not gpu_uuid or not gpu_name or not driver or memory_mib <= 0:
        raise Phase4SourceFreezeError("NVIDIA host identity is incomplete.")
    observation = {
        "gpu_index": gpu_index,
        "gpu_memory_total_bytes": memory_mib * 1024 * 1024,
        "gpu_name": gpu_name,
        "gpu_thermal_limits": (
            dict(thermal.limits) if thermal.limits is not None else None
        ),
        "gpu_thermal_limits_status": thermal.status,
        "gpu_thermal_limits_support_binding": thermal.support_binding,
        "gpu_uuid_sha256": sha256_bytes(gpu_uuid.encode("utf-8")),
        **_trusted_host_facts(),
        "machine_id_sha256": _machine_id_sha256(),
        "nvidia_driver_version": driver,
        "nvidia_smi_binding_sha256": trusted.binding_sha256,
    }
    return _validate_host_observation(observation), trusted


def derive_trusted_phase4_boundary(
    repository_root: Path,
    *,
    nvidia_smi_path: str | None = None,
    gpu_index: int = 0,
) -> TrustedPhase4Boundary:
    """Derive the clean Git and current Linux/NVIDIA authority in production."""

    source = _derive_source_binding(repository_root)
    host, trusted = _derive_host_observation(
        nvidia_smi_path=nvidia_smi_path, gpu_index=gpu_index
    )
    try:
        journal = LinuxNvidiaJournalEventProvider.production()
        journal.snapshot()
        journal_authority = journal.authority().as_record()
    except (ProbeFailure, TypeError, ValueError) as error:
        raise Phase4SourceFreezeError(
            "Current-boot journal authority is unavailable."
        ) from error
    return TrustedPhase4Boundary(
        source_binding=source,
        host_observation=host,
        journal_boot_authority=journal_authority,
        nvidia_smi=trusted,
    )


def _validate_phase4_configuration(
    value: Any,
    *,
    host_observation: Mapping[str, Any],
    journal_boot_authority: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _CONFIGURATION_FIELDS:
        raise Phase4SourceFreezeError(
            "Phase-4 telemetry configuration fields are not exact."
        )
    unsigned = dict(value)
    digest = unsigned.pop("configuration_sha256")
    if (
        not isinstance(digest, str)
        or _DIGEST.fullmatch(digest) is None
        or sha256_bytes(compact_canonical_json_bytes(unsigned)) != digest
    ):
        raise Phase4SourceFreezeError(
            "Phase-4 telemetry configuration digest is invalid."
        )
    if value["format_version"] != "aptus.cuda-telemetry-configuration.v1" or value[
        "profile"
    ] != {
        "id": "phase1-frozen-qualifying",
        "qualifying": True,
        "reason_code": None,
    }:
        raise Phase4SourceFreezeError(
            "Phase-4 telemetry profile is not production qualifying."
        )
    provenance = value["provenance"]
    if type(provenance) is not dict or set(provenance) != {
        "disk_growth_binding",
        "ownership_binding",
        "provider",
        "support_bindings",
    }:
        raise Phase4SourceFreezeError("Phase-4 telemetry provenance is invalid.")
    if (
        provenance["disk_growth_binding"] != "factory-owned-statvfs-baseline-v1"
        or provenance["ownership_binding"]
        != "factory-owned-job-service-process-group-v1"
        or provenance["provider"]
        != {
            "name": "linux-nvidia-host-probe",
            "version": "aptus-cuda-campaign-v1",
        }
    ):
        raise Phase4SourceFreezeError(
            "Phase-4 telemetry provider is not the production provider."
        )
    support = provenance["support_bindings"]
    expected_journal_support = _journal_support_bindings(journal_boot_authority)
    if (
        type(support) is not dict
        or set(support) != _SUPPORT_BINDINGS
        or any(not isinstance(item, str) or not item for item in support.values())
        or support["nvidia_smi_binary"]
        != "sha256:" + str(host_observation["nvidia_smi_binding_sha256"])
        or support["gpu_thermal_limits"]
        != host_observation["gpu_thermal_limits_support_binding"]
        or support["hardware_events"] != expected_journal_support["hardware_events"]
        or support["xid_projection"] != expected_journal_support["xid_projection"]
    ):
        raise Phase4SourceFreezeError(
            "Phase-4 telemetry support provenance is misbound."
        )
    expected_available = host_observation["gpu_thermal_limits_status"] == "supported"
    if value["thermal_policy"] != {
        "initial_limits_available": expected_available,
        "mode": (
            "reported-limits-bound"
            if expected_available
            else "frozen-conservative-fallback"
        ),
    }:
        raise Phase4SourceFreezeError("Phase-4 thermal policy is not host-derived.")
    if value["sampling"] != {
        "interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "minimum_qualifying_coverage": MINIMUM_QUALIFYING_COVERAGE,
        "watchdog_interval_seconds": 0.25,
    }:
        raise Phase4SourceFreezeError(
            "Phase-4 telemetry sampling policy is not frozen."
        )
    return json.loads(compact_canonical_json_bytes(value))


def _validate_baseline_samples(
    payload: bytes,
    *,
    host_observation: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in payload.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4SourceFreezeError(
            "Phase-4 baseline samples are invalid JSONL."
        ) from error
    if len(rows) != PHASE4_BASELINE_SAMPLE_COUNT or any(
        type(row) is not dict for row in rows
    ):
        raise Phase4SourceFreezeError(
            "Phase-4 baseline must contain exactly 600 samples."
        )
    try:
        samples = [validate_telemetry_sample(row) for row in rows]
    except (TypeError, ValueError) as error:
        raise Phase4SourceFreezeError(
            "A Phase-4 baseline sample is invalid."
        ) from error
    if payload != canonical_jsonl_bytes(samples):
        raise Phase4SourceFreezeError("Phase-4 baseline JSONL bytes are not canonical.")
    run_ids = {sample["experiment_run_id"] for sample in samples}
    origin = samples[0]["scheduled_monotonic_ns"]
    if (
        len(run_ids) != 1
        or [sample["sequence"] for sample in samples]
        != list(range(PHASE4_BASELINE_SAMPLE_COUNT))
        or [sample["scheduled_slot"] for sample in samples]
        != list(range(PHASE4_BASELINE_SAMPLE_COUNT))
        or any(
            sample["scheduled_monotonic_ns"] != origin + index * 1_000_000_000
            for index, sample in enumerate(samples)
        )
    ):
        raise Phase4SourceFreezeError(
            "Phase-4 baseline sequence is not contiguous at 1 Hz."
        )
    gpu_uuids = {sample["gpu"]["uuid"] for sample in samples}
    total_memory = {sample["gpu"]["memory"]["total"]["bytes"] for sample in samples}
    if (
        len(gpu_uuids) != 1
        or sha256_bytes(next(iter(gpu_uuids)).encode("utf-8"))
        != host_observation["gpu_uuid_sha256"]
        or total_memory != {host_observation["gpu_memory_total_bytes"]}
    ):
        raise Phase4SourceFreezeError("Phase-4 baseline was captured from another GPU.")
    validation = validate_idle_baseline(
        samples, required_samples=PHASE4_BASELINE_SAMPLE_COUNT
    )
    if not validation.valid:
        raise Phase4SourceFreezeError(
            "Phase-4 idle baseline did not pass its frozen window."
        )
    summary = dict(validation.summary)
    process_counts = [len(sample["gpu"]["compute_processes"]) for sample in samples]
    if any(process_counts):
        raise Phase4SourceFreezeError(
            "Phase-4 idle baseline contains a GPU compute process."
        )
    summary.update(
        gpu_compute_process_maximum_count=max(process_counts),
        gpu_compute_process_nonempty_sample_count=sum(
            count > 0 for count in process_counts
        ),
        gpu_throttle_event_sample_count=sum(
            bool(sample["gpu"]["throttle_reasons"]) for sample in samples
        ),
        gpu_xid_event_sample_count=sum(
            bool(sample["gpu"]["xid_errors"]) for sample in samples
        ),
        gpu_reset_sample_count=sum(
            sample["gpu"]["reset_detected"] for sample in samples
        ),
        gpu_device_lost_sample_count=sum(
            sample["gpu"]["device_lost"] for sample in samples
        ),
        gpu_hardware_error_sample_count=sum(
            sample["gpu"]["hardware_error"] for sample in samples
        ),
        aptus_lease_active_sample_count=sum(
            sample["host"]["aptus_lease_active"] for sample in samples
        ),
        watchdog_ownership_uncertain_sample_count=sum(
            not sample["watchdog"]["ownership_certain"] for sample in samples
        ),
    )
    return samples, summary


def _record_digest(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(value)))


def _validate_identity_chain(
    campaign: Mapping[str, Any],
    cohort: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        canonical_campaign = validate_record(
            dict(campaign), SCHEMA_VERSIONS["campaign"]
        )
        canonical_cohort = validate_record(
            dict(cohort), SCHEMA_VERSIONS["comparison_cohort"]
        )
        canonical_cell = validate_record(dict(cell), SCHEMA_VERSIONS["comparison_cell"])
    except (TypeError, ValueError) as error:
        raise Phase4SourceFreezeError(
            "A Phase-4 campaign identity record is invalid."
        ) from error
    if (
        canonical_cohort["campaign_id"] != canonical_campaign["campaign_id"]
        or canonical_cell["campaign_id"] != canonical_campaign["campaign_id"]
        or canonical_cell["comparison_cell_id"]
        not in canonical_cohort["member_cell_ids"]
    ):
        raise Phase4SourceFreezeError(
            "Phase-4 campaign/cohort/cell membership is invalid."
        )
    return canonical_campaign, canonical_cohort, canonical_cell


def _build_freeze_record(
    *,
    campaign: Mapping[str, Any],
    cohort: Mapping[str, Any],
    cell: Mapping[str, Any],
    boundary: TrustedPhase4Boundary,
    configuration: Mapping[str, Any],
    samples_sha256: str,
    samples: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    source = _validate_source_binding(boundary.source_binding)
    if source != cell["source_binding"]:
        raise Phase4SourceFreezeError(
            "The clean Git source differs from the comparison cell."
        )
    host_binding = cell["host_binding"]
    if (
        type(host_binding) is not dict
        or set(host_binding) != {"host_id", *_HOST_OBSERVATION_FIELDS}
        or not isinstance(host_binding.get("host_id"), str)
        or _HOST_ID.fullmatch(host_binding["host_id"]) is None
        or {name: host_binding[name] for name in sorted(_HOST_OBSERVATION_FIELDS)}
        != {
            name: boundary.host_observation[name]
            for name in sorted(_HOST_OBSERVATION_FIELDS)
        }
    ):
        raise Phase4SourceFreezeError(
            "The comparison-cell host binding differs from the current trusted host."
        )
    environment = cell["environment_binding"]
    if (
        isinstance(environment, dict)
        and "nvidia_driver_version" in environment
        and environment["nvidia_driver_version"]
        != boundary.host_observation["nvidia_driver_version"]
    ):
        raise Phase4SourceFreezeError(
            "The NVIDIA driver differs from the cell environment."
        )
    return {
        "schema_version": PHASE4_SOURCE_FREEZE_SCHEMA,
        "producer": dict(_PRODUCTION_TOOL),
        "campaign_id": campaign["campaign_id"],
        "comparison_cohort_id": cohort["comparison_cohort_id"],
        "comparison_cell_id": cell["comparison_cell_id"],
        "campaign_sha256": _record_digest(campaign),
        "comparison_cohort_sha256": _record_digest(cohort),
        "comparison_cell_sha256": _record_digest(cell),
        "source_binding": source,
        "host_binding_sha256": sha256_bytes(canonical_json_bytes(cell["host_binding"])),
        "environment_binding_sha256": sha256_bytes(
            canonical_json_bytes(cell["environment_binding"])
        ),
        "model_binding_sha256": sha256_bytes(
            canonical_json_bytes(cell["model_binding"])
        ),
        "dataset_and_split_binding_sha256": sha256_bytes(
            canonical_json_bytes(cell["dataset_and_split_binding"])
        ),
        "method": cell["method"],
        "retention_policy_id": cell["retention_policy_id"],
        "current_host_observation": dict(boundary.host_observation),
        "current_host_binding_sha256": sha256_bytes(
            canonical_json_bytes(dict(boundary.host_observation))
        ),
        "current_boot_authority": dict(boundary.journal_boot_authority),
        "telemetry_configuration": dict(configuration),
        "telemetry_configuration_sha256": configuration["configuration_sha256"],
        "idle_baseline_experiment_run_id": samples[0]["experiment_run_id"],
        "idle_baseline_samples_sha256": samples_sha256,
        "idle_baseline_sample_count": PHASE4_BASELINE_SAMPLE_COUNT,
        "idle_baseline_summary": dict(summary),
        "created_at_utc": _normalized_utc_timestamp(
            utc_now(), "Phase-4 created_at_utc"
        ),
    }


def _create_phase4_source_freeze_artifact(
    directory: Path,
    *,
    repository_root: Path,
    campaign: Mapping[str, Any],
    comparison_cohort: Mapping[str, Any],
    comparison_cell: Mapping[str, Any],
    telemetry_configuration: Mapping[str, Any],
    telemetry_samples_path: Path,
    nvidia_smi_path: str | None = None,
    gpu_index: int = 0,
    _trusted_boundary: TrustedPhase4Boundary | None = None,
    _test_token: object | None = None,
    _nonproduction_test: bool = False,
) -> Phase4SourceFreezeVerification:
    """Create one no-clobber Phase-4 authority from production-derived facts."""

    campaign_record, cohort_record, cell_record = _validate_identity_chain(
        campaign, comparison_cohort, comparison_cell
    )
    repository = repository_root.resolve(strict=True)
    destination_parent = directory.expanduser().parent.resolve(strict=True)
    if destination_parent == repository or repository in destination_parent.parents:
        raise Phase4SourceFreezeError(
            "Phase-4 source-freeze artifacts must be stored outside the source tree."
        )
    if _trusted_boundary is None:
        if _nonproduction_test:
            raise TypeError("Nonproduction Phase-4 creation requires test authority.")
        boundary = derive_trusted_phase4_boundary(
            repository,
            nvidia_smi_path=nvidia_smi_path,
            gpu_index=gpu_index,
        )
    elif _nonproduction_test and _phase4_test_token_is_authorized(_test_token):
        boundary = _trusted_boundary
    else:
        raise TypeError("Caller-supplied Phase-4 authority is forbidden.")
    configuration = _validate_phase4_configuration(
        dict(telemetry_configuration),
        host_observation=boundary.host_observation,
        journal_boot_authority=boundary.journal_boot_authority,
    )
    samples_payload = _read_pinned(
        telemetry_samples_path,
        exact_mode=0o600,
        require_current_owner=True,
    )
    samples, summary = _validate_baseline_samples(
        samples_payload, host_observation=boundary.host_observation
    )
    samples_sha256 = sha256_bytes(samples_payload)
    record = _build_freeze_record(
        campaign=campaign_record,
        cohort=cohort_record,
        cell=cell_record,
        boundary=boundary,
        configuration=configuration,
        samples_sha256=samples_sha256,
        samples=samples,
        summary=summary,
    )
    if _nonproduction_test:
        record["producer"] = dict(_NONPRODUCTION_TEST_TOOL)
    record_bytes = canonical_json_bytes(record)
    root = _private_root(directory, create=True)
    try:
        _write_exclusive(root / PHASE4_SOURCE_FREEZE_NAME, record_bytes)
        _write_exclusive(root / PHASE4_IDLE_SAMPLES_NAME, samples_payload)
        if not _nonproduction_test:
            repeated = derive_trusted_phase4_boundary(
                repository,
                nvidia_smi_path=nvidia_smi_path,
                gpu_index=gpu_index,
            )
            if dict(repeated.source_binding) != dict(boundary.source_binding) or dict(
                repeated.host_observation
            ) != dict(boundary.host_observation):
                raise Phase4SourceFreezeError(
                    "Phase-4 source or host changed before the seal was written."
                )
            for name in ("boot_id_sha256", "journalctl_binding_sha256"):
                if (
                    repeated.journal_boot_authority[name]
                    != (boundary.journal_boot_authority[name])
                ):
                    raise Phase4SourceFreezeError(
                        "Phase-4 boot or journal authority changed before sealing."
                    )
        seal = {
            "schema_version": PHASE4_SOURCE_FREEZE_SEAL_SCHEMA,
            "source_freeze_sha256": sha256_bytes(record_bytes),
            "source_freeze_size_bytes": len(record_bytes),
            "idle_baseline_samples_sha256": samples_sha256,
            "idle_baseline_samples_size_bytes": len(samples_payload),
            "sealed_at_utc": _normalized_utc_timestamp(
                utc_now(), "Phase-4 sealed_at_utc"
            ),
        }
        _write_exclusive(
            root / PHASE4_SOURCE_FREEZE_SEAL_NAME, canonical_json_bytes(seal)
        )
        _fsync_directory(root)
    except BaseException:
        # The no-clobber directory is intentionally left in place as evidence
        # of an incomplete freeze; verification will reject it.
        raise
    return _verify_phase4_source_freeze_artifact(
        root,
        repository_root=repository,
        campaign=campaign_record,
        comparison_cohort=cohort_record,
        comparison_cell=cell_record,
        nvidia_smi_path=nvidia_smi_path,
        gpu_index=gpu_index,
        _trusted_boundary=boundary if _nonproduction_test else None,
        _test_token=_test_token,
        _nonproduction_test=_nonproduction_test,
    )


def _load_canonical_object(
    payload: bytes, expected_fields: frozenset[str]
) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4SourceFreezeError(
            "A Phase-4 metadata file is invalid JSON."
        ) from error
    if (
        type(value) is not dict
        or set(value) != expected_fields
        or canonical_json_bytes(value) != payload
    ):
        raise Phase4SourceFreezeError(
            "Phase-4 metadata bytes are not exact canonical JSON."
        )
    return value


def _verify_phase4_source_freeze_artifact(
    directory: Path,
    *,
    repository_root: Path,
    campaign: Mapping[str, Any],
    comparison_cohort: Mapping[str, Any],
    comparison_cell: Mapping[str, Any],
    nvidia_smi_path: str | None = None,
    gpu_index: int = 0,
    _trusted_boundary: TrustedPhase4Boundary | None = None,
    _test_token: object | None = None,
    _nonproduction_test: bool = False,
) -> Phase4SourceFreezeVerification:
    """Deep-verify a sealed freeze against the still-current source and host."""

    campaign_record, cohort_record, cell_record = _validate_identity_chain(
        campaign, comparison_cohort, comparison_cell
    )
    root = _private_root(directory, create=False)
    expected_names = {
        PHASE4_SOURCE_FREEZE_NAME,
        PHASE4_IDLE_SAMPLES_NAME,
        PHASE4_SOURCE_FREEZE_SEAL_NAME,
    }
    try:
        root_before = root.lstat()
        root_fingerprint = _file_fingerprint(root_before)
        root_descriptor = os.open(
            root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise Phase4SourceFreezeError(
            "Phase-4 source-freeze root cannot be pinned."
        ) from error
    try:
        if _file_fingerprint(os.fstat(root_descriptor)) != root_fingerprint:
            raise Phase4SourceFreezeError(
                "Phase-4 source-freeze root changed during open."
            )
        if set(os.listdir(root_descriptor)) != expected_names:
            raise Phase4SourceFreezeError(
                "Phase-4 source-freeze file inventory is not exact."
            )
        record_bytes = _read_pinned_at(
            root_descriptor,
            PHASE4_SOURCE_FREEZE_NAME,
            exact_mode=0o600,
            require_current_owner=True,
            maximum_bytes=2 * 1024 * 1024,
        )
        samples_bytes = _read_pinned_at(
            root_descriptor,
            PHASE4_IDLE_SAMPLES_NAME,
            exact_mode=0o600,
            require_current_owner=True,
            maximum_bytes=64 * 1024 * 1024,
        )
        seal_bytes = _read_pinned_at(
            root_descriptor,
            PHASE4_SOURCE_FREEZE_SEAL_NAME,
            exact_mode=0o600,
            require_current_owner=True,
            maximum_bytes=64 * 1024,
        )
        if (
            set(os.listdir(root_descriptor)) != expected_names
            or _file_fingerprint(os.fstat(root_descriptor)) != root_fingerprint
        ):
            raise Phase4SourceFreezeError(
                "Phase-4 source-freeze inventory changed during verification."
            )
    finally:
        os.close(root_descriptor)
    try:
        root_after = root.lstat()
    except OSError as error:
        raise Phase4SourceFreezeError(
            "Phase-4 source-freeze root disappeared during verification."
        ) from error
    if _file_fingerprint(root_after) != root_fingerprint:
        raise Phase4SourceFreezeError(
            "Phase-4 source-freeze root changed during verification."
        )
    record = _load_canonical_object(record_bytes, _FREEZE_FIELDS)
    seal = _load_canonical_object(seal_bytes, _SEAL_FIELDS)
    _normalized_utc_timestamp(record["created_at_utc"], "Phase-4 created_at_utc")
    _normalized_utc_timestamp(seal["sealed_at_utc"], "Phase-4 sealed_at_utc")
    if (
        record["schema_version"] != PHASE4_SOURCE_FREEZE_SCHEMA
        or record["producer"]
        != (_NONPRODUCTION_TEST_TOOL if _nonproduction_test else _PRODUCTION_TOOL)
        or seal["schema_version"] != PHASE4_SOURCE_FREEZE_SEAL_SCHEMA
        or seal["source_freeze_sha256"] != sha256_bytes(record_bytes)
        or seal["source_freeze_size_bytes"] != len(record_bytes)
        or seal["idle_baseline_samples_sha256"] != sha256_bytes(samples_bytes)
        or seal["idle_baseline_samples_size_bytes"] != len(samples_bytes)
    ):
        raise Phase4SourceFreezeError("Phase-4 seal does not bind its exact bytes.")
    if _trusted_boundary is None:
        if _nonproduction_test:
            raise TypeError(
                "Nonproduction Phase-4 verification requires test authority."
            )
        boundary = derive_trusted_phase4_boundary(
            repository_root,
            nvidia_smi_path=nvidia_smi_path,
            gpu_index=gpu_index,
        )
    elif _nonproduction_test and _phase4_test_token_is_authorized(_test_token):
        boundary = _trusted_boundary
    else:
        raise TypeError("Caller-supplied Phase-4 authority is forbidden.")
    sealed_journal = _validate_journal_boot_authority(record["current_boot_authority"])
    for name in ("boot_id_sha256", "journalctl_binding_sha256"):
        if sealed_journal[name] != boundary.journal_boot_authority[name]:
            raise Phase4SourceFreezeError(
                "Phase-4 artifact belongs to another boot or journal binary."
            )
    configuration = _validate_phase4_configuration(
        record["telemetry_configuration"],
        host_observation=boundary.host_observation,
        journal_boot_authority=boundary.journal_boot_authority,
    )
    samples, summary = _validate_baseline_samples(
        samples_bytes, host_observation=boundary.host_observation
    )
    expected_record = _build_freeze_record(
        campaign=campaign_record,
        cohort=cohort_record,
        cell=cell_record,
        boundary=boundary,
        configuration=configuration,
        samples_sha256=sha256_bytes(samples_bytes),
        samples=samples,
        summary=summary,
    )
    if _nonproduction_test:
        expected_record["producer"] = dict(_NONPRODUCTION_TEST_TOOL)
    # Creation times are auditable but not recomputable; every other field is.
    expected_record["created_at_utc"] = record["created_at_utc"]
    # Cursor digests prove the sealed baseline bracket. A later verification may
    # legitimately observe a newer clean cursor on the same boot.
    expected_record["current_boot_authority"] = sealed_journal
    if record != expected_record:
        raise Phase4SourceFreezeError(
            "Phase-4 source-freeze facts differ from current trusted authority."
        )
    source_digest = sha256_bytes(record_bytes)
    seal_digest = sha256_bytes(seal_bytes)
    samples_digest = sha256_bytes(samples_bytes)
    baseline_summary = {
        name: summary[name]
        for name in (
            "gpu_temperature_median_c",
            "gpu_temperature_p95_c",
            "gpu_free_vram_median_bytes",
            "gpu_power_draw_p95_w",
        )
    }
    baseline_binding = {
        "schema_version": "aptus.cuda-campaign-idle-baseline-binding.v1",
        "phase4_source_freeze_sha256": source_digest,
        "phase4_source_freeze_seal_sha256": seal_digest,
        "idle_baseline_samples_sha256": samples_digest,
        "telemetry_configuration_sha256": configuration["configuration_sha256"],
        "host_binding_sha256": record["host_binding_sha256"],
        "current_host_binding_sha256": record["current_host_binding_sha256"],
        "current_boot_id_sha256": sealed_journal["boot_id_sha256"],
        "journalctl_binding_sha256": sealed_journal["journalctl_binding_sha256"],
        "summary": baseline_summary,
    }
    return Phase4SourceFreezeVerification(
        directory=root,
        source_freeze=MappingProxyType(record),
        seal=MappingProxyType(seal),
        baseline_binding=MappingProxyType(baseline_binding),
        source_freeze_sha256=source_digest,
        seal_sha256=seal_digest,
        samples_sha256=samples_digest,
    )


def _validate_retained_phase4_source_freeze(
    *,
    source_freeze_bytes: bytes,
    idle_samples_bytes: bytes,
    seal_bytes: bytes,
    campaign: Mapping[str, Any],
    comparison_cohort: Mapping[str, Any],
    comparison_cell: Mapping[str, Any],
    _nonproduction_test: bool = False,
) -> Phase4SourceFreezeVerification:
    """Deep-verify retained Phase-4 bytes without claiming current-host parity.

    This is for later sealed-artifact integrity checks. New qualifying admission
    must use :func:`verify_phase4_source_freeze_artifact`, which additionally
    re-derives the live Git and Linux/NVIDIA boundary.
    """

    campaign_record, cohort_record, cell_record = _validate_identity_chain(
        campaign, comparison_cohort, comparison_cell
    )
    record = _load_canonical_object(source_freeze_bytes, _FREEZE_FIELDS)
    seal = _load_canonical_object(seal_bytes, _SEAL_FIELDS)
    _normalized_utc_timestamp(
        record["created_at_utc"], "Retained Phase-4 created_at_utc"
    )
    _normalized_utc_timestamp(seal["sealed_at_utc"], "Retained Phase-4 sealed_at_utc")
    source_digest = sha256_bytes(source_freeze_bytes)
    samples_digest = sha256_bytes(idle_samples_bytes)
    seal_digest = sha256_bytes(seal_bytes)
    if (
        record["schema_version"] != PHASE4_SOURCE_FREEZE_SCHEMA
        or record["producer"]
        != (_NONPRODUCTION_TEST_TOOL if _nonproduction_test else _PRODUCTION_TOOL)
        or seal["schema_version"] != PHASE4_SOURCE_FREEZE_SEAL_SCHEMA
        or seal["source_freeze_sha256"] != source_digest
        or seal["source_freeze_size_bytes"] != len(source_freeze_bytes)
        or seal["idle_baseline_samples_sha256"] != samples_digest
        or seal["idle_baseline_samples_size_bytes"] != len(idle_samples_bytes)
    ):
        raise Phase4SourceFreezeError("Retained Phase-4 seal is misbound.")
    host = _validate_host_observation(record["current_host_observation"])
    journal_authority = _validate_journal_boot_authority(
        record["current_boot_authority"]
    )
    boundary = TrustedPhase4Boundary(
        source_binding=record["source_binding"],
        host_observation=host,
        journal_boot_authority=journal_authority,
    )
    configuration = _validate_phase4_configuration(
        record["telemetry_configuration"],
        host_observation=host,
        journal_boot_authority=record["current_boot_authority"],
    )
    samples, summary = _validate_baseline_samples(
        idle_samples_bytes, host_observation=host
    )
    expected = _build_freeze_record(
        campaign=campaign_record,
        cohort=cohort_record,
        cell=cell_record,
        boundary=boundary,
        configuration=configuration,
        samples_sha256=samples_digest,
        samples=samples,
        summary=summary,
    )
    if _nonproduction_test:
        expected["producer"] = dict(_NONPRODUCTION_TEST_TOOL)
    expected["created_at_utc"] = record["created_at_utc"]
    if expected != record:
        raise Phase4SourceFreezeError(
            "Retained Phase-4 authority is internally misbound."
        )
    baseline_binding = {
        "schema_version": "aptus.cuda-campaign-idle-baseline-binding.v1",
        "phase4_source_freeze_sha256": source_digest,
        "phase4_source_freeze_seal_sha256": seal_digest,
        "idle_baseline_samples_sha256": samples_digest,
        "telemetry_configuration_sha256": configuration["configuration_sha256"],
        "host_binding_sha256": record["host_binding_sha256"],
        "current_host_binding_sha256": record["current_host_binding_sha256"],
        "current_boot_id_sha256": journal_authority["boot_id_sha256"],
        "journalctl_binding_sha256": journal_authority["journalctl_binding_sha256"],
        "summary": {
            name: summary[name]
            for name in (
                "gpu_temperature_median_c",
                "gpu_temperature_p95_c",
                "gpu_free_vram_median_bytes",
                "gpu_power_draw_p95_w",
            )
        },
    }
    return Phase4SourceFreezeVerification(
        directory=Path("."),
        source_freeze=MappingProxyType(record),
        seal=MappingProxyType(seal),
        baseline_binding=MappingProxyType(baseline_binding),
        source_freeze_sha256=source_digest,
        seal_sha256=seal_digest,
        samples_sha256=samples_digest,
    )


def _test_phase4_boundary(
    *,
    source_binding: Mapping[str, str],
    host_observation: Mapping[str, Any],
    journal_boot_authority: Mapping[str, Any] | None = None,
) -> TrustedPhase4Boundary:
    """Build fixture authority; production entry points never accept it directly."""

    authority = journal_boot_authority or {
        "schema_version": JOURNAL_BOOT_AUTHORITY_SCHEMA,
        "boot_id_sha256": "2" * 64,
        "journalctl_binding_sha256": "3" * 64,
        "initial_cursor_sha256": "4" * 64,
        "final_cursor_sha256": "5" * 64,
        "initial_projection": dict(_CLEAN_JOURNAL_PROJECTION),
        "final_projection": dict(_CLEAN_JOURNAL_PROJECTION),
    }
    return TrustedPhase4Boundary(
        source_binding=source_binding,
        host_observation=host_observation,
        journal_boot_authority=authority,
    )


def create_phase4_source_freeze_artifact(
    directory: Path,
    *,
    repository_root: Path,
    campaign: Mapping[str, Any],
    comparison_cohort: Mapping[str, Any],
    comparison_cell: Mapping[str, Any],
    telemetry_configuration: Mapping[str, Any],
    telemetry_samples_path: Path,
    nvidia_smi_path: str | None = None,
    gpu_index: int = 0,
) -> Phase4SourceFreezeVerification:
    """Create only from freshly derived production source and host authority."""

    return _create_phase4_source_freeze_artifact(
        directory,
        repository_root=repository_root,
        campaign=campaign,
        comparison_cohort=comparison_cohort,
        comparison_cell=comparison_cell,
        telemetry_configuration=telemetry_configuration,
        telemetry_samples_path=telemetry_samples_path,
        nvidia_smi_path=nvidia_smi_path,
        gpu_index=gpu_index,
    )


def verify_phase4_source_freeze_artifact(
    directory: Path,
    *,
    repository_root: Path,
    campaign: Mapping[str, Any],
    comparison_cohort: Mapping[str, Any],
    comparison_cell: Mapping[str, Any],
    nvidia_smi_path: str | None = None,
    gpu_index: int = 0,
) -> Phase4SourceFreezeVerification:
    """Verify only against freshly derived production source and host authority."""

    return _verify_phase4_source_freeze_artifact(
        directory,
        repository_root=repository_root,
        campaign=campaign,
        comparison_cohort=comparison_cohort,
        comparison_cell=comparison_cell,
        nvidia_smi_path=nvidia_smi_path,
        gpu_index=gpu_index,
    )


def validate_retained_phase4_source_freeze(
    *,
    source_freeze_bytes: bytes,
    idle_samples_bytes: bytes,
    seal_bytes: bytes,
    campaign: Mapping[str, Any],
    comparison_cohort: Mapping[str, Any],
    comparison_cell: Mapping[str, Any],
) -> Phase4SourceFreezeVerification:
    """Deep-verify only retained production Phase-4 artifact bytes."""

    return _validate_retained_phase4_source_freeze(
        source_freeze_bytes=source_freeze_bytes,
        idle_samples_bytes=idle_samples_bytes,
        seal_bytes=seal_bytes,
        campaign=campaign,
        comparison_cohort=comparison_cohort,
        comparison_cell=comparison_cell,
    )


def _install_nonproduction_phase4_test_boundary() -> tuple[
    Callable[[object], bool],
    Callable[..., Phase4SourceFreezeVerification],
    Callable[..., Phase4SourceFreezeVerification],
    Callable[..., Phase4SourceFreezeVerification],
]:
    token = object()

    def is_authorized(value: object) -> bool:
        return value is token

    def create_for_test(**kwargs: Any) -> Phase4SourceFreezeVerification:
        return _create_phase4_source_freeze_artifact(
            **kwargs,
            _test_token=token,
            _nonproduction_test=True,
        )

    def verify_for_test(**kwargs: Any) -> Phase4SourceFreezeVerification:
        return _verify_phase4_source_freeze_artifact(
            **kwargs,
            _test_token=token,
            _nonproduction_test=True,
        )

    def validate_retained_for_test(
        **kwargs: Any,
    ) -> Phase4SourceFreezeVerification:
        return _validate_retained_phase4_source_freeze(
            **kwargs,
            _nonproduction_test=True,
        )

    return is_authorized, create_for_test, verify_for_test, validate_retained_for_test


(
    _phase4_test_token_is_authorized,
    _create_phase4_source_freeze_for_test,
    _verify_phase4_source_freeze_for_test,
    _validate_retained_phase4_source_freeze_for_test,
) = _install_nonproduction_phase4_test_boundary()
del _install_nonproduction_phase4_test_boundary


__all__ = [
    "PHASE4_BASELINE_SAMPLE_COUNT",
    "PHASE4_IDLE_SAMPLES_NAME",
    "PHASE4_SOURCE_FREEZE_NAME",
    "PHASE4_SOURCE_FREEZE_SCHEMA",
    "PHASE4_SOURCE_FREEZE_SEAL_NAME",
    "PHASE4_SOURCE_FREEZE_SEAL_SCHEMA",
    "Phase4SourceFreezeError",
    "Phase4SourceFreezeVerification",
    "TrustedPhase4Boundary",
    "create_phase4_source_freeze_artifact",
    "derive_trusted_phase4_boundary",
    "validate_retained_phase4_source_freeze",
    "verify_phase4_source_freeze_artifact",
]
