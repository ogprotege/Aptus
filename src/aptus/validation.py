from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any, Literal, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows only.
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX only.
    msvcrt = None

from .catalog import bundle_requirements
from .domain import (
    ValidationFinding,
    ValidationReport,
    ValidationState,
    to_primitive,
    training_plan_from_primitive,
)
from .execution import _actual_hardware_binding as _job_hardware_binding
from .plan_contract import (
    bundle_fingerprint,
    sha256_file,
    validate_bundle_manifest,
    validate_plan_payload,
)
from .planning import plan_training


ValidationLevel = Literal[
    "contract", "static", "dependency", "model-data", "measured-preflight", "pilot"
]
LEVELS: tuple[ValidationLevel, ...] = (
    "contract",
    "static",
    "dependency",
    "model-data",
    "measured-preflight",
    "pilot",
)
LEVEL_STATES = {
    "contract": ValidationState.CONTRACT_PASS,
    "static": ValidationState.STATIC_PASS,
    "dependency": ValidationState.DEPENDENCY_PASS,
    "model-data": ValidationState.MODEL_DATA_PASS,
    "measured-preflight": ValidationState.MEASURED_PREFLIGHT_PASS,
    "pilot": ValidationState.PILOT_PASS,
}
STATE_RANK = {
    ValidationState.CONTRACT_PASS: 1,
    ValidationState.STATIC_PASS: 2,
    ValidationState.DEPENDENCY_PASS: 3,
    ValidationState.MODEL_DATA_PASS: 4,
    ValidationState.MEASURED_PREFLIGHT_PASS: 5,
    ValidationState.PILOT_PASS: 6,
    ValidationState.EXECUTION_APPROVED: 7,
    ValidationState.MEASURED_RUN_PASS: 8,
}
REQUIRED_BUNDLE_FILES = (
    "README.md",
    "config/accelerate.yaml",
    "config/trainer.json",
    "bundle-manifest.json",
    "candidates.json",
    "decision-report.md",
    "evidence.jsonl",
    "plan.json",
    "plan_contract.py",
    "preflight.py",
    "profiles/dataset.json",
    "profiles/hardware.json",
    "profiles/model.json",
    "requirements.txt",
    "runbook.md",
    "run.py",
    "runtime_lease.py",
    "train.py",
    "validate.py",
)


def _finding(
    code: str, message: str, *, severity: str = "error", path: str | None = None
) -> ValidationFinding:
    return ValidationFinding(code=code, message=message, severity=severity, path=path)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _environment_binding(requirements: tuple[str, ...]) -> str:
    direct_constraints: dict[str, str] = {}
    for requirement in requirements:
        name = requirement.split("==", 1)[0]
        try:
            direct_constraints[name] = version(name)
        except PackageNotFoundError:
            direct_constraints[name] = "missing"
    runtime_distributions = _runtime_distribution_closure(direct_constraints)
    return _json_hash(
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "direct_constraints": direct_constraints,
            "runtime_distributions": runtime_distributions,
        }
    )


def _runtime_distribution_closure(names: dict[str, str]) -> dict[str, str]:
    """Bind the installed dependency closure, excluding unrelated PYTHONPATH tools."""

    pending = list(names)
    observed: dict[str, str] = {}
    visited: set[str] = set()
    while pending:
        requested = pending.pop()
        normalized = requested.lower().replace("_", "-").replace(".", "-")
        if normalized in visited:
            continue
        visited.add(normalized)
        try:
            package = distribution(requested)
        except PackageNotFoundError:
            continue
        canonical = (package.metadata.get("Name") or requested).lower()
        canonical = canonical.replace("_", "-").replace(".", "-")
        observed[canonical] = package.version
        for requirement in package.requires or ():
            token = requirement.split(";", 1)[0].strip()
            boundary = min(
                (
                    token.find(character)
                    for character in "[ (<>=!~"
                    if character in token
                ),
                default=len(token),
            )
            dependency = token[:boundary].strip()
            if dependency:
                pending.append(dependency)
    return dict(sorted(observed.items()))


def _actual_hardware_binding(device_indices: list[int]) -> str:
    return _job_hardware_binding(device_indices)


def _load_json(path: Path, findings: list[ValidationFinding], code: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        findings.append(_finding(code, str(error), path=path.name))
        return None


def _read_preflight_metrics(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Measured-preflight metrics are unreadable.") from error
    if not isinstance(metrics, dict):
        raise ValueError("Measured-preflight metrics must be a JSON object.")
    candidate = plan.get("recommended")
    if not isinstance(candidate, dict):
        raise ValueError("Plan has no selected candidate for measured preflight.")
    expected = {
        "schema_version": "aptus.preflight-metrics.v1",
        "candidate_id": candidate.get("candidate_id"),
        "method": candidate.get("method"),
        "precision": candidate.get("precision"),
        "quantization": candidate.get("quantization"),
        "distribution": candidate.get("distribution"),
        "world_size": candidate.get("world_size"),
        "scope": "synthetic-method-preflight-not-model-data-pilot",
    }
    for name, value in expected.items():
        if metrics.get(name) != value:
            raise ValueError(f"Measured-preflight metrics do not bind {name}.")
    measured_peak = metrics.get("measured_peak_cuda_bytes")
    if (
        not isinstance(measured_peak, int)
        or isinstance(measured_peak, bool)
        or measured_peak <= 0
    ):
        raise ValueError(
            "Measured-preflight metrics require a positive measured_peak_cuda_bytes integer."
        )
    return metrics


def _completed_run_evidence_is_current(
    previous: ValidationReport, bundle_dir: Path, plan: dict[str, Any]
) -> bool:
    final_export = previous.final_export
    measured_run = previous.measured_run
    candidate = plan.get("recommended")
    if (
        not isinstance(final_export, Mapping)
        or not isinstance(measured_run, Mapping)
        or not isinstance(candidate, dict)
        or not previous.measured_run_completed_at
    ):
        return False
    expected_binding = {
        "plan_id": plan.get("plan_id"),
        "candidate_id": candidate.get("candidate_id"),
        "distribution": candidate.get("distribution"),
        "world_size": candidate.get("world_size"),
    }
    if any(final_export.get(name) != value for name, value in expected_binding.items()):
        return False
    if any(measured_run.get(name) != value for name, value in expected_binding.items()):
        return False
    try:
        runs_root = (bundle_dir / "runs").resolve()
        run_dir = Path(str(measured_run["output_dir"])).resolve(strict=True)
        final_dir = Path(str(final_export["path"])).resolve(strict=True)
    except (KeyError, OSError):
        return False
    if (
        run_dir.parent != runs_root
        or not run_dir.name.startswith("run_")
        or final_dir != (run_dir / "final").resolve()
    ):
        return False
    export_path = run_dir / "final-export.json"
    metrics_path = run_dir / "metrics.json"
    if not export_path.is_file() or not metrics_path.is_file():
        return False
    if final_export.get("manifest_sha256") != sha256_file(
        export_path
    ) or measured_run.get("metrics_sha256") != sha256_file(metrics_path):
        return False
    try:
        export = json.loads(export_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(export, dict) or not isinstance(metrics, dict):
        return False
    if (
        export.get("schema_version") != "aptus.final-export.v1"
        or export.get("method") != candidate.get("method")
        or export.get("distribution") != candidate.get("distribution")
        or export.get("world_size") != candidate.get("world_size")
        or metrics.get("plan_id") != plan.get("plan_id")
        or metrics.get("candidate_id") != candidate.get("candidate_id")
        or metrics.get("distribution") != candidate.get("distribution")
        or metrics.get("actual_world_size") != candidate.get("world_size")
        or metrics.get("global_step") != measured_run.get("global_step")
        or metrics.get("per_rank_cuda_peaks") != measured_run.get("per_rank_cuda_peaks")
        or metrics.get("final_export") != export
    ):
        return False
    entries = export.get("files")
    if not isinstance(entries, list) or not entries:
        return False
    observed_paths: set[str] = set()
    observed_total = 0
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            return False
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            return False
        normalized = relative.as_posix()
        if normalized in observed_paths:
            return False
        artifact = final_dir.joinpath(*relative.parts)
        try:
            resolved_artifact = artifact.resolve(strict=True)
        except OSError:
            return False
        if not artifact.is_file() or final_dir not in resolved_artifact.parents:
            return False
        size = artifact.stat().st_size
        if entry.get("size_bytes") != size or entry.get("sha256") != sha256_file(
            artifact
        ):
            return False
        observed_paths.add(normalized)
        observed_total += size
    actual_paths = {
        path.relative_to(final_dir).as_posix()
        for path in final_dir.rglob("*")
        if path.is_file()
    }
    return bool(
        observed_paths == actual_paths
        and export.get("total_bytes") == observed_total
        and final_export.get("total_bytes") == observed_total
    )


def _write_report(path: Path, report: ValidationReport) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(to_primitive(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@contextmanager
def _report_lock(bundle_dir: Path) -> Any:
    path = bundle_dir / ".validation-report.lock"
    with path.open("a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - Windows only.
            lock_file.seek(0)
            if not lock_file.read(1):
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows only.
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _read_report(path: Path) -> ValidationReport | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None
        state = ValidationState(value["state"])
        findings = tuple(
            ValidationFinding(
                code=str(item["code"]),
                message=str(item["message"]),
                severity=str(item["severity"]),
                path=str(item["path"]) if item.get("path") is not None else None,
            )
            for item in value.get("findings", [])
            if isinstance(item, dict)
        )
        command = value.get("smoke_command")
        return ValidationReport(
            state=state,
            findings=findings,
            checked_files=tuple(str(item) for item in value.get("checked_files", [])),
            artifact_fingerprint=str(value.get("artifact_fingerprint", "")),
            smoke_command=(
                tuple(str(item) for item in command)
                if isinstance(command, list)
                else None
            ),
            runtime_evidence=tuple(
                str(item) for item in value.get("runtime_evidence", [])
            ),
            validation_level=str(value.get("validation_level", "contract")),
            bindings={
                str(key): str(item) for key, item in value.get("bindings", {}).items()
            }
            if isinstance(value.get("bindings"), dict)
            else {},
            validator_version=str(value.get("validator_version", "aptus-validator-v2")),
            validated_at=value.get("validated_at"),
            preflight_metrics=(
                value.get("preflight_metrics")
                if isinstance(value.get("preflight_metrics"), dict)
                else None
            ),
            pilot_metrics=(
                value.get("pilot_metrics")
                if isinstance(value.get("pilot_metrics"), dict)
                else None
            ),
            final_export=(
                value.get("final_export")
                if isinstance(value.get("final_export"), dict)
                else None
            ),
            measured_run=(
                value.get("measured_run")
                if isinstance(value.get("measured_run"), dict)
                else None
            ),
            measured_run_completed_at=(
                str(value["measured_run_completed_at"])
                if value.get("measured_run_completed_at") is not None
                else None
            ),
            latest_recheck=(
                value.get("latest_recheck")
                if isinstance(value.get("latest_recheck"), dict)
                else None
            ),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _preserves_stronger_attestation(
    previous: ValidationReport,
    current: ValidationReport,
    bundle_dir: Path,
) -> bool:
    plan: dict[str, Any] | None = None
    previous_rank = STATE_RANK.get(previous.state, 0)
    current_rank = STATE_RANK.get(current.state, 0)
    if previous_rank <= current_rank or current.state == ValidationState.INVALID:
        return False
    if (
        not current.artifact_fingerprint
        or previous.artifact_fingerprint != current.artifact_fingerprint
        or previous.bindings.get("bundle") != current.artifact_fingerprint
    ):
        return False
    for key in ("dataset", "plan_id", "candidate_id", "model_revision"):
        if previous.bindings.get(key) != current.bindings.get(key):
            return False
    historical_run = previous.state == ValidationState.MEASURED_RUN_PASS
    if not historical_run and previous.bindings.get(
        "environment"
    ) != current.bindings.get("environment"):
        return False
    if (
        not historical_run
        and previous_rank >= STATE_RANK[ValidationState.MODEL_DATA_PASS]
    ):
        if plan is None:
            try:
                loaded_plan = json.loads(
                    (bundle_dir / "plan.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                return False
            if not isinstance(loaded_plan, dict):
                return False
            plan = loaded_plan
        candidate = plan.get("recommended")
        if not isinstance(candidate, dict):
            return False
        world_size = candidate.get("world_size")
        device_indices = candidate.get("device_indices")
        if (
            not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or not isinstance(device_indices, list)
            or len(device_indices) != world_size
        ):
            return False
        try:
            current_hardware = _actual_hardware_binding(device_indices)
        except (RuntimeError, ValueError):
            return False
        if previous.bindings.get("hardware") != current_hardware:
            return False
    if previous_rank >= STATE_RANK[ValidationState.MEASURED_PREFLIGHT_PASS]:
        metrics_path = bundle_dir / "preflight-metrics.json"
        try:
            loaded_plan = json.loads(
                (bundle_dir / "plan.json").read_text(encoding="utf-8")
            )
            if not isinstance(loaded_plan, dict):
                return False
            plan = loaded_plan
            metrics = _read_preflight_metrics(metrics_path, plan)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if (
            previous.bindings.get("preflight_metrics") != sha256_file(metrics_path)
            or previous.preflight_metrics != metrics
        ):
            return False
    if previous_rank >= STATE_RANK[ValidationState.PILOT_PASS]:
        metrics_path = bundle_dir / "pilot-output" / "metrics.json"
        if not metrics_path.is_file() or previous.bindings.get(
            "pilot_metrics"
        ) != sha256_file(metrics_path):
            return False
    if previous.state == ValidationState.MEASURED_RUN_PASS:
        if plan is None:
            try:
                loaded_plan = json.loads(
                    (bundle_dir / "plan.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                return False
            if not isinstance(loaded_plan, dict):
                return False
            plan = loaded_plan
        if not _completed_run_evidence_is_current(previous, bundle_dir, plan):
            return False
    return True


def validate_bundle(
    bundle_dir: Path,
    *,
    level: ValidationLevel = "static",
    run: bool = False,
) -> ValidationReport:
    """Validate only the requested evidence level and never synthesize runtime passes."""

    if level not in LEVELS:
        raise ValueError(f"Unknown validation level: {level}")
    bundle_dir = bundle_dir.resolve()
    if not bundle_dir.is_dir():
        raise ValueError(f"Bundle directory does not exist: {bundle_dir}")
    report_path = bundle_dir / "validation-report.json"
    findings: list[ValidationFinding] = []
    checked: set[str] = set()
    runtime_evidence: list[str] = []
    portable_bindings: dict[str, str] = {}
    portable_preflight_metrics: dict[str, Any] | None = None

    for relative in REQUIRED_BUNDLE_FILES:
        path = bundle_dir / relative
        if path.is_file():
            checked.add(relative)
        else:
            findings.append(
                _finding(
                    "MISSING_FILE",
                    f"Required bundle file is missing: {relative}",
                    path=relative,
                )
            )

    plan = (
        _load_json(bundle_dir / "plan.json", findings, "PLAN_JSON_ERROR")
        if (bundle_dir / "plan.json").is_file()
        else None
    )
    if plan is not None:
        plan_contract_errors = validate_plan_payload(
            plan, root=bundle_dir, verify_dataset=True
        )
        for error in plan_contract_errors:
            findings.append(_finding("PLAN_CONTRACT_ERROR", error, path="plan.json"))
        if not plan_contract_errors:
            try:
                restored = training_plan_from_primitive(plan)
                replanned = plan_training(
                    model=restored.model,
                    dataset=restored.dataset,
                    hardware=restored.hardware,
                    target=restored.target,
                )
            except (KeyError, TypeError, ValueError) as error:
                findings.append(
                    _finding(
                        "PLANNER_PARITY_ERROR",
                        f"Could not reproduce the plan from its bound facts: {error}",
                        path="plan.json",
                    )
                )
            else:
                reproduced = to_primitive(replanned)
                if (
                    reproduced["candidates"] != plan.get("candidates")
                    or reproduced["recommended"] != plan.get("recommended")
                    or reproduced["plan_id"] != plan.get("plan_id")
                ):
                    findings.append(
                        _finding(
                            "PLANNER_PARITY_MISMATCH",
                            "Candidates or recommendation do not match deterministic Aptus v0.2 replanning.",
                            path="plan.json",
                        )
                    )

    manifest = (
        _load_json(bundle_dir / "bundle-manifest.json", findings, "MANIFEST_JSON_ERROR")
        if (bundle_dir / "bundle-manifest.json").is_file()
        else None
    )
    if isinstance(manifest, dict):
        if manifest.get("schema_version") != "aptus.bundle.v2":
            findings.append(
                _finding(
                    "MANIFEST_SCHEMA",
                    "Manifest schema must be aptus.bundle.v2.",
                    path="bundle-manifest.json",
                )
            )
        if (bundle_dir / "plan.json").is_file() and manifest.get(
            "plan_sha256"
        ) != sha256_file(bundle_dir / "plan.json"):
            findings.append(
                _finding(
                    "MANIFEST_PLAN_DIGEST",
                    "Manifest plan digest does not match plan.json.",
                    path="bundle-manifest.json",
                )
            )
        entries = manifest.get("files")
        if not isinstance(entries, list) or not entries:
            findings.append(
                _finding(
                    "MANIFEST_EMPTY",
                    "Manifest files must be a non-empty list.",
                    path="bundle-manifest.json",
                )
            )
        else:
            seen: set[str] = set()
            for item in entries:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    findings.append(
                        _finding(
                            "MANIFEST_ENTRY_INVALID",
                            "Every manifest entry requires a path.",
                            path="bundle-manifest.json",
                        )
                    )
                    continue
                relative = item["path"]
                if (
                    relative in seen
                    or Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                ):
                    findings.append(
                        _finding(
                            "MANIFEST_PATH_INVALID",
                            f"Unsafe or duplicate manifest path: {relative}",
                            path="bundle-manifest.json",
                        )
                    )
                    continue
                seen.add(relative)
                path = bundle_dir / relative
                if not path.is_file():
                    findings.append(
                        _finding(
                            "MANIFEST_FILE_MISSING",
                            f"Manifest file is absent: {relative}",
                            path=relative,
                        )
                    )
                    continue
                checked.add(relative)
                if (
                    item.get("sha256") != sha256_file(path)
                    or item.get("size_bytes") != path.stat().st_size
                ):
                    findings.append(
                        _finding(
                            "MANIFEST_MISMATCH",
                            f"Checksum or size mismatch: {relative}",
                            path=relative,
                        )
                    )
        for error in validate_bundle_manifest(bundle_dir):
            findings.append(
                _finding("MANIFEST_INTEGRITY", error, path="bundle-manifest.json")
            )

    if LEVELS.index(level) >= LEVELS.index("static"):
        for relative in (
            "plan_contract.py",
            "preflight.py",
            "run.py",
            "runtime_lease.py",
            "train.py",
            "validate.py",
        ):
            path = bundle_dir / relative
            if not path.is_file():
                continue
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except SyntaxError as error:
                findings.append(
                    _finding(
                        "PYTHON_PARSE_ERROR",
                        f"{error.msg} at line {error.lineno}.",
                        path=relative,
                    )
                )
        for relative in (
            "README.md",
            "decision-report.md",
            "runbook.md",
            "run.py",
            "runtime_lease.py",
            "train.py",
            "preflight.py",
            "validate.py",
        ):
            path = bundle_dir / relative
            if path.is_file() and any(
                marker in path.read_text(encoding="utf-8")
                for marker in ("{{", "}}", "TODO")
            ):
                findings.append(
                    _finding(
                        "UNRESOLVED_TEMPLATE",
                        "Generated file contains an unresolved marker.",
                        path=relative,
                    )
                )

    expected_requirements: tuple[str, ...] = ()
    if isinstance(plan, dict) and isinstance(plan.get("recommended"), dict):
        method = plan["recommended"].get("method")
        try:
            expected_requirements = bundle_requirements(method)
        except ValueError:
            expected_requirements = ()
        requirements_path = bundle_dir / "requirements.txt"
        if requirements_path.is_file():
            actual = tuple(
                line.strip()
                for line in requirements_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if actual != expected_requirements:
                findings.append(
                    _finding(
                        "DEPENDENCY_SET_MISMATCH",
                        "requirements.txt does not equal the method-specific direct pinned set.",
                        path="requirements.txt",
                    )
                )
        train_path = bundle_dir / "train.py"
        if train_path.is_file():
            source = train_path.read_text(encoding="utf-8")
            values = (
                plan.get("model", {}).get("model_id"),
                plan.get("dataset", {}).get("source_path"),
            )
            if any(
                isinstance(value, str) and value and value in source for value in values
            ):
                findings.append(
                    _finding(
                        "USER_VALUE_EMBEDDED_IN_SOURCE",
                        "Executable source contains a user model or dataset value.",
                        path="train.py",
                    )
                )
        trainer_path = bundle_dir / "config" / "trainer.json"
        if trainer_path.is_file():
            trainer = _load_json(trainer_path, findings, "TRAINER_CONFIG_JSON_ERROR")
            candidate = plan["recommended"]
            target = plan.get("target", {})
            expected = {
                "task": target.get("task"),
                "sequence_length": target.get("sequence_length"),
                "packing": target.get("packing"),
                "per_device_train_batch_size": candidate.get("micro_batch_size"),
                "gradient_accumulation_steps": candidate.get(
                    "gradient_accumulation_steps"
                ),
                "effective_global_batch_size": candidate.get("effective_batch_size"),
                "world_size": candidate.get("world_size"),
                "precision": candidate.get("precision"),
            }
            if isinstance(trainer, dict):
                for key, value in expected.items():
                    if trainer.get(key) != value:
                        findings.append(
                            _finding(
                                "TRAINER_CONFIG_MISMATCH",
                                f"config/trainer.json {key} does not match plan.json.",
                                path="config/trainer.json",
                            )
                        )

    structural_errors = any(item.severity == "error" for item in findings)
    achieved_level: ValidationLevel = "contract"
    if LEVELS.index(level) >= LEVELS.index("static"):
        achieved_level = "static"
    runtime_level = LEVELS.index(level) >= LEVELS.index("dependency")
    if runtime_level and not run:
        findings.append(
            _finding(
                "RUNTIME_NOT_EXECUTED",
                f"{level} was requested without run=true; report remains at static-pass.",
                severity="warning",
            )
        )
    elif runtime_level and not structural_errors:
        command = [sys.executable, str(bundle_dir / "validate.py"), "--level", level]
        with tempfile.TemporaryFile() as runtime_log:
            completed = subprocess.run(
                command,
                cwd=bundle_dir,
                stdout=runtime_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            runtime_log.seek(0, os.SEEK_END)
            length = runtime_log.tell()
            runtime_log.seek(max(0, length - 16_000))
            output_tail = runtime_log.read().decode("utf-8", errors="replace")
        runtime_evidence.extend(
            (
                "command=" + json.dumps(command),
                f"return_code={completed.returncode}",
                "output_tail=" + output_tail,
            )
        )
        if completed.returncode:
            findings.append(
                _finding(
                    "RUNTIME_VALIDATION_FAILED",
                    f"{level} validation exited {completed.returncode}.",
                )
            )
        else:
            runtime_attestation_valid = True
            portable_report_path = bundle_dir / "validation-report.json"
            try:
                portable_report = json.loads(
                    portable_report_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                portable_report = None
            if not isinstance(portable_report, dict) or not isinstance(
                portable_report.get("bindings"), dict
            ):
                findings.append(
                    _finding(
                        "RUNTIME_ATTESTATION_INVALID",
                        "Runtime validation did not publish a readable bound validation report.",
                        path="validation-report.json",
                    )
                )
                runtime_attestation_valid = False
            else:
                portable_bindings = {
                    str(key): str(value)
                    for key, value in portable_report["bindings"].items()
                }
            if LEVELS.index(level) >= LEVELS.index("measured-preflight") and isinstance(
                plan, dict
            ):
                metrics_path = bundle_dir / "preflight-metrics.json"
                try:
                    measured_metrics = _read_preflight_metrics(metrics_path, plan)
                except ValueError as error:
                    findings.append(
                        _finding(
                            "PREFLIGHT_METRICS_INVALID",
                            str(error),
                            path="preflight-metrics.json",
                        )
                    )
                    runtime_attestation_valid = False
                else:
                    expected_digest = sha256_file(metrics_path)
                    if (
                        portable_bindings.get("preflight_metrics") != expected_digest
                        or not isinstance(portable_report, dict)
                        or portable_report.get("preflight_metrics") != measured_metrics
                    ):
                        findings.append(
                            _finding(
                                "PREFLIGHT_METRICS_UNBOUND",
                                "Runtime validation report does not bind the exact measured-preflight metrics.",
                                path="validation-report.json",
                            )
                        )
                        runtime_attestation_valid = False
                    else:
                        portable_preflight_metrics = measured_metrics
            if runtime_attestation_valid:
                achieved_level = level

    has_errors = any(item.severity == "error" for item in findings)
    state = ValidationState.INVALID if has_errors else LEVEL_STATES[achieved_level]
    try:
        fingerprint = bundle_fingerprint(bundle_dir)
    except FileNotFoundError:
        fingerprint = ""
    validated_at = datetime.now(timezone.utc).isoformat()
    data_digest = (
        plan.get("dataset", {}).get("source_sha256", "")
        if isinstance(plan, dict)
        else ""
    )
    hardware_value = plan.get("hardware", {}) if isinstance(plan, dict) else {}
    planned_hardware = _json_hash(hardware_value)
    bindings = {
        "bundle": fingerprint,
        "dataset": str(data_digest),
        "environment": portable_bindings.get(
            "environment", _environment_binding(expected_requirements)
        ),
        "hardware": portable_bindings.get("hardware", planned_hardware),
        "planned_hardware": planned_hardware,
        "validator": "aptus-validator-v2",
        "validated_at": validated_at,
    }
    if isinstance(plan, dict):
        bindings["plan_id"] = str(plan.get("plan_id", ""))
        bindings["candidate_id"] = str(
            plan.get("recommended", {}).get("candidate_id", "")
        )
        bindings["model_revision"] = str(plan.get("model", {}).get("revision", ""))
    preflight_metrics_path = bundle_dir / "preflight-metrics.json"
    if (
        LEVELS.index(achieved_level) >= LEVELS.index("measured-preflight")
        and preflight_metrics_path.is_file()
    ):
        bindings["preflight_metrics"] = portable_bindings.get(
            "preflight_metrics", sha256_file(preflight_metrics_path)
        )
    pilot_metrics = bundle_dir / "pilot-output" / "metrics.json"
    if achieved_level == "pilot" and pilot_metrics.is_file():
        bindings["pilot_metrics"] = portable_bindings.get(
            "pilot_metrics", sha256_file(pilot_metrics)
        )
    pilot_metrics_payload: dict[str, Any] | None = None
    if achieved_level == "pilot" and pilot_metrics.is_file():
        try:
            loaded_pilot_metrics = json.loads(pilot_metrics.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded_pilot_metrics = None
        if isinstance(loaded_pilot_metrics, dict):
            pilot_metrics_payload = loaded_pilot_metrics
    report = ValidationReport(
        state=state,
        findings=tuple(findings),
        checked_files=tuple(sorted(checked)),
        artifact_fingerprint=fingerprint,
        smoke_command=(sys.executable, "validate.py", "--level", "measured-preflight"),
        runtime_evidence=tuple(runtime_evidence),
        validation_level=achieved_level,
        bindings=bindings,
        validated_at=validated_at,
        preflight_metrics=portable_preflight_metrics,
        pilot_metrics=pilot_metrics_payload,
    )
    with _report_lock(bundle_dir):
        latest_report = _read_report(report_path) if report_path.is_file() else None
        if latest_report is not None and _preserves_stronger_attestation(
            latest_report, report, bundle_dir
        ):
            if latest_report.state == ValidationState.MEASURED_RUN_PASS:
                latest_report = replace(
                    latest_report,
                    latest_recheck={
                        "state": report.state.value,
                        "validation_level": report.validation_level,
                        "validated_at": report.validated_at,
                        "artifact_fingerprint": report.artifact_fingerprint,
                        "findings": to_primitive(report.findings),
                    },
                )
                _write_report(report_path, latest_report)
            return latest_report
        _write_report(report_path, report)
        return report
