from __future__ import annotations

import ast
import json
from pathlib import Path

from .catalog import bundle_requirements
from .domain import (
    ValidationFinding,
    ValidationReport,
    ValidationState,
    to_primitive,
)
from .plan_contract import bundle_fingerprint, validate_plan_payload


REQUIRED_BUNDLE_FILES = (
    "README.md",
    "plan.json",
    "plan_contract.py",
    "requirements.txt",
    "train.py",
    "validate.py",
)


def _finding(
    code: str,
    message: str,
    *,
    path: str | None = None,
    severity: str = "error",
) -> ValidationFinding:
    return ValidationFinding(
        code=code,
        message=message,
        severity=severity,
        path=path,
    )


def validate_bundle(bundle_dir: Path) -> ValidationReport:
    bundle_dir = bundle_dir.resolve()
    findings: list[ValidationFinding] = []
    checked_files: list[str] = []

    for filename in REQUIRED_BUNDLE_FILES:
        path = bundle_dir / filename
        if not path.is_file():
            findings.append(
                _finding(
                    "MISSING_FILE",
                    f"Required bundle file is missing: {filename}",
                    path=filename,
                )
            )
        else:
            checked_files.append(filename)

    for filename in ("plan_contract.py", "train.py", "validate.py"):
        path = bundle_dir / filename
        if not path.is_file():
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=filename)
        except SyntaxError as error:
            findings.append(
                _finding(
                    "PYTHON_PARSE_ERROR",
                    f"{error.msg} at line {error.lineno}.",
                    path=filename,
                )
            )

    plan: object = None
    plan_path = bundle_dir / "plan.json"
    if plan_path.is_file():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            findings.append(
                _finding(
                    "PLAN_JSON_ERROR",
                    str(error),
                    path="plan.json",
                )
            )
    if plan_path.is_file():
        for error in validate_plan_payload(plan, verify_dataset=True):
            findings.append(
                _finding(
                    "PLAN_CONTRACT_ERROR",
                    error,
                    path="plan.json",
                )
            )

    for filename in ("README.md", "train.py", "validate.py"):
        path = bundle_dir / filename
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "{{" in text or "}}" in text or "TODO" in text:
            findings.append(
                _finding(
                    "UNRESOLVED_TEMPLATE",
                    "Generated file contains a template marker or TODO.",
                    path=filename,
                )
            )

    requirements_path = bundle_dir / "requirements.txt"
    if requirements_path.is_file():
        actual_requirements = tuple(
            line.strip()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        method = (
            plan.get("recommended", {}).get("method")
            if isinstance(plan, dict)
            else None
        )
        if method in {"lora", "qlora"}:
            expected_requirements = bundle_requirements(method)
            if actual_requirements != expected_requirements:
                findings.append(
                    _finding(
                        "DEPENDENCY_SET_MISMATCH",
                        "Requirements must exactly match the tested dependency "
                        f"set for {method}: {expected_requirements}.",
                        path="requirements.txt",
                    )
                )

    train_path = bundle_dir / "train.py"
    if isinstance(plan, dict) and train_path.is_file():
        train_source = train_path.read_text(encoding="utf-8")
        embedded_values = (
            plan.get("model", {}).get("model_id"),
            plan.get("dataset", {}).get("source_path"),
        )
        if any(value and value in train_source for value in embedded_values):
            findings.append(
                _finding(
                    "USER_VALUE_EMBEDDED_IN_SOURCE",
                    "Model or dataset input was embedded in executable source.",
                    path="train.py",
                )
            )

    try:
        current_fingerprint = bundle_fingerprint(bundle_dir)
    except FileNotFoundError:
        current_fingerprint = ""

    previous_state = ValidationState.STATIC_PASS
    runtime_evidence: tuple[str, ...] = ()
    previous_report_path = bundle_dir / "validation-report.json"
    if previous_report_path.is_file():
        try:
            previous_report = json.loads(
                previous_report_path.read_text(encoding="utf-8")
            )
            if (
                current_fingerprint
                and previous_report.get("artifact_fingerprint")
                == current_fingerprint
            ):
                recorded_state = previous_report.get("state")
                if recorded_state in {
                    ValidationState.ENVIRONMENT_PASS.value,
                    ValidationState.SMOKE_PASS.value,
                }:
                    previous_state = ValidationState(recorded_state)
                recorded_evidence = previous_report.get("runtime_evidence", [])
                if isinstance(recorded_evidence, list) and all(
                    isinstance(item, str) for item in recorded_evidence
                ):
                    runtime_evidence = tuple(recorded_evidence)
        except (json.JSONDecodeError, ValueError):
            pass

    has_errors = any(finding.severity == "error" for finding in findings)
    report = ValidationReport(
        state=ValidationState.INVALID if has_errors else previous_state,
        findings=tuple(findings),
        checked_files=tuple(sorted(checked_files)),
        artifact_fingerprint=current_fingerprint,
        smoke_command=("python", "train.py", "--smoke"),
        runtime_evidence=runtime_evidence,
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "validation-report.json").write_text(
        json.dumps(to_primitive(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
