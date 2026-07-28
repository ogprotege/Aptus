#!/usr/bin/env python3
"""Portable parent runner for an Aptus full-training bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / "__pycache__").exists():
    raise RuntimeError(
        "Bundle contains an unmanifested __pycache__; remove it before execution."
    )
sys.dont_write_bytecode = True
from runtime_lease import portable_execution_lease, run_with_lease


def bind_visible_cuda_devices(plan: dict) -> None:
    candidate = plan["recommended"]
    world_size = int(candidate["world_size"])
    device_indices = candidate.get("device_indices", list(range(world_size)))
    if (
        not isinstance(device_indices, list)
        or len(device_indices) != world_size
        or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in device_indices
        )
        or len(set(device_indices)) != len(device_indices)
    ):
        raise RuntimeError("Selected CUDA device indices do not match the planned world.")
    marker = os.environ.get("APTUS_BOUND_DEVICE_CANDIDATE")
    if marker is not None:
        if marker != candidate["candidate_id"]:
            raise RuntimeError("Inherited Aptus CUDA visibility belongs to another candidate.")
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        inherited_tokens = (
            [token.strip() for token in inherited.split(",") if token.strip()]
            if inherited is not None
            else []
        )
        if len(inherited_tokens) != world_size or any(
            token.lower() in {"-1", "nodevfiles", "none"}
            for token in inherited_tokens
        ):
            raise RuntimeError("Inherited Aptus CUDA visibility is missing or malformed.")
        return
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing is not None:
        visible_tokens = [token.strip() for token in existing.split(",") if token.strip()]
        if not visible_tokens or any(
            token.lower() in {"-1", "nodevfiles", "none"}
            for token in visible_tokens
        ):
            raise RuntimeError("CUDA_VISIBLE_DEVICES exposes no selectable CUDA devices.")
        if any(index >= len(visible_tokens) for index in device_indices):
            raise RuntimeError("Selected CUDA device index is outside CUDA_VISIBLE_DEVICES.")
        selected_tokens = [visible_tokens[index] for index in device_indices]
    else:
        selected_tokens = [str(index) for index in device_indices]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_tokens)
    os.environ["APTUS_BOUND_DEVICE_CANDIDATE"] = candidate["candidate_id"]


def require_runs_root() -> Path:
    runs_root = ROOT / "runs"
    if runs_root.exists() and (runs_root.is_symlink() or not runs_root.is_dir()):
        raise RuntimeError("The Aptus runs path must be a real directory.")
    runs_root.mkdir(mode=0o700, exist_ok=True)
    if runs_root.is_symlink() or runs_root.resolve() != ROOT / "runs":
        raise RuntimeError("The Aptus runs directory escapes the bundle root.")
    return runs_root.resolve()


def normalize_run_output(output_dir: Path, *, fresh: bool) -> Path:
    runs_root = require_runs_root()
    unresolved = output_dir.expanduser()
    if unresolved.is_symlink():
        raise RuntimeError("Aptus run output cannot be a symlink.")
    resolved = unresolved.resolve()
    if resolved.parent != runs_root or not resolved.name.startswith("run_"):
        raise RuntimeError("Run output must be a ROOT/runs/run_* child.")
    if fresh and resolved.exists():
        raise RuntimeError(f"Aptus run output already exists: {resolved}")
    return resolved


def recover_pending_run() -> int | None:
    report_path = ROOT / "validation-report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pending_names = {
        "measured_run_pending_at",
        "pending_final_export",
        "pending_measured_run",
    }
    if not isinstance(report, dict) or not any(
        name in report for name in pending_names
    ):
        return None
    if report.get("state") != "execution-approved":
        raise RuntimeError("Pending run evidence exists outside execution-approved state.")
    active = report.get("active_run")
    if not isinstance(active, dict) or not isinstance(active.get("output_dir"), str):
        raise RuntimeError("Pending run evidence lacks a bound active output directory.")
    resolved = normalize_run_output(Path(active["output_dir"]), fresh=False)
    promoted = run_with_lease(
        [sys.executable, str(ROOT / "train.py"), "--promote-pending", str(resolved)],
        cwd=ROOT,
    )
    if promoted.returncode:
        return promoted.returncode
    print(f"Recovered and attested the completed pending Aptus run: {resolved}")
    return 0


def launch_full_training(arguments: argparse.Namespace) -> int:
    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    bind_visible_cuda_devices(plan)
    recovered = recover_pending_run()
    if recovered is not None:
        return recovered
    output_dir = arguments.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = ROOT / "runs" / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    output_dir = normalize_run_output(output_dir, fresh=True)
    train_arguments = [
        str(ROOT / "train.py"),
        "--confirm-full-train",
        "--output-dir",
        str(output_dir),
    ]
    if arguments.local_files_only:
        train_arguments.append("--local-files-only")
    if plan["recommended"]["distribution"] == "single":
        command = [sys.executable, *train_arguments]
    else:
        command = [
            sys.executable,
            "-m",
            "accelerate.commands.accelerate_cli",
            "launch",
            "--config_file",
            str(ROOT / "config" / "accelerate.yaml"),
            *train_arguments,
        ]
    completed = run_with_lease(command, cwd=ROOT)
    if completed.returncode:
        return completed.returncode
    promoted = run_with_lease(
        [sys.executable, str(ROOT / "train.py"), "--promote-pending", str(output_dir)],
        cwd=ROOT,
    )
    if promoted.returncode:
        return promoted.returncode
    print(f"Aptus measured run completed and was attested: {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch, wait for, verify, and attest one Aptus full-training run."
    )
    parser.add_argument("--confirm-full-train", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.confirm_full_train:
        parser.error("Full training requires --confirm-full-train.")
    with portable_execution_lease(ROOT, action="train"):
        return launch_full_training(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
