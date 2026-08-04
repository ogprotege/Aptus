#!/usr/bin/env python3
"""Reload one MLX adapter in a fresh process and perform bounded generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
from plan_contract import load_json_object, validate_bundle_manifest, validate_plan_payload
from train import (
    build_mlx_model_load_binding,
    download_pinned_model,
    require_method_model,
    require_mlx_model_load_binding,
    require_unified_memory_admission,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-parent-pid", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.expected_parent_pid <= 0 or os.getppid() != arguments.expected_parent_pid:
        raise RuntimeError("Reload verifier is not the expected fresh child process.")
    plan = load_json_object(ROOT / "plan.json", "Bundle plan")
    errors = validate_plan_payload(plan, root=ROOT, verify_dataset=True)
    errors += validate_bundle_manifest(ROOT)
    if errors:
        raise RuntimeError("Invalid Aptus bundle: " + " | ".join(errors))
    metrics_path = arguments.training_metrics.resolve(strict=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    candidate = plan["recommended"]
    if (
        not isinstance(metrics, dict)
        or metrics.get("plan_id") != plan["plan_id"]
        or metrics.get("candidate_id") != candidate["candidate_id"]
        or metrics.get("action") not in {"pilot", "full"}
        or metrics.get("execution_semantics") != "uninterrupted"
        or metrics.get("resume_supported") is not False
    ):
        raise RuntimeError("Reload verifier received unbound training metrics.")
    adapter_path = arguments.adapter_path.resolve(strict=True)
    output_root = metrics_path.parent.resolve()
    if output_root not in adapter_path.parents:
        raise RuntimeError("Reload adapter escapes the owned run root.")
    expected_manifest = metrics.get("adapter_manifest")
    observed_manifest = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(item for item in adapter_path.iterdir() if item.is_file())
    ]
    if not isinstance(expected_manifest, list) or observed_manifest != expected_manifest:
        raise RuntimeError("Reload adapter does not match its immutable training manifest.")
    output = arguments.output.resolve()
    if output.parent != output_root or output.exists():
        raise RuntimeError("Reload evidence path is not a fresh file in the owned run root.")

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.utils import load_adapters

    model_path = download_pinned_model(plan, plan["model"]["model_id"])
    architecture_contract = require_method_model(plan, candidate, model_path)
    admission = require_unified_memory_admission(plan, model_path)
    mx.reset_peak_memory()
    model, tokenizer = load(
        str(model_path),
        tokenizer_config={"trust_remote_code": False},
    )
    loaded_binding = build_mlx_model_load_binding(
        model,
        plan,
        observed_safetensors_bytes=admission["observed_safetensors_bytes"],
        architecture_contract=architecture_contract,
    )
    recorded_binding = require_mlx_model_load_binding(
        plan, metrics.get("model_load_binding")
    )
    if loaded_binding != recorded_binding:
        raise RuntimeError(
            "Fresh-process MLX model census differs from the training-time model-load binding."
        )
    model = load_adapters(model, str(adapter_path))
    model.eval()
    responses = list(
        stream_generate(
            model,
            tokenizer,
            "Aptus adapter reload verification:",
            max_tokens=4,
        )
    )
    if not responses:
        raise RuntimeError("Fresh-process adapter generation returned no response evidence.")
    generation_tokens = int(responses[-1].generation_tokens)
    if generation_tokens < 1 or generation_tokens > 4:
        raise RuntimeError("Fresh-process adapter generation exceeded its token bound.")
    generated_text = "".join(str(response.text) for response in responses)
    peak = int(mx.get_peak_memory())
    if peak <= 0:
        raise RuntimeError("Fresh-process adapter reload reported no positive MLX peak.")
    evidence = {
        "schema_version": "aptus.mlx-reload-evidence.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "method": candidate["method"],
        "training_runtime": "mlx-lm",
        "compute_backend": "mps",
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "fresh_process_observed": True,
        "parent_pid": os.getppid(),
        "verifier_pid": os.getpid(),
        "adapter_manifest_sha256": hashlib.sha256(
            json.dumps(
                observed_manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "generation_max_tokens": 4,
        "generation_tokens": generation_tokens,
        "generation_text_sha256": hashlib.sha256(
            generated_text.encode("utf-8")
        ).hexdigest(),
        "measured_peak_bytes": peak,
        "unified_memory_admission": admission,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(f"Fresh-process MLX adapter reload passed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
