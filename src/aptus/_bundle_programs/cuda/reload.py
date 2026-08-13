#!/usr/bin/env python3
"""Reload one CUDA PEFT adapter in a fresh process and perform bounded generation."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_files(adapter_path: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(adapter_path).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(item for item in adapter_path.rglob("*") if item.is_file())
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--final-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-parent-pid", type=int, required=True)
    parser.add_argument("--bundle-root", type=Path, default=None)
    arguments = parser.parse_args()
    if arguments.expected_parent_pid <= 0 or os.getppid() != arguments.expected_parent_pid:
        raise RuntimeError("Reload verifier is not the expected fresh child process.")

    bundle_root = (arguments.bundle_root or ROOT).resolve()
    sys.path.insert(0, str(bundle_root))
    from plan_contract import load_json_object, validate_bundle_manifest, validate_plan_payload

    plan = load_json_object(bundle_root / "plan.json", "Bundle plan")
    errors = validate_plan_payload(plan, root=bundle_root, verify_dataset=True)
    errors += validate_bundle_manifest(bundle_root)
    if errors:
        raise RuntimeError("Invalid Aptus bundle: " + " | ".join(errors))
    candidate = plan["recommended"]
    if candidate["method"] == "full":
        raise RuntimeError("CUDA adapter reload does not apply to full-parameter export.")

    export = json.loads(arguments.final_export.resolve(strict=True).read_text(encoding="utf-8"))
    if (
        not isinstance(export, dict)
        or export.get("schema_version") != "aptus.final-export.v1"
        or export.get("method") != candidate["method"]
        or export.get("base_model", {}).get("model_id") != plan["model"]["model_id"]
        or export.get("base_model", {}).get("revision") != plan["model"]["revision"]
    ):
        raise RuntimeError("Reload final-export does not bind the planned adapter identity.")

    adapter_path = arguments.adapter_path.resolve(strict=True)
    observed = _adapter_files(adapter_path)
    expected_files = export.get("files")
    if not isinstance(expected_files, list) or observed != expected_files:
        raise RuntimeError("Reload adapter does not match its immutable final-export manifest.")

    output = arguments.output.resolve()
    if output.exists():
        raise RuntimeError("Reload evidence path is not a fresh file.")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA adapter reload requires a visible CUDA device.")

    model_id = plan["model"]["model_id"]
    revision = plan["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"].get("tokenizer_id") or model_id,
        revision=revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer must define an EOS or padding token.")
        tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.reset_peak_memory_stats()
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.to("cuda")
    model.eval()
    prompt = "Aptus adapter reload verification:"
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=4,
            do_sample=False,
        )
    prompt_length = int(encoded["input_ids"].shape[1])
    new_tokens = generated[0, prompt_length:]
    generation_tokens = int(new_tokens.shape[0])
    if generation_tokens < 1 or generation_tokens > 4:
        raise RuntimeError("Fresh-process adapter generation exceeded its token bound.")
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    peak = int(torch.cuda.max_memory_allocated())
    if peak <= 0:
        raise RuntimeError("Fresh-process adapter reload reported no positive CUDA peak.")

    evidence = {
        "schema_version": "aptus.cuda-reload-evidence.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": revision,
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "method": candidate["method"],
        "training_runtime": "transformers-peft-cuda",
        "compute_backend": "cuda",
        "resume_supported": False,
        "fresh_process_observed": True,
        "parent_pid": os.getppid(),
        "verifier_pid": os.getpid(),
        "adapter_export_sha256": hashlib.sha256(
            json.dumps(observed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "generation_max_tokens": 4,
        "generation_tokens": generation_tokens,
        "generation_text_sha256": hashlib.sha256(
            generated_text.encode("utf-8")
        ).hexdigest(),
        "measured_peak_bytes": peak,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(f"Fresh-process CUDA adapter reload passed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
