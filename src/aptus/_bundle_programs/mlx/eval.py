#!/usr/bin/env python3
"""Generate CompletionsDataset-shaped predictions for an optional exact-match eval.

This portable program does not import the Aptus package. It writes only a ``prediction``
field so a copied gold ``completion`` cannot be scored as a match.
Exact-match is not general model quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
from plan_contract import load_json_object, validate_bundle_manifest, validate_plan_payload
from train import download_pinned_model, require_method_model


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object.")
        rows.append(parsed)
    if not rows:
        raise ValueError(f"{path} is empty.")
    return rows


def prompt_and_gold(row: dict[str, Any]) -> tuple[str, str]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        user = next(
            (
                item.get("content")
                for item in messages
                if isinstance(item, dict) and item.get("role") == "user"
            ),
            None,
        )
        assistant = next(
            (
                item.get("content")
                for item in reversed(messages)
                if isinstance(item, dict) and item.get("role") == "assistant"
            ),
            None,
        )
        if isinstance(user, str) and isinstance(assistant, str):
            return user, assistant
        raise ValueError("messages rows need user and assistant content.")
    prompt, completion = row.get("prompt"), row.get("completion")
    if isinstance(prompt, str) and isinstance(completion, str):
        return prompt, completion
    raise ValueError("Each gold row needs prompt/completion or messages.")


def prediction_record(row: dict[str, Any], index: int, text: str) -> dict[str, Any]:
    prompt, _gold = prompt_and_gold(row)
    record: dict[str, Any] = {
        "id": row["id"] if row.get("id") else f"row-{index}",
        "prediction": text,
        "prompt": prompt,
    }
    if row.get("split_group"):
        record["split_group"] = row["split_group"]
    return record


def require_adapter(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    weights = resolved / "adapters.safetensors"
    if not weights.is_file():
        raise RuntimeError(f"Adapter directory has no adapters.safetensors: {resolved}")
    return resolved


def generate_predictions(
    *,
    gold_path: Path,
    adapter_path: Path,
    output: Path,
    max_tokens: int,
    seed: int,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    plan = load_json_object(ROOT / "plan.json", "Bundle plan")
    errors = validate_plan_payload(plan, root=ROOT, verify_dataset=True)
    errors += validate_bundle_manifest(ROOT)
    if errors:
        raise RuntimeError("Invalid Aptus bundle: " + " | ".join(errors))
    candidate = plan["recommended"]
    model_path = download_pinned_model(plan, plan["model"]["model_id"])
    require_method_model(plan, candidate, model_path)
    adapter = require_adapter(adapter_path)
    rows = load_jsonl(gold_path)
    mx.random.seed(seed)
    model, tokenizer = load(
        str(model_path),
        adapter_path=str(adapter),
        tokenizer_config={"trust_remote_code": False},
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Pinned tokenizer has no apply_chat_template.")
    sampler = make_sampler(temp=0.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            prompt, _gold = prompt_and_gold(row)
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            text = generate(
                model,
                tokenizer,
                prompt=formatted,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
            handle.write(
                json.dumps(prediction_record(row, index, text), ensure_ascii=False)
                + "\n"
            )
            handle.flush()
            print(f"{index + 1}/{len(rows)} chars={len(text)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Greedy MLX recitation generate. Writes prediction-only JSONL. "
            "Not a quality claim."
        )
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args()
    if arguments.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    generate_predictions(
        gold_path=arguments.gold,
        adapter_path=arguments.adapter,
        output=arguments.output,
        max_tokens=arguments.max_tokens,
        seed=arguments.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
