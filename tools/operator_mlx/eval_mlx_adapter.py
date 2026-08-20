"""Generate CompletionsDataset-shaped predictions and score with aptus eval.

Matches mlx-lm training: user=prompt, assistant=completion, greedy.
Writes only a ``prediction`` field so a copied gold ``completion`` cannot
be scored as a match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aptus.evaluation import build_evaluation_contract, evaluate_predictions

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
REVISION = "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed"
MAX_TOKENS = 256
SEED = 17


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _prompt_and_gold(row: dict) -> tuple[str, str]:
    if "messages" in row:
        messages = row["messages"]
        return messages[0]["content"], messages[-1]["content"]
    return row["prompt"], row["completion"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def generate_predictions(gold_path: Path, adapter: Path, output: Path) -> None:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    rows = _rows(gold_path)
    mx.random.seed(SEED)
    model, tokenizer = load(MODEL_ID, adapter_path=str(adapter), revision=REVISION)
    sampler = make_sampler(temp=0.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            prompt, _gold = _prompt_and_gold(row)
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            text = generate(
                model,
                tokenizer,
                prompt=formatted,
                max_tokens=MAX_TOKENS,
                sampler=sampler,
                verbose=False,
            )
            record = {
                "id": row.get("id") or f"row-{index}",
                "prediction": text,
                "prompt": prompt,
                "split_group": row.get("split_group"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index + 1}/{len(rows)} chars={len(text)}", flush=True)


def ensure_gold_ids(path: Path, output: Path) -> Path:
    rows = _rows(path)
    changed = False
    for index, row in enumerate(rows):
        if not row.get("id"):
            row["id"] = f"row-{index}"
            changed = True
    if not changed and path.resolve() == output.resolve():
        return path
    if output.exists() and output.resolve() != path.resolve():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Greedy MLX recitation eval against a gold JSONL."
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--plan-id")
    parser.add_argument("--candidate-id")
    parser.add_argument("--job-id")
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    gold = ensure_gold_ids(args.gold, workdir / "gold-with-ids.jsonl")
    predictions = workdir / "gold-predictions.jsonl"
    if not args.skip_generate:
        if predictions.exists():
            raise FileExistsError(f"Refusing to overwrite {predictions}")
        generate_predictions(gold, args.adapter, predictions)
    adapter_weights = args.adapter / "adapters.safetensors"
    export_digest = _sha256(adapter_weights) if adapter_weights.is_file() else None
    contract = build_evaluation_contract(
        dataset_path=gold,
        claim=(
            "Exact-match of operator gold completions on this adapter. "
            "Not general model quality."
        ),
        threshold=0.000001,
        gold_field="completion",
        id_field="id",
        plan_id=args.plan_id,
        candidate_id=args.candidate_id,
        job_id=args.job_id,
        export_digest=export_digest,
        export_kind="adapter" if export_digest else None,
    )
    contract_path = workdir / "gold-contract.json"
    if contract_path.exists():
        raise FileExistsError(f"Refusing to overwrite {contract_path}")
    contract_path.write_text(
        json.dumps(contract.to_primitive(), indent=2) + "\n", encoding="utf-8"
    )
    result = evaluate_predictions(
        contract,
        gold,
        predictions,
        expected_export_digest=export_digest,
    )
    result_path = workdir / "gold-result.json"
    result_path.write_text(
        json.dumps(result.to_primitive(), indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": result.decision,
                "score": result.score,
                "n_gold": result.n_gold,
                "n_scored": result.n_scored,
                "reasons": list(result.decision_reasons),
                "result": str(result_path),
            },
            indent=2,
        )
    )
    return 0 if result.decision != "abstain" else 2


if __name__ == "__main__":
    raise SystemExit(main())
