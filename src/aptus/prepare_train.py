"""Order a prompt/completion JSONL so MLX compile keeps named rows in train.

This is an Aptus operator command, not a quality claim.


Aptus MLX takes the last round(n * evaluation_fraction) rows as valid
(see ``src/aptus/generation.py``). Concatenating gold at the end of the
file parks those rows in valid. Exact-match on them is then 0 even when
the adapter recites train.

This script mixes ``--include`` rows into the compiled-train prefix and
puts only non-include rows in the tail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object.")
        prompt, completion = parsed.get("prompt"), parsed.get("completion")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise ValueError(
                f"{path}:{line_number} needs string prompt and completion."
            )
        if not prompt.strip() or not completion.strip():
            raise ValueError(
                f"{path}:{line_number} needs non-empty prompt and completion."
            )
        rows.append(parsed)
    if not rows:
        raise ValueError(f"{path} is empty.")
    return rows


def _key(row: dict[str, Any]) -> str:
    return str(row["prompt"])


def _slim(row: dict[str, Any]) -> dict[str, Any]:
    out = {
        "prompt": row["prompt"],
        "completion": row["completion"],
    }
    if row.get("split_group"):
        out["split_group"] = row["split_group"]
    if row.get("id"):
        out["id"] = row["id"]
    return out


def mlx_valid_count(example_count: int, evaluation_fraction: float) -> int:
    if example_count < 2:
        raise ValueError("Need at least two rows for a disjoint MLX split.")
    if not 0 < evaluation_fraction < 1:
        raise ValueError("evaluation_fraction must be in (0, 1).")
    return max(1, round(example_count * evaluation_fraction))


def order_rows_for_mlx_split(
    corpus: list[dict[str, Any]],
    include: list[dict[str, Any]],
    *,
    evaluation_fraction: float = 0.1,
    seed: int = 20260820,
) -> list[dict[str, Any]]:
    """Return corpus ordered so every include prompt sits before valid_start."""

    by_prompt: dict[str, dict[str, Any]] = {}
    for row in corpus:
        key = _key(row)
        if key in by_prompt:
            raise ValueError("Duplicate prompt in corpus; Aptus will not collapse it.")
        by_prompt[key] = _slim(row)
    missing: list[str] = []
    include_keys: list[str] = []
    seen_include: set[str] = set()
    for row in include:
        key = _key(row)
        if key not in by_prompt:
            missing.append(key[:80])
            continue
        if key not in seen_include:
            include_keys.append(key)
            seen_include.add(key)
        by_prompt[key] = _slim(row)
    if missing:
        raise ValueError(
            "Include rows not in corpus (" + str(len(missing)) + "): " + missing[0]
        )
    ordered_corpus = list(by_prompt.values())
    n = len(ordered_corpus)
    valid_count = mlx_valid_count(n, evaluation_fraction)
    train_count = n - valid_count
    if len(include_keys) > train_count:
        raise ValueError(
            f"Cannot keep {len(include_keys)} include rows in a {train_count}-row "
            "compiled train prefix."
        )
    include_set = set(include_keys)
    rest = [row for row in ordered_corpus if _key(row) not in include_set]
    rng = random.Random(seed)
    head = [_slim(by_prompt[key]) for key in include_keys] + rest[
        : train_count - len(include_keys)
    ]
    rng.shuffle(head)
    tail = rest[train_count - len(include_keys) :]
    ordered = head + tail
    if len(ordered) != n:
        raise ValueError("Ordered row count drifted from the corpus.")
    prefix = {_key(row) for row in ordered[:train_count]}
    if not include_set <= prefix:
        raise ValueError("Include rows leaked into the MLX valid tail.")
    return ordered


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_train_file(
    *,
    corpus: Path,
    include: Path,
    output: Path,
    evaluation_fraction: float = 0.1,
    seed: int = 20260820,
    manifest: Path | None = None,
) -> dict[str, Any]:
    """Write the ordered train JSONL and return the split manifest."""

    corpus_rows = load_jsonl(corpus)
    include_rows = load_jsonl(include)
    ordered = order_rows_for_mlx_split(
        corpus_rows,
        include_rows,
        evaluation_fraction=evaluation_fraction,
        seed=seed,
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    write_jsonl(output, ordered)
    valid_count = mlx_valid_count(len(ordered), evaluation_fraction)
    include_keys = {_key(row) for row in include_rows}
    prefix_keys = {_key(row) for row in ordered[: len(ordered) - valid_count]}
    payload = {
        "schema_version": "aptus.operator-mlx-split.v1",
        "seed": seed,
        "evaluation_fraction": evaluation_fraction,
        "corpus_rows": len(corpus_rows),
        "include_rows": len(include_keys),
        "output_rows": len(ordered),
        "compiled_train_prefix": len(ordered) - valid_count,
        "compiled_valid_tail": valid_count,
        "include_in_train_prefix": len(include_keys & prefix_keys),
        "include_in_valid_tail": len(include_keys - prefix_keys),
        "output_sha256": sha256_file(output),
    }
    if payload["include_in_valid_tail"]:
        raise ValueError("include rows landed in the MLX valid tail.")
    if manifest is not None:
        if manifest.exists():
            raise FileExistsError(f"Refusing to overwrite {manifest}")
        manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write a train JSONL whose Aptus MLX valid tail contains none of "
            "the --include recitation rows."
        )
    )
    parser.add_argument("--corpus", type=Path, required=True, help="All SFT rows.")
    parser.add_argument(
        "--include",
        type=Path,
        required=True,
        help="Rows that must land in compiled train (usually gold.jsonl).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    payload = prepare_train_file(
        corpus=args.corpus,
        include=args.include,
        output=args.output,
        evaluation_fraction=args.evaluation_fraction,
        seed=args.seed,
        manifest=args.manifest,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
