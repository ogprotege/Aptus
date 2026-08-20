from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aptus.cli import main
from aptus.prepare_train import (
    mlx_valid_count,
    order_rows_for_mlx_split,
    prepare_train_file,
)


def _row(i: int, group: str = "g") -> dict[str, str]:
    return {
        "prompt": f"q{i}\nAnswer:",
        "completion": f"a{i}",
        "split_group": group,
    }


class OperatorMlxPrepareTests(unittest.TestCase):
    def test_include_rows_stay_out_of_mlx_valid_tail(self) -> None:
        corpus = [_row(i) for i in range(395)]
        include = corpus[-62:]
        ordered = order_rows_for_mlx_split(
            corpus, include, evaluation_fraction=0.1, seed=20260820
        )
        self.assertEqual(len(ordered), 395)
        valid_count = mlx_valid_count(395, 0.1)
        self.assertEqual(valid_count, 40)
        prefix = {row["prompt"] for row in ordered[: 395 - valid_count]}
        tail = {row["prompt"] for row in ordered[395 - valid_count :]}
        include_prompts = {row["prompt"] for row in include}
        self.assertTrue(include_prompts <= prefix)
        self.assertFalse(include_prompts & tail)

    def test_too_many_include_rows_are_refused(self) -> None:
        corpus = [_row(i) for i in range(10)]
        with self.assertRaises(ValueError):
            order_rows_for_mlx_split(corpus, corpus, evaluation_fraction=0.1)

    def test_empty_completion_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            include = root / "gold.jsonl"
            corpus.write_text(
                json.dumps({"prompt": "q", "completion": "   "}) + "\n",
                encoding="utf-8",
            )
            include.write_text(
                json.dumps({"prompt": "q", "completion": "a"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-empty"):
                prepare_train_file(
                    corpus=corpus, include=include, output=root / "train.jsonl"
                )

    def test_duplicate_prompt_is_refused(self) -> None:
        corpus = [_row(1), _row(1)]
        with self.assertRaisesRegex(ValueError, "Duplicate prompt"):
            order_rows_for_mlx_split(corpus, [_row(1)], evaluation_fraction=0.1)

    def test_prepare_train_file_refuses_overwrite_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            include = root / "gold.jsonl"
            output = root / "train.jsonl"
            manifest = root / "split.json"
            corpus.write_text(
                "".join(
                    json.dumps(_row(i), ensure_ascii=False) + "\n" for i in range(20)
                ),
                encoding="utf-8",
            )
            include.write_text(
                "".join(
                    json.dumps(_row(i), ensure_ascii=False) + "\n"
                    for i in range(16, 20)
                ),
                encoding="utf-8",
            )
            payload = prepare_train_file(
                corpus=corpus,
                include=include,
                output=output,
                manifest=manifest,
            )
            self.assertEqual(payload["schema_version"], "aptus.operator-mlx-split.v1")
            self.assertEqual(payload["include_in_valid_tail"], 0)
            with self.assertRaises(FileExistsError):
                prepare_train_file(
                    corpus=corpus,
                    include=include,
                    output=output,
                    manifest=root / "other.json",
                )

    def test_prepare_train_cli_writes_ordered_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            include = root / "gold.jsonl"
            output = root / "train.jsonl"
            corpus.write_text(
                "".join(
                    json.dumps(_row(i), ensure_ascii=False) + "\n" for i in range(20)
                ),
                encoding="utf-8",
            )
            include.write_text(
                "".join(
                    json.dumps(_row(i), ensure_ascii=False) + "\n"
                    for i in range(16, 20)
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                main(
                    [
                        "prepare-train",
                        "--corpus",
                        str(corpus),
                        "--include",
                        str(include),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            valid_count = mlx_valid_count(20, 0.1)
            prefix = {row["prompt"] for row in rows[: 20 - valid_count]}
            self.assertTrue({_row(i)["prompt"] for i in range(16, 20)} <= prefix)
