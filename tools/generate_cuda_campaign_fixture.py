from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / "examples/cuda-campaign-sft-v1.jsonl"
GENERATOR_VERSION = "aptus.cuda-campaign-fixture-generator.v1"
FIXTURE_ID = "aptus.cuda-campaign-sft.v1"
FIXTURE_SEED = 20260808
ROWS_PER_GROUP = 4
GROUP_COUNT = 128
LENGTH_BANDS = (
    (128, 256),
    (256, 128),
    (512, 64),
    (1024, 32),
    (2048, 32),
)
VOCABULARY = (
    "account",
    "action",
    "archive",
    "audit",
    "backup",
    "billing",
    "browser",
    "cache",
    "case",
    "checkpoint",
    "client",
    "configuration",
    "connection",
    "customer",
    "dataset",
    "device",
    "diagnostic",
    "document",
    "download",
    "evidence",
    "export",
    "failure",
    "file",
    "gateway",
    "identity",
    "invoice",
    "job",
    "lease",
    "manifest",
    "memory",
    "model",
    "network",
    "operator",
    "password",
    "policy",
    "process",
    "project",
    "queue",
    "receipt",
    "record",
    "recovery",
    "request",
    "resource",
    "response",
    "restart",
    "revision",
    "runtime",
    "safety",
    "service",
    "session",
    "setting",
    "source",
    "state",
    "storage",
    "support",
    "telemetry",
    "token",
    "training",
    "transfer",
    "validation",
    "vault",
    "verification",
    "version",
    "workflow",
)


def _word(*, row_index: int, position: int, channel: str) -> str:
    digest = hashlib.sha256(
        f"{GENERATOR_VERSION}:{FIXTURE_SEED}:{row_index}:{channel}:{position}".encode(
            "ascii"
        )
    ).digest()
    return VOCABULARY[int.from_bytes(digest[:2], "big") % len(VOCABULARY)]


def _bands() -> tuple[int, ...]:
    values = tuple(length for length, count in LENGTH_BANDS for _ in range(count))
    if len(values) != GROUP_COUNT * ROWS_PER_GROUP:
        raise RuntimeError("CUDA campaign fixture bands must define exactly 512 rows.")
    return values


def _row(row_index: int, intended_length: int) -> dict[str, int | str]:
    group_index = row_index // ROWS_PER_GROUP
    prompt_prefix = (
        "Synthetic",
        "support",
        "campaign",
        "case",
        f"row-{row_index:04d}",
        f"group-{group_index:03d}",
        f"band-{intended_length}",
        "Review",
        "the",
        "following",
        "operational",
        "context",
        "and",
        "prepare",
        "a",
        "bounded",
        "response",
        "with",
        "ordered",
        "verification",
        "steps",
    )
    completion_prefix = (
        "First",
        "preserve",
        "the",
        "record",
        "then",
        "verify",
        "the",
        "identity",
        "and",
        "state",
        "before",
        "applying",
        "the",
        "smallest",
        "safe",
        "action",
        "and",
        "recording",
        "the",
        "result",
    )
    generated_count = intended_length - len(prompt_prefix) - len(completion_prefix)
    if generated_count <= 0:
        raise RuntimeError("Fixture length band is too short for its fixed prefixes.")
    completion_generated = max(16, generated_count // 4)
    prompt_generated = generated_count - completion_generated
    prompt_words = prompt_prefix + tuple(
        _word(row_index=row_index, position=index, channel="prompt")
        for index in range(prompt_generated)
    )
    completion_words = completion_prefix + tuple(
        _word(row_index=row_index, position=index, channel="completion")
        for index in range(completion_generated)
    )
    if len(prompt_words) + len(completion_words) != intended_length:
        raise RuntimeError("Fixture row did not reach its intended lexical length.")
    return {
        "completion": " ".join(completion_words) + ".",
        "target_content_words": intended_length,
        "prompt": " ".join(prompt_words) + ".\nAgent:",
        "row_id": f"cuda-campaign-row-{row_index:04d}",
        "split_group": f"cuda-campaign-group-{group_index:03d}",
    }


def render_fixture() -> bytes:
    rows = [_row(index, length) for index, length in enumerate(_bands())]
    if len({row["prompt"] for row in rows}) != len(rows):
        raise RuntimeError("CUDA campaign fixture prompts must be unique.")
    group_counts = Counter(row["split_group"] for row in rows)
    if len(group_counts) != GROUP_COUNT or set(group_counts.values()) != {
        ROWS_PER_GROUP
    }:
        raise RuntimeError("CUDA campaign split groups must contain four rows each.")
    text = "\n".join(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for row in rows
    )
    return (text + "\n").encode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or verify the deterministic CUDA campaign fixture."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when the output differs from canonical bytes.",
    )
    return parser


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY).as_posix()
    except (OSError, ValueError):
        return path.name


def main() -> int:
    arguments = _parser().parse_args()
    expected = render_fixture()
    display_path = _display_path(arguments.output)
    if arguments.check:
        try:
            actual = arguments.output.read_bytes()
        except OSError as error:
            raise SystemExit(
                "CUDA campaign fixture is unreadable "
                f"({display_path}; {type(error).__name__}; errno={error.errno})."
            ) from error
        if actual != expected:
            raise SystemExit(
                "CUDA campaign fixture differs from deterministic generator output."
            )
    else:
        try:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_bytes(expected)
        except OSError as error:
            raise SystemExit(
                "CUDA campaign fixture is unwritable "
                f"({display_path}; {type(error).__name__}; errno={error.errno})."
            ) from error
    digest = hashlib.sha256(expected).hexdigest()
    print(
        f"{display_path}: rows={GROUP_COUNT * ROWS_PER_GROUP} "
        f"bytes={len(expected)} sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
