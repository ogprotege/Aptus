from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .domain import (
    Backend,
    DatasetProfile,
    DeviceSpec,
    HardwareSpec,
    MeasurementKind,
    ModelSpec,
    gibibytes,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_examples(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        examples = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            examples = value
        elif isinstance(value, dict) and isinstance(value.get("train"), list):
            examples = value["train"]
        elif isinstance(value, dict):
            examples = [value]
        else:
            raise ValueError("JSON dataset must contain objects or a list of objects.")
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as source:
            examples = list(csv.DictReader(source))
    elif suffix == ".txt":
        examples = [
            {"text": line.strip()}
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raise ValueError(f"Unsupported dataset format: {suffix or '<none>'}")

    if not examples:
        raise ValueError("Dataset contains no examples.")
    if not all(isinstance(example, dict) for example in examples):
        raise ValueError("Every dataset example must be an object.")
    return examples


def _extract_text(example: dict[str, Any]) -> tuple[str, str]:
    text = example.get("text")
    if isinstance(text, str) and text.strip():
        return text, "text"
    content = example.get("content")
    if isinstance(content, str) and content.strip():
        return content, "content"

    messages = example.get("messages")
    if isinstance(messages, list):
        parts = [
            message.get("content", "")
            for message in messages
            if isinstance(message, dict)
            and isinstance(message.get("content"), str)
        ]
        text = "\n".join(part for part in parts if part.strip())
        if text:
            return text, "messages"

    if isinstance(example.get("prompt"), str) and isinstance(
        example.get("completion"), str
    ):
        return f"{example['prompt']}\n{example['completion']}", "prompt-completion"

    if isinstance(example.get("instruction"), str) and isinstance(
        example.get("output"), str
    ):
        parts = [example["instruction"]]
        if isinstance(example.get("input"), str) and example["input"].strip():
            parts.append(example["input"])
        parts.append(example["output"])
        return "\n".join(parts), "instruction"

    raise ValueError(
        "Dataset example has no supported text, messages, prompt/completion, "
        "or instruction/output schema."
    )


def _estimated_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def profile_dataset(
    source_path: Path,
    *,
    sample_limit: int | None = None,
) -> DatasetProfile:
    source_path = source_path.resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(f"Dataset source is not a file: {source_path}")
    if sample_limit is not None and sample_limit <= 0:
        raise ValueError("sample_limit must be positive.")

    all_examples = _load_examples(source_path)
    all_extracted = [_extract_text(example) for example in all_examples]
    extracted = (
        all_extracted[:sample_limit]
        if sample_limit is not None
        else all_extracted
    )
    schemas = {schema for _, schema in all_extracted}
    schema_name = schemas.pop() if len(schemas) == 1 else "mixed"
    sample_token_counts = [_estimated_tokens(text) for text, _ in extracted]

    scale = len(all_examples) / len(extracted)
    total_tokens = round(sum(sample_token_counts) * scale)
    warnings = [
        "Token counts use a deterministic four-characters-per-token estimate; "
        "supply a tokenizer in a later validation phase."
    ]
    if len(extracted) < len(all_examples):
        warnings.append(
            f"Profile sampled the first {len(extracted)} of "
            f"{len(all_examples)} examples."
        )
    if schema_name != "text":
        warnings.append(
            "The first generated Aptus bundle supports plain text training; "
            f"detected schema '{schema_name}' is analysis-only."
        )

    return DatasetProfile(
        source_path=source_path,
        source_sha256=_sha256(source_path),
        source_format=source_path.suffix.lower().lstrip("."),
        schema_name=schema_name,
        example_count=len(all_examples),
        total_estimated_tokens=total_tokens,
        sequence_p50=_nearest_rank(sample_token_counts, 0.50),
        sequence_p95=_nearest_rank(sample_token_counts, 0.95),
        sequence_max=max(sample_token_counts),
        measurement=MeasurementKind.ESTIMATED,
        warnings=tuple(warnings),
    )


def build_hardware_spec(
    *,
    backend: Backend,
    gpu_count: int,
    vram_gib: float,
    supports_bf16: bool,
    supports_4bit: bool,
    host_ram_gib: float,
    reserve_gib: float,
) -> HardwareSpec:
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive for this vertical slice.")
    devices = tuple(
        DeviceSpec(
            name=f"{backend.value.upper()} GPU {index}",
            backend=backend,
            total_vram_bytes=gibibytes(vram_gib),
            supports_bf16=supports_bf16,
            supports_4bit=supports_4bit,
        )
        for index in range(gpu_count)
    )
    return HardwareSpec(
        devices=devices,
        host_ram_bytes=gibibytes(host_ram_gib),
        reserve_per_device_bytes=gibibytes(reserve_gib),
    )


def build_model_spec(
    *,
    model_id: str,
    revision: str,
    family: str,
    parameters_b: float,
    hidden_size: int,
    layers: int,
    context_length: int,
    license_name: str,
    training_allowed: bool,
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        revision=revision,
        family=family.lower(),
        parameters=round(parameters_b * 1_000_000_000),
        hidden_size=hidden_size,
        layers=layers,
        context_length=context_length,
        license_name=license_name,
        training_allowed=training_allowed,
    )
