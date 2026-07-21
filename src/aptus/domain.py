from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .plan_contract import IMMUTABLE_REVISION


def gibibytes(value: int | float) -> int:
    if value < 0:
        raise ValueError("GiB value must be non-negative.")
    return round(value * 1024**3)


class Backend(StrEnum):
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    CPU = "cpu"


class Objective(StrEnum):
    QUALITY = "quality"
    MEMORY = "memory"
    SPEED = "speed"


class Method(StrEnum):
    LORA = "lora"
    QLORA = "qlora"


class MeasurementKind(StrEnum):
    ESTIMATED = "estimated"
    TOKENIZER_MEASURED = "tokenizer-measured"


class ValidationState(StrEnum):
    INVALID = "invalid"
    STATIC_PASS = "static-pass"
    ENVIRONMENT_PASS = "environment-pass"
    SMOKE_PASS = "smoke-pass"


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    backend: Backend
    total_vram_bytes: int
    supports_bf16: bool
    supports_4bit: bool

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Device name is required.")
        if self.total_vram_bytes <= 0:
            raise ValueError("Device total_vram_bytes must be positive.")


@dataclass(frozen=True)
class HardwareSpec:
    devices: tuple[DeviceSpec, ...]
    host_ram_bytes: int
    reserve_per_device_bytes: int

    def __post_init__(self) -> None:
        if self.host_ram_bytes <= 0:
            raise ValueError("host_ram_bytes must be positive.")
        if self.reserve_per_device_bytes < 0:
            raise ValueError("reserve_per_device_bytes must be non-negative.")
        if any(
            self.reserve_per_device_bytes >= device.total_vram_bytes
            for device in self.devices
        ):
            raise ValueError("Device reserve must be smaller than total VRAM.")

    @property
    def gpu_count(self) -> int:
        return len(self.devices)

    @property
    def limiting_vram_bytes(self) -> int:
        if not self.devices:
            return 0
        return min(
            device.total_vram_bytes - self.reserve_per_device_bytes
            for device in self.devices
        )


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    family: str
    parameters: int
    hidden_size: int
    layers: int
    context_length: int
    license_name: str
    training_allowed: bool

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id is required.")
        if not IMMUTABLE_REVISION.fullmatch(self.revision):
            raise ValueError(
                "revision must be an immutable 40-64 character hexadecimal "
                "commit identifier."
            )
        if not self.family.strip():
            raise ValueError("family is required.")
        if self.parameters <= 0:
            raise ValueError("parameters must be positive.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.layers <= 0:
            raise ValueError("layers must be positive.")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive.")
        if not self.license_name.strip():
            raise ValueError("license_name is required.")
        if not self.training_allowed:
            raise ValueError("Model training permission must be explicitly allowed.")


@dataclass(frozen=True)
class DatasetProfile:
    source_path: Path
    source_sha256: str
    source_format: str
    schema_name: str
    example_count: int
    total_estimated_tokens: int
    sequence_p50: int
    sequence_p95: int
    sequence_max: int
    measurement: MeasurementKind
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a SHA-256 hex digest.")
        if self.example_count <= 0:
            raise ValueError("Dataset must contain at least one example.")
        if self.total_estimated_tokens <= 0:
            raise ValueError("Dataset token estimate must be positive.")
        if not (
            0 < self.sequence_p50 <= self.sequence_p95 <= self.sequence_max
        ):
            raise ValueError("Sequence percentiles must be positive and ordered.")


@dataclass(frozen=True)
class TrainingTarget:
    objective: Objective
    sequence_length: int
    effective_batch_size: int
    max_epochs: int
    method_preference: Method | None = None

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if self.effective_batch_size <= 0:
            raise ValueError("effective_batch_size must be positive.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")


@dataclass(frozen=True)
class MemoryBreakdown:
    base_weights_bytes: int
    quantization_metadata_bytes: int
    adapter_weights_bytes: int
    adapter_gradients_bytes: int
    optimizer_states_bytes: int
    activations_bytes: int
    temporary_overhead_bytes: int
    safety_margin_bytes: int

    @property
    def estimated_peak_bytes(self) -> int:
        return sum(
            getattr(self, field.name)
            for field in fields(self)
            if field.name.endswith("_bytes")
        )


@dataclass(frozen=True)
class CandidatePlan:
    method: Method
    feasible: bool
    rejection_reasons: tuple[str, ...]
    precision: str
    quantization: str | None
    micro_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    rank: int
    alpha: int
    learning_rate: float
    target_modules: tuple[str, ...]
    memory: MemoryBreakdown
    preference_score: float
    confidence: str
    assumptions: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class TrainingPlan:
    schema_version: str
    model: ModelSpec
    dataset: DatasetProfile
    hardware: HardwareSpec
    target: TrainingTarget
    recommended: CandidatePlan
    candidates: tuple[CandidatePlan, ...]
    warnings: tuple[str, ...]
    recommendation_rationale: tuple[str, ...]


@dataclass(frozen=True)
class ValidationFinding:
    code: str
    message: str
    severity: str
    path: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    state: ValidationState
    findings: tuple[ValidationFinding, ...]
    checked_files: tuple[str, ...]
    artifact_fingerprint: str
    smoke_command: tuple[str, ...] | None = None
    runtime_evidence: tuple[str, ...] = ()


def to_primitive(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_primitive(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [to_primitive(item) for item in value]
    return value
