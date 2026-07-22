from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "aptus.training-plan.v2"
FACTS_SCHEMA_VERSION = "aptus.facts.v2"
PROVIDER_MODEL_ID = re.compile(
    r"^(?:[A-Za-z0-9][A-Za-z0-9._-]*/)?[A-Za-z0-9][A-Za-z0-9._-]*$"
)


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
    FULL = "full"
    LORA = "lora"
    INT8_LORA = "int8-lora"
    QLORA = "qlora"


class Distribution(StrEnum):
    SINGLE = "single"
    DDP = "ddp"
    FSDP = "fsdp"


class CandidateStatus(StrEnum):
    FEASIBLE = "feasible"
    CONDITIONAL = "conditional"
    INFEASIBLE = "infeasible"
    UNSUPPORTED = "unsupported"


class MeasurementKind(StrEnum):
    ESTIMATED = "estimated"
    TOKENIZER_MEASURED = "tokenizer-measured"


class ProvenanceKind(StrEnum):
    MEASURED = "measured"
    PROVIDER_DECLARED = "provider-declared"
    USER_ATTESTED = "user-attested"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class ValidationState(StrEnum):
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"
    CONTRACT_PASS = "contract-pass"
    STATIC_PASS = "static-pass"
    DEPENDENCY_PASS = "dependency-pass"
    ENVIRONMENT_PASS = "dependency-pass"  # compatibility alias; never structural-only
    MODEL_DATA_PASS = "model-data-pass"
    MEASURED_PREFLIGHT_PASS = "measured-preflight-pass"
    PILOT_PASS = "pilot-pass"
    EXECUTION_APPROVED = "execution-approved"
    MEASURED_RUN_PASS = "measured-run-pass"
    SMOKE_PASS = "measured-preflight-pass"  # compatibility alias


class RunState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Provenance:
    kind: ProvenanceKind
    source: str
    observed_at: str | None = None
    digest: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    claim: str
    source: str
    source_kind: str
    scope: str
    confidence: str
    revision: str | None = None


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    backend: Backend
    total_vram_bytes: int
    supports_bf16: bool
    supports_4bit: bool
    supports_8bit: bool = False
    free_vram_bytes: int | None = None
    compute_capability: str | None = None
    driver_version: str | None = None
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Device name is required.")
        if self.total_vram_bytes <= 0:
            raise ValueError("Device total_vram_bytes must be positive.")
        if (
            self.free_vram_bytes is not None
            and not 0 < self.free_vram_bytes <= self.total_vram_bytes
        ):
            raise ValueError(
                "free_vram_bytes must be positive and no greater than total VRAM."
            )


@dataclass(frozen=True)
class HardwareSpec:
    devices: tuple[DeviceSpec, ...]
    host_ram_bytes: int
    reserve_per_device_bytes: int
    disk_free_bytes: int | None = None
    host_ram_free_bytes: int | None = None
    cuda_version: str | None = None
    interconnect: str | None = None
    provenance: Provenance | None = None

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
        if self.disk_free_bytes is not None and self.disk_free_bytes <= 0:
            raise ValueError("disk_free_bytes must be positive when supplied.")
        if (
            self.host_ram_free_bytes is not None
            and not 0 < self.host_ram_free_bytes <= self.host_ram_bytes
        ):
            raise ValueError(
                "host_ram_free_bytes must be positive and no greater than total host RAM."
            )

    @property
    def gpu_count(self) -> int:
        return len(self.devices)

    @property
    def limiting_vram_bytes(self) -> int:
        if not self.devices:
            return 0
        return min(
            (device.free_vram_bytes or device.total_vram_bytes)
            - self.reserve_per_device_bytes
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
    intermediate_size: int | None = None
    architecture: str = "causal-lm"
    tokenizer_id: str | None = None
    provenance: Mapping[str, Provenance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model_id = self.model_id.strip()
        if (
            not model_id
            or len(model_id) > 96
            or not PROVIDER_MODEL_ID.fullmatch(model_id)
            or ".." in model_id
            or "--" in model_id
            or model_id.endswith(".git")
        ):
            raise ValueError(
                "model_id must be a provider repository identifier, not a local path."
            )
        if not (
            40 <= len(self.revision) <= 64
            and all(c in "0123456789abcdefABCDEF" for c in self.revision)
        ):
            raise ValueError(
                "revision must be an immutable 40-64 character hexadecimal commit identifier."
            )
        if not self.family.strip():
            raise ValueError("family is required.")
        structural = {
            "parameters": self.parameters,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "context_length": self.context_length,
        }
        if self.intermediate_size is not None:
            structural["intermediate_size"] = self.intermediate_size
        invalid = [name for name, value in structural.items() if value <= 0]
        if invalid:
            raise ValueError(
                "Model structural facts must be positive: " + ", ".join(invalid) + "."
            )
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
    schema_counts: Mapping[str, int] = field(default_factory=dict)
    sampled_examples: int = 0
    sample_indices: tuple[int, ...] = ()
    duplicate_count: int = 0
    empty_count: int = 0
    truncation_count: int = 0
    truncation_rate: float = 0.0
    source_size_bytes: int = 0
    canonical_size_bytes: int = 0
    max_canonical_row_bytes: int = 0
    bundle_path: str | None = None
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if len(self.source_sha256) != 64 or any(
            c not in "0123456789abcdefABCDEF" for c in self.source_sha256
        ):
            raise ValueError("source_sha256 must be a SHA-256 hex digest.")
        if self.example_count <= 0 or self.total_estimated_tokens <= 0:
            raise ValueError("Dataset must contain at least one non-empty example.")
        if not (0 < self.sequence_p50 <= self.sequence_p95 <= self.sequence_max):
            raise ValueError("Sequence percentiles must be positive and ordered.")
        if (
            self.source_size_bytes <= 0
            or self.canonical_size_bytes <= 0
            or self.max_canonical_row_bytes <= 0
        ):
            raise ValueError("Dataset source and canonical sizes must be positive.")


@dataclass(frozen=True)
class TrainingTarget:
    objective: Objective
    sequence_length: int
    effective_batch_size: int
    max_epochs: int
    method_preference: Method | None = None
    task: str = "sft"
    evaluation_fraction: float = 0.1
    packing: bool = False
    checkpoint_steps: int = 100
    max_wall_time_minutes: int | None = None

    def __post_init__(self) -> None:
        if (
            min(
                self.sequence_length,
                self.effective_batch_size,
                self.max_epochs,
                self.checkpoint_steps,
            )
            <= 0
        ):
            raise ValueError("Training target numeric values must be positive.")
        if self.max_wall_time_minutes is not None and self.max_wall_time_minutes <= 0:
            raise ValueError("max_wall_time_minutes must be positive when supplied.")
        if not 0 <= self.evaluation_fraction < 1:
            raise ValueError("evaluation_fraction must be in [0, 1).")


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
    communication_bytes: int = 0
    workspace_bytes: int = 0
    allocator_bytes: int = 0
    load_transient_bytes: int = 0
    component_upper_bounds: Mapping[str, int] = field(default_factory=dict)
    upper_estimate_bytes: int = 0
    formula_version: str = "aptus-memory-v2"
    assumptions: tuple[str, ...] = ()

    @property
    def point_estimate_bytes(self) -> int:
        return sum(
            (
                self.base_weights_bytes,
                self.quantization_metadata_bytes,
                self.adapter_weights_bytes,
                self.adapter_gradients_bytes,
                self.optimizer_states_bytes,
                self.activations_bytes,
                self.temporary_overhead_bytes,
                self.communication_bytes,
                self.workspace_bytes,
                self.allocator_bytes,
                self.load_transient_bytes,
            )
        )

    @property
    def estimated_peak_bytes(self) -> int:
        return self.point_estimate_bytes

    @property
    def upper_bytes(self) -> int:
        if self.component_upper_bounds:
            return sum(self.component_upper_bounds.values())
        return max(
            self.point_estimate_bytes + self.safety_margin_bytes,
            self.upper_estimate_bytes,
        )

    @property
    def uncertainty_bytes(self) -> int:
        """Named uncertainty term retained separately from the point estimate."""

        return self.safety_margin_bytes


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
    candidate_id: str = ""
    status: CandidateStatus = CandidateStatus.INFEASIBLE
    distribution: Distribution = Distribution.SINGLE
    world_size: int = 1
    device_indices: tuple[int, ...] = (0,)
    user_reserve_bytes: int = 0
    pareto_frontier: bool = False
    ranking_basis: tuple[str, ...] = ()
    required_host_ram_bytes: int = 0
    required_disk_bytes: int = 0
    checkpoint_retention_bytes: int = 0
    final_export_bytes: int = 0


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
    evidence_records: tuple[EvidenceRecord, ...] = ()
    formula_version: str = "aptus-memory-v2"
    plan_id: str = ""


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
    validation_level: str = "contract"
    bindings: Mapping[str, str] = field(default_factory=dict)
    validator_version: str = "aptus-validator-v2"
    validated_at: str | None = None
    preflight_metrics: Mapping[str, Any] | None = None
    pilot_metrics: Mapping[str, Any] | None = None
    final_export: Mapping[str, Any] | None = None
    measured_run: Mapping[str, Any] | None = None
    measured_run_completed_at: str | None = None
    latest_recheck: Mapping[str, Any] | None = None


def to_primitive(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        result = {
            item.name: to_primitive(getattr(value, item.name)) for item in fields(value)
        }
        if isinstance(value, MemoryBreakdown):
            result["point_estimate_bytes"] = value.point_estimate_bytes
            result["estimated_peak_bytes"] = value.estimated_peak_bytes
            result["upper_estimate_bytes"] = value.upper_bytes
            result["uncertainty_bytes"] = value.uncertainty_bytes
        return result
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_primitive(item) for item in value]
    return value


def _provenance_from(value: Any) -> Provenance | None:
    if not isinstance(value, Mapping):
        return None
    return Provenance(
        kind=ProvenanceKind(value["kind"]),
        source=str(value["source"]),
        observed_at=value.get("observed_at"),
        digest=value.get("digest"),
        detail=value.get("detail"),
    )


def training_plan_from_primitive(value: Mapping[str, Any]) -> TrainingPlan:
    """Rehydrate the persisted v2 JSON contract without accepting executable values."""

    model_value = value["model"]
    hardware_value = value["hardware"]
    dataset_value = value["dataset"]
    target_value = value["target"]
    model = ModelSpec(
        **{
            **{
                key: model_value[key]
                for key in (
                    "model_id",
                    "revision",
                    "family",
                    "parameters",
                    "hidden_size",
                    "layers",
                    "context_length",
                    "license_name",
                    "training_allowed",
                )
            },
            "intermediate_size": model_value.get("intermediate_size"),
            "architecture": model_value.get("architecture", "causal-lm"),
            "tokenizer_id": model_value.get("tokenizer_id"),
            "provenance": {
                str(key): provenance
                for key, item in model_value.get("provenance", {}).items()
                if (provenance := _provenance_from(item)) is not None
            },
        }
    )
    devices = tuple(
        DeviceSpec(
            name=item["name"],
            backend=Backend(item["backend"]),
            total_vram_bytes=item["total_vram_bytes"],
            supports_bf16=item["supports_bf16"],
            supports_4bit=item["supports_4bit"],
            supports_8bit=item.get("supports_8bit", False),
            free_vram_bytes=item.get("free_vram_bytes"),
            compute_capability=item.get("compute_capability"),
            driver_version=item.get("driver_version"),
            provenance=_provenance_from(item.get("provenance")),
        )
        for item in hardware_value["devices"]
    )
    hardware = HardwareSpec(
        devices=devices,
        host_ram_bytes=hardware_value["host_ram_bytes"],
        reserve_per_device_bytes=hardware_value["reserve_per_device_bytes"],
        disk_free_bytes=hardware_value.get("disk_free_bytes"),
        host_ram_free_bytes=hardware_value.get("host_ram_free_bytes"),
        cuda_version=hardware_value.get("cuda_version"),
        interconnect=hardware_value.get("interconnect"),
        provenance=_provenance_from(hardware_value.get("provenance")),
    )
    dataset = DatasetProfile(
        source_path=Path(dataset_value["source_path"]),
        source_sha256=dataset_value["source_sha256"],
        source_format=dataset_value["source_format"],
        schema_name=dataset_value["schema_name"],
        example_count=dataset_value["example_count"],
        total_estimated_tokens=dataset_value["total_estimated_tokens"],
        sequence_p50=dataset_value["sequence_p50"],
        sequence_p95=dataset_value["sequence_p95"],
        sequence_max=dataset_value["sequence_max"],
        measurement=MeasurementKind(dataset_value["measurement"]),
        warnings=tuple(dataset_value.get("warnings", ())),
        schema_counts=dict(dataset_value.get("schema_counts", {})),
        sampled_examples=dataset_value.get("sampled_examples", 0),
        sample_indices=tuple(dataset_value.get("sample_indices", ())),
        duplicate_count=dataset_value.get("duplicate_count", 0),
        empty_count=dataset_value.get("empty_count", 0),
        truncation_count=dataset_value.get("truncation_count", 0),
        truncation_rate=dataset_value.get("truncation_rate", 0.0),
        source_size_bytes=dataset_value.get("source_size_bytes", 0),
        canonical_size_bytes=dataset_value.get("canonical_size_bytes", 0),
        max_canonical_row_bytes=dataset_value.get("max_canonical_row_bytes", 0),
        bundle_path=dataset_value.get("bundle_path"),
        provenance=_provenance_from(dataset_value.get("provenance")),
    )
    target = TrainingTarget(
        objective=Objective(target_value["objective"]),
        sequence_length=target_value["sequence_length"],
        effective_batch_size=target_value["effective_batch_size"],
        max_epochs=target_value["max_epochs"],
        method_preference=Method(target_value["method_preference"])
        if target_value.get("method_preference")
        else None,
        task=target_value.get("task", "sft"),
        evaluation_fraction=target_value.get("evaluation_fraction", 0.1),
        packing=target_value.get("packing", False),
        checkpoint_steps=target_value.get("checkpoint_steps", 100),
        max_wall_time_minutes=target_value.get("max_wall_time_minutes"),
    )

    memory_fields = {item.name for item in fields(MemoryBreakdown)}

    def candidate_from(item: Mapping[str, Any]) -> CandidatePlan:
        memory_value = item["memory"]
        memory_arguments = {
            key: memory_value[key] for key in memory_fields if key in memory_value
        }
        memory_arguments["assumptions"] = tuple(memory_arguments.get("assumptions", ()))
        memory_arguments["component_upper_bounds"] = dict(
            memory_arguments.get("component_upper_bounds", {})
        )
        memory = MemoryBreakdown(**memory_arguments)
        return CandidatePlan(
            method=Method(item["method"]),
            feasible=item["feasible"],
            rejection_reasons=tuple(item.get("rejection_reasons", ())),
            precision=item["precision"],
            quantization=item.get("quantization"),
            micro_batch_size=item["micro_batch_size"],
            gradient_accumulation_steps=item["gradient_accumulation_steps"],
            effective_batch_size=item["effective_batch_size"],
            rank=item["rank"],
            alpha=item["alpha"],
            learning_rate=item["learning_rate"],
            target_modules=tuple(item.get("target_modules", ())),
            memory=memory,
            preference_score=item.get("preference_score", 0.0),
            confidence=item.get("confidence", "unknown"),
            assumptions=tuple(item.get("assumptions", ())),
            evidence=tuple(item.get("evidence", ())),
            candidate_id=item["candidate_id"],
            status=CandidateStatus(item["status"]),
            distribution=Distribution(item["distribution"]),
            world_size=item["world_size"],
            device_indices=tuple(item.get("device_indices", range(item["world_size"]))),
            user_reserve_bytes=item.get("user_reserve_bytes", 0),
            pareto_frontier=item.get("pareto_frontier", False),
            ranking_basis=tuple(item.get("ranking_basis", ())),
            required_host_ram_bytes=item.get("required_host_ram_bytes", 0),
            required_disk_bytes=item.get("required_disk_bytes", 0),
            checkpoint_retention_bytes=item.get("checkpoint_retention_bytes", 0),
            final_export_bytes=item.get("final_export_bytes", 0),
        )

    candidates = tuple(candidate_from(item) for item in value["candidates"])
    recommended_id = value["recommended"]["candidate_id"]
    recommended = next(
        item for item in candidates if item.candidate_id == recommended_id
    )
    evidence = tuple(
        EvidenceRecord(**item) for item in value.get("evidence_records", ())
    )
    return TrainingPlan(
        schema_version=value["schema_version"],
        model=model,
        dataset=dataset,
        hardware=hardware,
        target=target,
        recommended=recommended,
        candidates=candidates,
        warnings=tuple(value.get("warnings", ())),
        recommendation_rationale=tuple(value.get("recommendation_rationale", ())),
        evidence_records=evidence,
        formula_version=value.get("formula_version", "aptus-memory-v2"),
        plan_id=value["plan_id"],
    )
