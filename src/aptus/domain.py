from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "aptus.training-plan.v6"
FACTS_SCHEMA_VERSION = "aptus.facts.v3"
RUNTIME_CONTRACT_VERSION = "aptus.runtime-contract.v1"
MODEL_COMPATIBILITY_SCHEMA_VERSION = "aptus.model-compatibility.v2"
MODEL_INSPECTION_RECEIPT_SCHEMA_VERSION = "aptus.model-inspection-receipt.v1"
MODEL_POLICY_BINDING_SCHEMA_VERSION = "aptus.model-policy-binding.v1"
PROVIDER_MODEL_ID = re.compile(
    r"^(?:[A-Za-z0-9][A-Za-z0-9._-]*/)?[A-Za-z0-9][A-Za-z0-9._-]*$"
)


def validate_provider_model_id(model_id: str) -> str:
    if model_id != model_id.strip():
        raise ValueError(
            "model_id must be a provider repository identifier, not a local path."
        )
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
    return model_id


def _sha256_is_valid(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _revision_is_valid(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and 40 <= len(value) <= 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _content_id_is_valid(value: Any, *, prefix: str) -> bool:
    return bool(
        isinstance(value, str)
        and value.startswith(prefix)
        and len(value) == len(prefix) + 20
        and all(character in "0123456789abcdef" for character in value[len(prefix) :])
    )


class UnsupportedPlanSchemaError(ValueError):
    """A persisted plan must be replanned instead of reinterpreted."""

    def __init__(self, found_schema: Any) -> None:
        self.found_schema = found_schema
        self.required_schema = SCHEMA_VERSION
        found = found_schema if isinstance(found_schema, str) else "missing"
        super().__init__(
            f"Replan required: persisted plan schema {found!r} cannot be "
            f"rehydrated as {SCHEMA_VERSION}. The source plan was not changed."
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


class TrainingRuntime(StrEnum):
    TRANSFORMERS_PEFT_CUDA = "transformers-peft-cuda"
    MLX_LM = "mlx-lm"
    PYTORCH_MPS = "pytorch-mps"


class EvidenceRequirement(StrEnum):
    PILOT_REQUIRED = "pilot-required"
    IMPLEMENTATION_REQUIRED = "implementation-required"


class Objective(StrEnum):
    QUALITY = "quality"
    MEMORY = "memory"
    SPEED = "speed"


class Method(StrEnum):
    FULL = "full"
    LORA = "lora"
    INT8_LORA = "int8-lora"
    QLORA = "qlora"


class AdapterProfile(StrEnum):
    ATTENTION_QKVO_V1 = "attention-qkvo.v1"
    DENSE_CAUSAL_LM_V1 = "dense-causal-lm.v1"


class ModelPolicyDecisionKind(StrEnum):
    PATH_MATCHED = "path-matched"
    FAMILY_RECOGNIZED = "family-recognized"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class ModelPolicyReasonCode(StrEnum):
    EXACT_REVIEWED_ARTIFACT = "exact-reviewed-artifact"
    REVIEWED_RUNTIME_PATH = "reviewed-runtime-path"
    PILOT_NOT_YET_PROVEN = "pilot-not-yet-proven"
    INVALID_FACTS = "invalid-compatibility-facts"
    IDENTITY_MISMATCH = "identity-mismatch"
    LAYER_COUNT_MISMATCH = "layer-count-mismatch"
    QUANTIZATION_LAYOUT_MISMATCH = "quantization-layout-mismatch"
    TOPOLOGY_INCOMPLETE = "topology-incomplete"
    DENSE_TOPOLOGY_REQUIRED = "dense-topology-required"
    SHARED_EXPERT_UNSUPPORTED = "shared-expert-unsupported"
    FOUR_BIT_REQUIRED = "four-bit-required"
    FAMILY_RECOGNIZED = "family-recognized"
    UNREVIEWED_SPARSE_MODEL = "unreviewed-sparse-model"
    NO_POLICY_MATCH = "no-policy-match"


class ModelPolicyBindingSource(StrEnum):
    PROVIDER_INSPECTION = "provider-inspection"
    USER_ATTESTED = "user-attested"


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
class RuntimeContract:
    """Explicit, versioned binding between a candidate and its runtime path."""

    compute_backend: Backend
    training_runtime: TrainingRuntime
    compiler_id: str | None
    estimator_id: str
    evidence_requirement: EvidenceRequirement
    export_kind: str | None
    schema_version: str = RUNTIME_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_CONTRACT_VERSION:
            raise ValueError(
                f"Runtime contract schema must be {RUNTIME_CONTRACT_VERSION}."
            )
        if not self.estimator_id.strip():
            raise ValueError("Runtime contract estimator_id is required.")
        expected_backend = {
            TrainingRuntime.TRANSFORMERS_PEFT_CUDA: Backend.CUDA,
            TrainingRuntime.MLX_LM: Backend.MPS,
            TrainingRuntime.PYTORCH_MPS: Backend.MPS,
        }[self.training_runtime]
        if self.compute_backend != expected_backend:
            raise ValueError(
                f"{self.training_runtime.value} requires {expected_backend.value} compute."
            )
        if self.evidence_requirement == EvidenceRequirement.PILOT_REQUIRED and (
            not self.compiler_id or not self.export_kind
        ):
            raise ValueError(
                "Pilot-gated runtime contracts require compiler and export identities."
            )


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
        if all(device.backend == Backend.MPS for device in self.devices):
            if self.host_ram_free_bytes is None:
                return 0
            return self.host_ram_free_bytes - self.reserve_per_device_bytes
        if any(device.free_vram_bytes is None for device in self.devices):
            return 0
        return min(
            int(device.free_vram_bytes) - self.reserve_per_device_bytes
            for device in self.devices
        )


@dataclass(frozen=True)
class MoETopology:
    """Exact provider configuration facts for one sparse expert architecture."""

    expert_count: int
    experts_per_token: int
    expert_intermediate_size: int
    decoder_sparse_step: int
    mlp_only_layers: tuple[int, ...] = ()
    shared_expert_intermediate_size: int | None = None

    def __post_init__(self) -> None:
        positive = {
            "expert_count": self.expert_count,
            "experts_per_token": self.experts_per_token,
            "expert_intermediate_size": self.expert_intermediate_size,
            "decoder_sparse_step": self.decoder_sparse_step,
        }
        invalid = [
            name
            for name, value in positive.items()
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ]
        if invalid:
            raise ValueError(
                "MoE topology facts must be positive: " + ", ".join(invalid) + "."
            )
        if self.experts_per_token > self.expert_count:
            raise ValueError("experts_per_token cannot exceed expert_count.")
        if self.shared_expert_intermediate_size is not None and (
            self.shared_expert_intermediate_size <= 0
        ):
            raise ValueError(
                "shared_expert_intermediate_size must be positive when supplied."
            )
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in self.mlp_only_layers
        ):
            raise ValueError("mlp_only_layers must contain non-negative integers.")
        if tuple(sorted(set(self.mlp_only_layers))) != self.mlp_only_layers:
            raise ValueError("mlp_only_layers must be sorted and unique.")


@dataclass(frozen=True)
class QuantizationOverride:
    """One provider-declared module exception to the default quantization."""

    module_path: str
    bits: int
    group_size: int

    def __post_init__(self) -> None:
        if (
            not self.module_path
            or len(self.module_path) > 256
            or not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*", self.module_path)
        ):
            raise ValueError(
                "quantization override module_path must be a dotted module identifier."
            )
        if (
            not isinstance(self.bits, int)
            or isinstance(self.bits, bool)
            or not 1 <= self.bits <= 16
        ):
            raise ValueError("quantization override bits must be between 1 and 16.")
        if (
            not isinstance(self.group_size, int)
            or isinstance(self.group_size, bool)
            or self.group_size <= 0
        ):
            raise ValueError("quantization override group_size must be positive.")


@dataclass(frozen=True)
class QuantizationLayout:
    """Canonical MLX groupwise quantization defaults and module overrides."""

    default_bits: int
    default_group_size: int
    module_overrides: tuple[QuantizationOverride, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.default_bits, int)
            or isinstance(self.default_bits, bool)
            or not 1 <= self.default_bits <= 16
        ):
            raise ValueError(
                "quantization layout default_bits must be between 1 and 16."
            )
        if (
            not isinstance(self.default_group_size, int)
            or isinstance(self.default_group_size, bool)
            or self.default_group_size <= 0
        ):
            raise ValueError("quantization layout default_group_size must be positive.")
        if any(
            not isinstance(item, QuantizationOverride) for item in self.module_overrides
        ):
            raise ValueError(
                "quantization layout module_overrides must contain override facts."
            )
        paths = tuple(item.module_path for item in self.module_overrides)
        if paths != tuple(sorted(set(paths))):
            raise ValueError(
                "quantization layout module_overrides must be sorted and unique."
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
    model_type: str | None = None
    quantization_bits: int | None = None
    quantization_layout: QuantizationLayout | None = None
    moe: MoETopology | None = None
    tokenizer_id: str | None = None
    provenance: Mapping[str, Provenance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_provider_model_id(self.model_id)
        if not (
            40 <= len(self.revision) <= 64
            and all(c in "0123456789abcdefABCDEF" for c in self.revision)
        ):
            raise ValueError(
                "revision must be an immutable 40-64 character hexadecimal commit identifier."
            )
        if not self.family.strip():
            raise ValueError("family is required.")
        if self.family != self.family.lower():
            raise ValueError("family must use its canonical lowercase identity.")
        if not self.architecture.strip():
            raise ValueError("architecture is required.")
        if self.model_type is not None and not self.model_type.strip():
            raise ValueError("model_type must be non-empty when supplied.")
        if self.quantization_bits is not None and (
            not isinstance(self.quantization_bits, int)
            or isinstance(self.quantization_bits, bool)
            or not 1 <= self.quantization_bits <= 16
        ):
            raise ValueError(
                "quantization_bits must be between 1 and 16 when supplied."
            )
        if self.quantization_layout is not None and (
            self.quantization_bits != self.quantization_layout.default_bits
        ):
            raise ValueError(
                "quantization_bits must equal quantization_layout default_bits."
            )
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
        if self.moe is not None and any(
            index >= self.layers for index in self.moe.mlp_only_layers
        ):
            raise ValueError("mlp_only_layers cannot reference a missing model layer.")
        if self.moe is not None and self.sparse_layer_count <= 0:
            raise ValueError("MoE topology must contain at least one sparse layer.")
        if self.active_parameters <= 0 or self.active_parameters > self.parameters:
            raise ValueError(
                "Derived active_parameters must be positive and no greater than total parameters."
            )

    @property
    def sparse_layer_count(self) -> int:
        if self.moe is None:
            return 0
        dense_layers = set(self.moe.mlp_only_layers)
        return sum(
            1
            for index in range(self.layers)
            if (index + 1) % self.moe.decoder_sparse_step == 0
            and index not in dense_layers
        )

    @property
    def active_parameters(self) -> int:
        """Logical parameters used per token; residency still uses ``parameters``."""

        if self.moe is None:
            return self.parameters
        inactive_expert_parameters = (
            self.sparse_layer_count
            * (self.moe.expert_count - self.moe.experts_per_token)
            * 3
            * self.hidden_size
            * self.moe.expert_intermediate_size
        )
        return self.parameters - inactive_expert_parameters


@dataclass(frozen=True)
class ModelCompatibilitySubject:
    """Normalized model facts evaluated against host compatibility policy."""

    family: str | None
    model_type: str | None
    architecture: str | None
    layers: int | None
    quantization_bits: int | None
    quantization_layout: QuantizationLayout | None
    moe: MoETopology | None
    fact_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.fact_errors, tuple):
            raise ValueError("Compatibility subject fact errors must be immutable.")
        text_values = (self.family, self.model_type, self.architecture)
        if any(
            value is not None and not isinstance(value, str) for value in text_values
        ):
            raise ValueError("Compatibility subject identities must be text.")
        if any(
            value is not None and (not value.strip() or value != value.strip())
            for value in text_values
        ):
            raise ValueError("Compatibility subject identities must be unpadded.")
        if self.family is not None and self.family != self.family.lower():
            raise ValueError(
                "Compatibility subject family must use its canonical lowercase identity."
            )
        if self.layers is not None and (
            not isinstance(self.layers, int)
            or isinstance(self.layers, bool)
            or self.layers <= 0
        ):
            raise ValueError("Compatibility subject layers must be positive.")
        if self.quantization_bits is not None and (
            not isinstance(self.quantization_bits, int)
            or isinstance(self.quantization_bits, bool)
            or not 1 <= self.quantization_bits <= 16
        ):
            raise ValueError(
                "Compatibility subject quantization_bits must be between 1 and 16."
            )
        if any(
            not isinstance(item, str) or not item.strip() or item != item.strip()
            for item in self.fact_errors
        ):
            raise ValueError("Compatibility subject fact errors must be unpadded.")
        if len(set(self.fact_errors)) != len(self.fact_errors):
            raise ValueError("Compatibility subject fact errors must be unique.")


@dataclass(frozen=True)
class ModelPolicyPath:
    """One fully bound execution path emitted by model policy evaluation."""

    path_id: str
    method: Method
    distribution: Distribution
    adapter_profile_id: AdapterProfile | None
    target_modules: tuple[str, ...]
    runtime_contract: RuntimeContract
    required_validation_levels: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path_id, str)
            or not self.path_id
            or self.path_id != self.path_id.strip()
        ):
            raise ValueError("Model policy path ID must be unpadded text.")
        if not isinstance(self.method, Method):
            raise ValueError("Model policy path method must be a known ID.")
        if not isinstance(self.distribution, Distribution):
            raise ValueError("Model policy path distribution must be a known ID.")
        if self.adapter_profile_id is not None and not isinstance(
            self.adapter_profile_id, AdapterProfile
        ):
            raise ValueError("Model policy path adapter profile must be a known ID.")
        if not isinstance(self.runtime_contract, RuntimeContract):
            raise ValueError("Model policy paths require a runtime contract.")
        if not isinstance(self.target_modules, tuple):
            raise ValueError("Model policy path target modules must be immutable.")
        if self.method == Method.FULL and self.adapter_profile_id is not None:
            raise ValueError("Full fine-tuning cannot carry an adapter profile.")
        if self.method != Method.FULL and self.adapter_profile_id is None:
            raise ValueError("Adapter model policy paths require an adapter profile.")
        if not self.target_modules:
            raise ValueError("Model policy paths require target modules.")
        if any(
            not module.strip() or module != module.strip()
            for module in self.target_modules
        ):
            raise ValueError("Model policy path target modules must be unpadded.")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("Model policy path target modules must be unique.")
        if not isinstance(self.required_validation_levels, tuple):
            raise ValueError("Model policy validation levels must be immutable.")
        if not self.required_validation_levels or any(
            item not in {"model-data", "measured-preflight", "pilot"}
            for item in self.required_validation_levels
        ):
            raise ValueError("Model policy validation levels must be known gates.")
        if len(set(self.required_validation_levels)) != len(
            self.required_validation_levels
        ):
            raise ValueError("Model policy validation levels must be unique.")
        if not isinstance(self.evidence_ids, tuple):
            raise ValueError("Model policy path evidence IDs must be immutable.")
        if not self.evidence_ids or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in self.evidence_ids
        ):
            raise ValueError("Model policy path evidence IDs must be unpadded text.")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("Model policy path evidence IDs must be unique.")


@dataclass(frozen=True)
class ModelPolicyDecision:
    """Compatibility-policy result kept separate from candidates and evidence."""

    schema_version: str
    decision_id: str
    subject_facts_sha256: str
    kind: ModelPolicyDecisionKind
    family: str | None
    policy_id: str | None
    policy_version: str | None
    paths: tuple[ModelPolicyPath, ...]
    reason_codes: tuple[ModelPolicyReasonCode, ...]
    evidence_ids: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_COMPATIBILITY_SCHEMA_VERSION:
            raise ValueError(
                "Model policy decision schema must be "
                f"{MODEL_COMPATIBILITY_SCHEMA_VERSION}."
            )
        if not _content_id_is_valid(self.decision_id, prefix="compat_"):
            raise ValueError("Model policy decision ID is invalid.")
        if not _sha256_is_valid(self.subject_facts_sha256):
            raise ValueError("Model policy subject digest must be SHA-256.")
        if not isinstance(self.kind, ModelPolicyDecisionKind):
            raise ValueError("Model policy decision kind must be a known ID.")
        if not isinstance(self.paths, tuple):
            raise ValueError("Model policy decision paths must be immutable.")
        if self.family is not None and not isinstance(self.family, str):
            raise ValueError("Model policy decision family must be text.")
        if not isinstance(self.reason, str):
            raise ValueError("Model policy decision reason must be text.")
        if not self.reason.strip() or self.reason != self.reason.strip():
            raise ValueError("Model policy decision reason must be unpadded.")
        if self.family is not None and (
            not self.family.strip() or self.family != self.family.strip()
        ):
            raise ValueError("Model policy decision family must be unpadded.")
        if (self.policy_id is None) != (self.policy_version is None):
            raise ValueError("Model policy ID and version must be supplied together.")
        if self.policy_id is not None and (
            not self.policy_id.strip() or self.policy_id != self.policy_id.strip()
        ):
            raise ValueError("Model policy ID must be unpadded.")
        if self.policy_version is not None and not re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+", self.policy_version
        ):
            raise ValueError("Model policy version must use semantic versioning.")
        if not isinstance(self.reason_codes, tuple) or not self.reason_codes:
            raise ValueError("Model policy decisions require immutable reason codes.")
        if any(
            not isinstance(item, ModelPolicyReasonCode) for item in self.reason_codes
        ):
            raise ValueError("Model policy decision reason codes must be known IDs.")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("Model policy decision reason codes must be unique.")
        if not isinstance(self.evidence_ids, tuple) or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in self.evidence_ids
        ):
            raise ValueError(
                "Model policy decision evidence IDs must be immutable text."
            )
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("Model policy decision evidence IDs must be unique.")
        if self.kind == ModelPolicyDecisionKind.PATH_MATCHED:
            if self.family is None:
                raise ValueError("A path-matched decision requires a family.")
            if self.policy_id is None:
                raise ValueError("A path-matched decision requires a policy identity.")
            if not self.paths:
                raise ValueError("A path-matched decision requires at least one path.")
            if len(set(self.paths)) != len(self.paths):
                raise ValueError("Model policy decision paths must be unique.")
        elif self.paths:
            raise ValueError("Only a path-matched decision may carry paths.")
        if (
            self.kind
            in {
                ModelPolicyDecisionKind.FAMILY_RECOGNIZED,
                ModelPolicyDecisionKind.UNKNOWN,
            }
            and self.policy_id is not None
        ):
            raise ValueError("Unregistered policy decisions cannot claim a policy ID.")
        if (
            self.kind == ModelPolicyDecisionKind.FAMILY_RECOGNIZED
            and self.family is None
        ):
            raise ValueError("A family-recognized decision requires a family.")


@dataclass(frozen=True)
class ModelInspectionProvenance:
    """One compatibility fact and the provenance class asserted by inspection."""

    field: str
    kind: ProvenanceKind
    source: str
    observed_at: str
    resolved_revision: str

    def __post_init__(self) -> None:
        if not self.field or self.field != self.field.strip():
            raise ValueError("Inspection provenance fields must be unpadded.")
        if not isinstance(self.kind, ProvenanceKind):
            raise ValueError("Inspection provenance kinds must be known IDs.")
        if not self.source or self.source != self.source.strip():
            raise ValueError("Inspection provenance sources must be unpadded.")
        try:
            observed = datetime.fromisoformat(self.observed_at)
        except ValueError as error:
            raise ValueError(
                "Inspection provenance observed_at must be ISO-8601."
            ) from error
        if observed.tzinfo is None or observed.utcoffset() is None:
            raise ValueError(
                "Inspection provenance observed_at must include a timezone."
            )
        if not _revision_is_valid(self.resolved_revision):
            raise ValueError("Inspection provenance revision must be immutable.")


@dataclass(frozen=True)
class ModelInspectionReceipt:
    """Tamper-evident provider observation bound to one policy decision."""

    schema_version: str
    receipt_id: str
    model_id: str
    resolved_revision: str
    observed_facts_sha256: str
    decision: ModelPolicyDecision
    provenance_summary: tuple[ModelInspectionProvenance, ...]
    provenance_requirement: ProvenanceKind | None
    provenance_requirement_met: bool
    evaluated_at: str

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_INSPECTION_RECEIPT_SCHEMA_VERSION:
            raise ValueError(
                "Model inspection receipt schema must be "
                f"{MODEL_INSPECTION_RECEIPT_SCHEMA_VERSION}."
            )
        if not _content_id_is_valid(self.receipt_id, prefix="receipt_"):
            raise ValueError("Model inspection receipt ID is invalid.")
        try:
            validate_provider_model_id(self.model_id)
        except ValueError as error:
            raise ValueError("Model inspection receipt model ID is invalid.") from error
        if not _revision_is_valid(self.resolved_revision):
            raise ValueError("Model inspection receipt revision must be immutable.")
        if not _sha256_is_valid(self.observed_facts_sha256):
            raise ValueError("Model inspection observed-facts digest must be SHA-256.")
        if not isinstance(self.decision, ModelPolicyDecision):
            raise ValueError("Model inspection receipts require a policy decision.")
        if not isinstance(self.provenance_summary, tuple):
            raise ValueError("Inspection receipt provenance must be immutable.")
        if not self.provenance_summary or any(
            not isinstance(item, ModelInspectionProvenance)
            for item in self.provenance_summary
        ):
            raise ValueError(
                "Inspection receipts require typed provenance for observed facts."
            )
        if any(
            item.kind not in {ProvenanceKind.PROVIDER_DECLARED, ProvenanceKind.INFERRED}
            for item in self.provenance_summary
        ):
            raise ValueError(
                "Inspection receipts may contain only provider-declared or "
                "provider-derived inferred facts."
            )
        fields_seen = tuple(item.field for item in self.provenance_summary)
        if fields_seen != tuple(sorted(set(fields_seen))):
            raise ValueError(
                "Inspection receipt provenance fields must be sorted and unique."
            )
        if self.provenance_requirement is not None and not isinstance(
            self.provenance_requirement, ProvenanceKind
        ):
            raise ValueError("Inspection receipt provenance requirement is invalid.")
        if self.provenance_requirement_met and self.provenance_requirement is None:
            raise ValueError("A met provenance requirement must identify its kind.")
        try:
            evaluated = datetime.fromisoformat(self.evaluated_at)
        except ValueError as error:
            raise ValueError(
                "Inspection receipt evaluated_at must be ISO-8601."
            ) from error
        if evaluated.tzinfo is None or evaluated.utcoffset() is None:
            raise ValueError("Inspection receipt evaluated_at must include a timezone.")


@dataclass(frozen=True)
class ModelPolicyBinding:
    """The exact registered path bound into one candidate identity."""

    schema_version: str
    decision_id: str
    subject_facts_sha256: str
    policy_id: str
    policy_version: str
    path_id: str
    source: ModelPolicyBindingSource
    inspection_receipt_id: str | None
    reason_codes: tuple[ModelPolicyReasonCode, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_POLICY_BINDING_SCHEMA_VERSION:
            raise ValueError(
                "Model policy binding schema must be "
                f"{MODEL_POLICY_BINDING_SCHEMA_VERSION}."
            )
        if not _content_id_is_valid(self.decision_id, prefix="compat_"):
            raise ValueError("Model policy binding decision ID is invalid.")
        if not _sha256_is_valid(self.subject_facts_sha256):
            raise ValueError("Model policy binding facts digest must be SHA-256.")
        if not self.policy_id or self.policy_id != self.policy_id.strip():
            raise ValueError("Model policy binding policy ID must be unpadded.")
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", self.policy_version):
            raise ValueError(
                "Model policy binding version must use semantic versioning."
            )
        if not self.path_id or self.path_id != self.path_id.strip():
            raise ValueError("Model policy binding path ID must be unpadded.")
        if not isinstance(self.source, ModelPolicyBindingSource):
            raise ValueError("Model policy binding source must be a known ID.")
        if self.source == ModelPolicyBindingSource.PROVIDER_INSPECTION:
            if not _content_id_is_valid(self.inspection_receipt_id, prefix="receipt_"):
                raise ValueError("Provider policy bindings require a receipt ID.")
        elif self.inspection_receipt_id is not None:
            raise ValueError("User-attested policy bindings cannot claim a receipt.")
        if not isinstance(self.reason_codes, tuple) or not self.reason_codes:
            raise ValueError("Model policy bindings require immutable reason codes.")
        if any(
            not isinstance(item, ModelPolicyReasonCode) for item in self.reason_codes
        ) or len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError(
                "Model policy binding reason codes must be known and unique."
            )
        if (
            not isinstance(self.evidence_ids, tuple)
            or not self.evidence_ids
            or any(
                not isinstance(item, str) or not item or item != item.strip()
                for item in self.evidence_ids
            )
            or len(set(self.evidence_ids)) != len(self.evidence_ids)
        ):
            raise ValueError("Model policy binding evidence IDs must be unique text.")


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
    training_runtime: TrainingRuntime | None = None
    optimizer_steps: int | None = None
    split_seed: int = 424242
    training_seed: int = 17
    data_order_seed: int = 1000017
    micro_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None

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
        if self.optimizer_steps is not None and self.optimizer_steps <= 0:
            raise ValueError("optimizer_steps must be positive when supplied.")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (self.split_seed, self.training_seed, self.data_order_seed)
        ):
            raise ValueError("Training seeds must be non-negative integers.")
        if self.data_order_seed != 1_000_000 + self.training_seed:
            raise ValueError("data_order_seed must equal 1000000 + training_seed.")
        if (self.micro_batch_size is None) != (
            self.gradient_accumulation_steps is None
        ):
            raise ValueError(
                "micro_batch_size and gradient_accumulation_steps must be supplied together."
            )
        if (
            self.micro_batch_size is not None
            and min(self.micro_batch_size, self.gradient_accumulation_steps) <= 0
        ):
            raise ValueError("Explicit batch controls must be positive.")
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
    model_policy_decision_id: str
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
    runtime_contract: RuntimeContract | None = None
    policy_binding: ModelPolicyBinding | None = None

    def __post_init__(self) -> None:
        if not _content_id_is_valid(self.model_policy_decision_id, prefix="compat_"):
            raise ValueError("Candidates require a valid model policy decision ID.")
        if (
            self.policy_binding is not None
            and self.policy_binding.decision_id != self.model_policy_decision_id
        ):
            raise ValueError(
                "Candidate policy binding must reference its policy decision."
            )


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
    model_policy_decision: ModelPolicyDecision
    model_policy_decision_source: ModelPolicyBindingSource
    inspection_receipt: ModelInspectionReceipt | None
    model_policy_snapshot_sha256: str
    evidence_records: tuple[EvidenceRecord, ...] = ()
    formula_version: str = "aptus-memory-v2"
    training_policy_version: str = "aptus-training-policy-v1"
    plan_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Training plans require schema {SCHEMA_VERSION}.")
        if not _sha256_is_valid(self.model_policy_snapshot_sha256):
            raise ValueError(
                "Training plans require a lowercase model policy snapshot SHA-256."
            )
        if not isinstance(self.model_policy_decision, ModelPolicyDecision):
            raise ValueError("Training plans require a model policy decision.")
        if not isinstance(self.model_policy_decision_source, ModelPolicyBindingSource):
            raise ValueError("Training plan policy source must be a known ID.")
        if (
            self.model_policy_decision_source
            == ModelPolicyBindingSource.PROVIDER_INSPECTION
            and self.inspection_receipt is None
        ):
            raise ValueError("Provider-inspection plans require an inspection receipt.")
        if (
            self.model_policy_decision_source == ModelPolicyBindingSource.USER_ATTESTED
            and self.inspection_receipt is not None
        ):
            raise ValueError("User-attested plans cannot carry an inspection receipt.")
        if self.inspection_receipt is not None and (
            self.inspection_receipt.decision.decision_id
            != self.model_policy_decision.decision_id
        ):
            raise ValueError("Inspection receipt must bind the plan policy decision.")
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ValueError("Training plans require immutable candidates.")
        if any(
            item.model_policy_decision_id != self.model_policy_decision.decision_id
            for item in self.candidates
        ):
            raise ValueError("Every candidate must bind the plan policy decision.")
        if self.recommended not in self.candidates:
            raise ValueError("The recommended candidate must belong to the plan.")
        for candidate in self.candidates:
            binding = candidate.policy_binding
            if binding is None:
                continue
            if binding.source != self.model_policy_decision_source:
                raise ValueError("Candidate policy binding source must match the plan.")
            expected_receipt_id = (
                self.inspection_receipt.receipt_id
                if self.inspection_receipt is not None
                else None
            )
            if binding.inspection_receipt_id != expected_receipt_id:
                raise ValueError(
                    "Candidate policy binding receipt must match the plan."
                )


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
    parent_promotion: Mapping[str, Any] | None = None


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
        if isinstance(value, ModelSpec):
            result["sparse_layer_count"] = value.sparse_layer_count
            result["active_parameters"] = value.active_parameters
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


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} must contain the exact versioned fields.")


def _mapping_sequence(value: Any, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a list of objects.")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{label} must contain only objects.")
    return tuple(value)


def _runtime_contract_from(value: Mapping[str, Any]) -> RuntimeContract:
    _require_exact_keys(
        value,
        {
            "compute_backend",
            "training_runtime",
            "compiler_id",
            "estimator_id",
            "evidence_requirement",
            "export_kind",
            "schema_version",
        },
        label="Runtime contract",
    )
    return RuntimeContract(
        compute_backend=Backend(value["compute_backend"]),
        training_runtime=TrainingRuntime(value["training_runtime"]),
        compiler_id=value.get("compiler_id"),
        estimator_id=str(value["estimator_id"]),
        evidence_requirement=EvidenceRequirement(value["evidence_requirement"]),
        export_kind=value.get("export_kind"),
        schema_version=value.get("schema_version", RUNTIME_CONTRACT_VERSION),
    )


def _model_policy_path_from(value: Mapping[str, Any]) -> ModelPolicyPath:
    _require_exact_keys(
        value,
        {
            "path_id",
            "method",
            "distribution",
            "adapter_profile_id",
            "target_modules",
            "runtime_contract",
            "required_validation_levels",
            "evidence_ids",
        },
        label="Model policy path",
    )
    runtime_value = value["runtime_contract"]
    if not isinstance(runtime_value, Mapping):
        raise ValueError("Model policy path runtime contract must be an object.")
    adapter_profile = value.get("adapter_profile_id")
    return ModelPolicyPath(
        path_id=str(value["path_id"]),
        method=Method(value["method"]),
        distribution=Distribution(value["distribution"]),
        adapter_profile_id=(
            AdapterProfile(adapter_profile) if adapter_profile is not None else None
        ),
        target_modules=tuple(value.get("target_modules", ())),
        runtime_contract=_runtime_contract_from(runtime_value),
        required_validation_levels=tuple(value.get("required_validation_levels", ())),
        evidence_ids=tuple(value.get("evidence_ids", ())),
    )


def _model_policy_decision_from(value: Mapping[str, Any]) -> ModelPolicyDecision:
    _require_exact_keys(
        value,
        {
            "schema_version",
            "decision_id",
            "subject_facts_sha256",
            "kind",
            "family",
            "policy_id",
            "policy_version",
            "paths",
            "reason_codes",
            "evidence_ids",
            "reason",
        },
        label="Model policy decision",
    )
    paths = _mapping_sequence(value.get("paths"), label="Model policy paths")
    return ModelPolicyDecision(
        schema_version=str(value["schema_version"]),
        decision_id=str(value["decision_id"]),
        subject_facts_sha256=str(value["subject_facts_sha256"]),
        kind=ModelPolicyDecisionKind(value["kind"]),
        family=value.get("family"),
        policy_id=value.get("policy_id"),
        policy_version=value.get("policy_version"),
        paths=tuple(_model_policy_path_from(item) for item in paths),
        reason_codes=tuple(
            ModelPolicyReasonCode(item) for item in value.get("reason_codes", ())
        ),
        evidence_ids=tuple(value.get("evidence_ids", ())),
        reason=str(value["reason"]),
    )


def model_policy_decision_from_primitive(
    value: Mapping[str, Any],
) -> ModelPolicyDecision:
    """Rehydrate one portable model-policy decision into the domain type."""

    if not isinstance(value, Mapping):
        raise ValueError("Model policy decision must be an object.")
    return _model_policy_decision_from(value)


def model_inspection_receipt_from_primitive(
    value: Mapping[str, Any],
) -> ModelInspectionReceipt:
    """Rehydrate a closed inspection receipt before planner verification."""

    if not isinstance(value, Mapping):
        raise ValueError("Model inspection receipt must be an object.")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "receipt_id",
            "model_id",
            "resolved_revision",
            "observed_facts_sha256",
            "decision",
            "provenance_summary",
            "provenance_requirement",
            "provenance_requirement_met",
            "evaluated_at",
        },
        label="Model inspection receipt",
    )
    decision_value = value.get("decision")
    if not isinstance(decision_value, Mapping):
        raise ValueError("Model inspection receipt decision must be an object.")
    requirement_met = value.get("provenance_requirement_met")
    if not isinstance(requirement_met, bool):
        raise ValueError("Receipt provenance_requirement_met must be boolean.")
    provenance_items = _mapping_sequence(
        value.get("provenance_summary"),
        label="Inspection receipt provenance",
    )
    for item in provenance_items:
        _require_exact_keys(
            item,
            {"field", "kind", "source", "observed_at", "resolved_revision"},
            label="Inspection provenance",
        )
    return ModelInspectionReceipt(
        schema_version=str(value["schema_version"]),
        receipt_id=str(value["receipt_id"]),
        model_id=str(value["model_id"]),
        resolved_revision=str(value["resolved_revision"]),
        observed_facts_sha256=str(value["observed_facts_sha256"]),
        decision=_model_policy_decision_from(decision_value),
        provenance_summary=tuple(
            ModelInspectionProvenance(
                field=str(item["field"]),
                kind=ProvenanceKind(item["kind"]),
                source=str(item["source"]),
                observed_at=str(item["observed_at"]),
                resolved_revision=str(item["resolved_revision"]),
            )
            for item in provenance_items
        ),
        provenance_requirement=(
            ProvenanceKind(value["provenance_requirement"])
            if value.get("provenance_requirement") is not None
            else None
        ),
        provenance_requirement_met=requirement_met,
        evaluated_at=str(value["evaluated_at"]),
    )


def _model_policy_binding_from(value: Mapping[str, Any]) -> ModelPolicyBinding:
    _require_exact_keys(
        value,
        {
            "schema_version",
            "decision_id",
            "subject_facts_sha256",
            "policy_id",
            "policy_version",
            "path_id",
            "source",
            "inspection_receipt_id",
            "reason_codes",
            "evidence_ids",
        },
        label="Model policy binding",
    )
    return ModelPolicyBinding(
        schema_version=str(value["schema_version"]),
        decision_id=str(value["decision_id"]),
        subject_facts_sha256=str(value["subject_facts_sha256"]),
        policy_id=str(value["policy_id"]),
        policy_version=str(value["policy_version"]),
        path_id=str(value["path_id"]),
        source=ModelPolicyBindingSource(value["source"]),
        inspection_receipt_id=value.get("inspection_receipt_id"),
        reason_codes=tuple(
            ModelPolicyReasonCode(item) for item in value.get("reason_codes", ())
        ),
        evidence_ids=tuple(value.get("evidence_ids", ())),
    )


def training_plan_from_primitive(value: Mapping[str, Any]) -> TrainingPlan:
    """Rehydrate the persisted v6 JSON contract without accepting older plans."""

    if not isinstance(value, Mapping):
        raise ValueError("Persisted plan must be an object.")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise UnsupportedPlanSchemaError(value.get("schema_version"))
    if "model_policy_snapshot_sha256" not in value:
        raise ValueError(
            "Persisted v6 plans require a model_policy_snapshot_sha256 field."
        )

    model_value = value["model"]
    hardware_value = value["hardware"]
    dataset_value = value["dataset"]
    target_value = value["target"]
    moe_value = model_value.get("moe")
    moe = None
    if isinstance(moe_value, Mapping):
        moe = MoETopology(
            expert_count=moe_value["expert_count"],
            experts_per_token=moe_value["experts_per_token"],
            expert_intermediate_size=moe_value["expert_intermediate_size"],
            decoder_sparse_step=moe_value["decoder_sparse_step"],
            mlp_only_layers=tuple(moe_value.get("mlp_only_layers", ())),
            shared_expert_intermediate_size=moe_value.get(
                "shared_expert_intermediate_size"
            ),
        )
    quantization_layout_value = model_value.get("quantization_layout")
    quantization_layout = None
    if isinstance(quantization_layout_value, Mapping):
        quantization_layout = QuantizationLayout(
            default_bits=quantization_layout_value["default_bits"],
            default_group_size=quantization_layout_value["default_group_size"],
            module_overrides=tuple(
                QuantizationOverride(
                    module_path=item["module_path"],
                    bits=item["bits"],
                    group_size=item["group_size"],
                )
                for item in quantization_layout_value.get("module_overrides", ())
            ),
        )
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
            "model_type": model_value.get("model_type"),
            "quantization_bits": model_value.get("quantization_bits"),
            "quantization_layout": quantization_layout,
            "moe": moe,
            "tokenizer_id": model_value.get("tokenizer_id"),
            "provenance": {
                str(key): provenance
                for key, item in model_value.get("provenance", {}).items()
                if (provenance := _provenance_from(item)) is not None
            },
        }
    )
    policy_decision_value = value.get("model_policy_decision")
    if not isinstance(policy_decision_value, Mapping):
        raise ValueError("Persisted v6 plans require a model policy decision.")
    model_policy_decision = _model_policy_decision_from(policy_decision_value)
    if "model_policy_decision_source" not in value:
        raise ValueError(
            "Persisted v6 plans require a model_policy_decision_source field."
        )
    model_policy_decision_source = ModelPolicyBindingSource(
        value["model_policy_decision_source"]
    )
    if "inspection_receipt" not in value:
        raise ValueError("Persisted v6 plans require an inspection_receipt field.")
    receipt_value = value.get("inspection_receipt")
    if receipt_value is not None and not isinstance(receipt_value, Mapping):
        raise ValueError("Persisted inspection_receipt must be an object or null.")
    inspection_receipt = (
        model_inspection_receipt_from_primitive(receipt_value)
        if isinstance(receipt_value, Mapping)
        else None
    )
    if (
        model_policy_decision_source == ModelPolicyBindingSource.PROVIDER_INSPECTION
        and inspection_receipt is None
    ):
        raise ValueError("Provider-inspection plans require an inspection receipt.")
    if (
        model_policy_decision_source == ModelPolicyBindingSource.USER_ATTESTED
        and inspection_receipt is not None
    ):
        raise ValueError("User-attested plans cannot carry an inspection receipt.")
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
        training_runtime=(
            TrainingRuntime(target_value["training_runtime"])
            if target_value.get("training_runtime")
            else None
        ),
        optimizer_steps=target_value.get("optimizer_steps"),
        split_seed=target_value["split_seed"],
        training_seed=target_value["training_seed"],
        data_order_seed=target_value["data_order_seed"],
        micro_batch_size=target_value.get("micro_batch_size"),
        gradient_accumulation_steps=target_value.get("gradient_accumulation_steps"),
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
        runtime_value = item.get("runtime_contract")
        if not isinstance(runtime_value, Mapping):
            raise ValueError("Persisted v5 candidates require a runtime contract.")
        runtime_contract = _runtime_contract_from(runtime_value)
        if "policy_binding" not in item:
            raise ValueError("Persisted v5 candidates require a policy_binding field.")
        policy_binding_value = item["policy_binding"]
        if policy_binding_value is not None and not isinstance(
            policy_binding_value, Mapping
        ):
            raise ValueError("Candidate policy_binding must be an object or null.")
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
            model_policy_decision_id=str(item["model_policy_decision_id"]),
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
            runtime_contract=runtime_contract,
            policy_binding=(
                _model_policy_binding_from(policy_binding_value)
                if isinstance(policy_binding_value, Mapping)
                else None
            ),
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
        model_policy_decision=model_policy_decision,
        model_policy_decision_source=model_policy_decision_source,
        inspection_receipt=inspection_receipt,
        model_policy_snapshot_sha256=str(value["model_policy_snapshot_sha256"]),
        evidence_records=evidence,
        formula_version=value.get("formula_version", "aptus-memory-v2"),
        training_policy_version=value.get(
            "training_policy_version", "aptus-training-policy-v1"
        ),
        plan_id=value["plan_id"],
    )
