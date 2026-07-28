from __future__ import annotations

from collections.abc import Iterable

from ..domain import Backend, Distribution, Method, TrainingRuntime
from .contracts import MethodDescriptor, MethodLifecycle, RuntimeBinding


def _descriptor(**values: object) -> MethodDescriptor:
    return MethodDescriptor(**values)  # type: ignore[arg-type]


def _runtime(**values: object) -> RuntimeBinding:
    return RuntimeBinding(**values)  # type: ignore[arg-type]


METHOD_REGISTRY: dict[str, MethodDescriptor] = {
    item.method_id: item
    for item in (
        _descriptor(
            method_id="full",
            display_name="Full fine-tuning",
            summary="Updates every model parameter and emits a complete model artifact.",
            lifecycle=MethodLifecycle.GATED_EXECUTABLE,
            selectable=True,
            parameter_scope="all-parameters",
            parameterization="dense-full",
            base_storage="unquantized",
            compiler_id="transformers.full.v2",
            export_kind="full-model-safetensors",
            supported_backends=("cuda",),
            supported_distributions=("single", "ddp"),
            evidence_ids=("method.full.transformers", "estimate.memory.v2"),
            pilot_requirement="Exact model-data validation, measured preflight, and a bounded real-model pilot are mandatory.",
            aliases=("full-parameter",),
            runtime_bindings=(
                _runtime(
                    training_runtime="transformers-peft-cuda",
                    compute_backend="cuda",
                    compiler_id="transformers.full.v2",
                    estimator_id="aptus-memory-v2",
                    export_kind="full-model-safetensors",
                    supported_distributions=("single", "ddp"),
                ),
            ),
        ),
        _descriptor(
            method_id="lora",
            display_name="LoRA",
            summary="Freezes the base model and trains low-rank updates on inspected target modules.",
            lifecycle=MethodLifecycle.GATED_EXECUTABLE,
            selectable=True,
            parameter_scope="frozen-base-plus-adapter",
            parameterization="lora",
            base_storage="unquantized",
            compiler_id="transformers.peft-lora.v2",
            export_kind="peft-adapter-safetensors",
            supported_backends=("cuda", "mps"),
            supported_distributions=("single", "ddp", "fsdp"),
            evidence_ids=("method.lora.paper", "estimate.memory.v2"),
            pilot_requirement="Target-module inspection, measured preflight, and a bounded real-model pilot are mandatory.",
            runtime_bindings=(
                _runtime(
                    training_runtime="transformers-peft-cuda",
                    compute_backend="cuda",
                    compiler_id="transformers.peft-lora.v2",
                    estimator_id="aptus-memory-v2",
                    export_kind="peft-adapter-safetensors",
                    supported_distributions=("single", "ddp", "fsdp"),
                ),
                _runtime(
                    training_runtime="mlx-lm",
                    compute_backend="mps",
                    compiler_id="mlx-lm.lora.v1",
                    estimator_id="aptus-memory-mlx-v2",
                    export_kind="mlx-lm-adapter",
                    supported_distributions=("single",),
                ),
            ),
        ),
        _descriptor(
            method_id="int8-lora",
            display_name="8-bit LoRA",
            summary="Trains LoRA adapters over a frozen bitsandbytes INT8 base.",
            lifecycle=MethodLifecycle.GATED_EXECUTABLE,
            selectable=True,
            parameter_scope="frozen-base-plus-adapter",
            parameterization="lora",
            base_storage="bitsandbytes-int8",
            compiler_id="transformers.peft-int8-lora.v2",
            export_kind="peft-adapter-safetensors",
            supported_backends=("cuda",),
            supported_distributions=("single", "ddp"),
            evidence_ids=(
                "method.lora.paper",
                "method.bitsandbytes.int8",
                "estimate.memory.v2",
            ),
            pilot_requirement="Exact INT8 kernel capability, target inspection, measured preflight, and a bounded pilot are mandatory.",
            aliases=("8bit-lora",),
            runtime_bindings=(
                _runtime(
                    training_runtime="transformers-peft-cuda",
                    compute_backend="cuda",
                    compiler_id="transformers.peft-int8-lora.v2",
                    estimator_id="aptus-memory-v2",
                    export_kind="peft-adapter-safetensors",
                    supported_distributions=("single", "ddp"),
                ),
            ),
        ),
        _descriptor(
            method_id="qlora",
            display_name="QLoRA",
            summary="Trains LoRA adapters through a frozen runtime-native four-bit base.",
            lifecycle=MethodLifecycle.GATED_EXECUTABLE,
            selectable=True,
            parameter_scope="frozen-base-plus-adapter",
            parameterization="lora",
            base_storage="runtime-native-four-bit",
            compiler_id="transformers.peft-qlora.v2",
            export_kind="peft-adapter-safetensors",
            supported_backends=("cuda", "mps"),
            supported_distributions=("single", "ddp"),
            evidence_ids=("method.qlora.paper", "estimate.memory.v2"),
            pilot_requirement="Exact four-bit kernel capability, target inspection, measured preflight, and a bounded pilot are mandatory.",
            aliases=("4bit-lora",),
            runtime_bindings=(
                _runtime(
                    training_runtime="transformers-peft-cuda",
                    compute_backend="cuda",
                    compiler_id="transformers.peft-qlora.v2",
                    estimator_id="aptus-memory-v2",
                    export_kind="peft-adapter-safetensors",
                    supported_distributions=("single", "ddp"),
                ),
                _runtime(
                    training_runtime="mlx-lm",
                    compute_backend="mps",
                    compiler_id="mlx-lm.qlora.v1",
                    estimator_id="aptus-memory-mlx-v2",
                    export_kind="mlx-lm-adapter",
                    supported_distributions=("single",),
                ),
            ),
        ),
        _descriptor(
            method_id="dora",
            display_name="DoRA",
            summary="Separates weight magnitude from direction and applies a low-rank update to the direction.",
            lifecycle=MethodLifecycle.EXPERIMENTAL,
            selectable=False,
            parameter_scope="frozen-base-plus-weight-decomposed-adapter",
            parameterization="dora",
            base_storage="unquantized",
            compiler_id=None,
            export_kind=None,
            supported_backends=(),
            supported_distributions=(),
            evidence_ids=("method.dora.paper",),
            pilot_requirement="A pinned PEFT use_dora path must pass target-type, trainable-state, save, reload, and bounded pilot checks.",
            blocker="No Aptus compiler, calibrated estimator, or verified export/reload contract exists yet.",
        ),
        _descriptor(
            method_id="bitfit",
            display_name="BitFit",
            summary="Freezes the model except for an explicitly enumerated set of existing bias tensors.",
            lifecycle=MethodLifecycle.EXPERIMENTAL,
            selectable=False,
            parameter_scope="selected-existing-biases",
            parameterization="bias-only",
            base_storage="unquantized",
            compiler_id=None,
            export_kind=None,
            supported_backends=(),
            supported_distributions=(),
            evidence_ids=("method.bitfit.paper",),
            pilot_requirement="The exact pinned architecture must expose a non-empty bias set, and the selected-name digest, optimizer set, export, reload, and pilot must agree.",
            blocker="Many decoder models, including default Llama configurations, expose no eligible attention or MLP biases; Aptus has no bias-delta export contract yet.",
            aliases=("bias-only",),
        ),
        _descriptor(
            method_id="adalora",
            display_name="AdaLoRA",
            summary="Allocates an adaptive rank budget across pseudo-SVD low-rank parameter groups.",
            lifecycle=MethodLifecycle.EXPERIMENTAL,
            selectable=False,
            parameter_scope="adaptive-low-rank-parameter-groups",
            parameterization="adaptive-budget-lora",
            base_storage="unquantized",
            compiler_id=None,
            export_kind=None,
            supported_backends=(),
            supported_distributions=(),
            evidence_ids=("method.adalora.paper",),
            pilot_requirement="A pinned implementation must bind its initial rank, final budget, schedule, importance state, optimizer membership, checkpoint continuation, export, and reload.",
            blocker="Aptus does not yet model the changing trainable budget or preserve its scheduler and importance state across restart.",
        ),
        _descriptor(
            method_id="loreft",
            display_name="LoReFT",
            summary="Learns low-rank interventions on hidden representations at explicit layers and token positions.",
            lifecycle=MethodLifecycle.RESEARCH_ONLY,
            selectable=False,
            parameter_scope="representation-interventions",
            parameterization="low-rank-reft",
            base_storage="unquantized",
            compiler_id=None,
            export_kind=None,
            supported_backends=(),
            supported_distributions=(),
            evidence_ids=("method.loreft.paper",),
            pilot_requirement="A pinned pyreft runtime must bind component paths, layers, token positions, gradients, checkpoints, export, and reload.",
            blocker="The current trainer, collator, checkpoint, and PEFT export contracts do not represent interventions.",
            aliases=("low-rank-reft",),
        ),
        _descriptor(
            method_id="aflora",
            display_name="AFLoRA",
            summary="Dynamically scores and freezes low-rank parameter groups during training.",
            lifecycle=MethodLifecycle.RESEARCH_ONLY,
            selectable=False,
            parameter_scope="dynamic-low-rank-parameter-groups",
            parameterization="adaptive-freezing-lora",
            base_storage="unquantized",
            compiler_id=None,
            export_kind=None,
            supported_backends=(),
            supported_distributions=(),
            evidence_ids=("method.aflora.paper",),
            pilot_requirement="A pinned implementation must cross a freeze event and prove deterministic scores, optimizer membership, restart equivalence, export, and reload.",
            blocker="No maintained Aptus-compatible compiler or checkpoint contract has passed those checks.",
        ),
        _descriptor(
            method_id="bilora",
            display_name="BiLoRA",
            summary="Uses bilevel optimization over disjoint data partitions and a pseudo-SVD low-rank update.",
            lifecycle=MethodLifecycle.RESEARCH_ONLY,
            selectable=False,
            parameter_scope="bilevel-low-rank-parameter-groups",
            parameterization="pseudo-svd-lora",
            base_storage="unquantized",
            compiler_id=None,
            export_kind=None,
            supported_backends=(),
            supported_distributions=(),
            evidence_ids=("method.bilora.paper",),
            pilot_requirement="A dedicated two-optimizer loop must bind D1/D2 partitions, both states, restart semantics, export, and reload.",
            blocker="The generic Trainer path cannot express the paper's inner and outer optimization contract.",
        ),
        _descriptor(
            method_id="sharelora",
            display_name="ShareLoRA",
            summary="Shares one or both low-rank factors across shape-compatible layers.",
            lifecycle=MethodLifecycle.EXPERIMENTAL,
            selectable=False,
            parameter_scope="shared-low-rank-parameter-groups",
            parameterization="shared-factor-lora",
            base_storage="unquantized",
            compiler_id=None,
            export_kind=None,
            supported_backends=(),
            supported_distributions=(),
            evidence_ids=("method.sharelora.paper",),
            pilot_requirement="A pinned compiler must prove shape grouping, unique-versus-logical parameter accounting, serialization, reload, and distributed synchronization.",
            blocker="Aptus has no shared-module serializer or distributed ownership contract for shared factors.",
        ),
    )
}


def _validate_registry(values: Iterable[MethodDescriptor]) -> None:
    descriptors = tuple(values)
    selectable = {item.method_id for item in descriptors if item.selectable}
    expected = {item.value for item in Method}
    if selectable != expected:
        raise RuntimeError(
            "Method registry selectable IDs differ from the executable Method enum."
        )
    aliases: set[str] = set(METHOD_REGISTRY)
    compiler_ids: set[str] = set()
    for descriptor in descriptors:
        if descriptor.selectable:
            if descriptor.compiler_id in compiler_ids:
                raise RuntimeError(
                    f"Duplicate method compiler ID: {descriptor.compiler_id!r}"
                )
            compiler_ids.add(str(descriptor.compiler_id))
            if not set(descriptor.supported_backends).issubset(
                {item.value for item in Backend}
            ):
                raise RuntimeError("Method registry contains an unknown backend ID.")
            if not set(descriptor.supported_distributions).issubset(
                {item.value for item in Distribution}
            ):
                raise RuntimeError(
                    "Method registry contains an unknown distribution ID."
                )
            binding_keys: set[tuple[str, str]] = set()
            for binding in descriptor.runtime_bindings:
                key = (binding.training_runtime, binding.compute_backend)
                if key in binding_keys:
                    raise RuntimeError(
                        f"Duplicate method runtime binding: {descriptor.method_id} {key}."
                    )
                binding_keys.add(key)
                if binding.training_runtime not in {
                    item.value for item in TrainingRuntime
                }:
                    raise RuntimeError(
                        "Method registry contains an unknown runtime ID."
                    )
                if binding.compute_backend not in {item.value for item in Backend}:
                    raise RuntimeError(
                        "Runtime binding contains an unknown backend ID."
                    )
                if not set(binding.supported_distributions).issubset(
                    {item.value for item in Distribution}
                ):
                    raise RuntimeError(
                        "Runtime binding contains an unknown distribution ID."
                    )
        for alias in descriptor.aliases:
            normalized = alias.strip().lower()
            if not normalized or normalized in aliases:
                raise RuntimeError(f"Duplicate or invalid method alias: {alias!r}")
            aliases.add(normalized)


_validate_registry(METHOD_REGISTRY.values())


def method_descriptors() -> tuple[MethodDescriptor, ...]:
    return tuple(METHOD_REGISTRY.values())


def method_descriptor(method: str | Method) -> MethodDescriptor:
    method_id = method.value if isinstance(method, Method) else method.strip().lower()
    try:
        return METHOD_REGISTRY[method_id]
    except KeyError as error:
        raise ValueError(f"Unknown fine-tuning method: {method_id!r}") from error


def descriptor_for_compiler(compiler_id: str) -> MethodDescriptor:
    matches = tuple(
        item
        for item in METHOD_REGISTRY.values()
        if item.compiler_id == compiler_id
        or any(binding.compiler_id == compiler_id for binding in item.runtime_bindings)
    )
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous method compiler ID: {compiler_id!r}")
    return matches[0]


def runtime_binding(
    method: str | Method,
    *,
    training_runtime: str | TrainingRuntime,
    compute_backend: str | Backend,
) -> RuntimeBinding | None:
    descriptor = method_descriptor(method)
    runtime_id = (
        training_runtime.value
        if isinstance(training_runtime, TrainingRuntime)
        else training_runtime
    )
    backend_id = (
        compute_backend.value
        if isinstance(compute_backend, Backend)
        else compute_backend
    )
    return next(
        (
            item
            for item in descriptor.runtime_bindings
            if item.training_runtime == runtime_id
            and item.compute_backend == backend_id
        ),
        None,
    )


def selectable_method_descriptors() -> tuple[MethodDescriptor, ...]:
    return tuple(item for item in METHOD_REGISTRY.values() if item.selectable)


def selectable_method_ids() -> tuple[str, ...]:
    return tuple(item.method_id for item in selectable_method_descriptors())
