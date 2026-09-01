from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "aptus.training-plan.v6"
FORMULA_VERSION = "aptus-memory-v2"
MLX_FORMULA_VERSION = "aptus-memory-mlx-v2"
TRAINING_POLICY_VERSION = "aptus-training-policy-v1"
RUNTIME_CONTRACT_VERSION = "aptus.runtime-contract.v1"
MODEL_COMPATIBILITY_SCHEMA_VERSION = "aptus.model-compatibility.v2"
MODEL_INSPECTION_RECEIPT_SCHEMA_VERSION = "aptus.model-inspection-receipt.v1"
MODEL_POLICY_BINDING_SCHEMA_VERSION = "aptus.model-policy-binding.v1"
MODEL_POLICY_SNAPSHOT_SCHEMA_VERSION = "aptus.model-policy-snapshot.v1"
MODEL_POLICY_SNAPSHOT_PATH = "policy/model-policy-snapshot.v1.json"
_POLICY_SNAPSHOT_MODULE: Any | None = None


def _policy_snapshot_module():
    global _POLICY_SNAPSHOT_MODULE
    if _POLICY_SNAPSHOT_MODULE is not None:
        return _POLICY_SNAPSHOT_MODULE
    try:
        from . import policy_snapshot

        _POLICY_SNAPSHOT_MODULE = policy_snapshot
        return _POLICY_SNAPSHOT_MODULE
    except ImportError:
        source = Path(__file__).with_name("policy_snapshot.py")
        spec = importlib.util.spec_from_file_location(
            "aptus_portable_policy_snapshot", source
        )
        if spec is None or spec.loader is None:
            raise ImportError("Portable model policy snapshot module is unavailable.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _POLICY_SNAPSHOT_MODULE = module
        return _POLICY_SNAPSHOT_MODULE


_policy_snapshot_module()
CANDIDATE_STATUSES = {"feasible", "conditional", "infeasible", "unsupported"}
METHODS = {"full", "lora", "int8-lora", "qlora"}
DISTRIBUTIONS = {"single", "ddp", "fsdp"}
TRAINING_RUNTIMES = {"transformers-peft-cuda", "mlx-lm", "pytorch-mps"}
EVIDENCE_REQUIREMENTS = {"pilot-required", "implementation-required"}
QWEN3_MOE_FAMILY = "qwen3_moe"
QWEN3_MOE_MODEL_TYPE = "qwen3_moe"
QWEN3_MOE_ARCHITECTURE = "Qwen3MoeForCausalLM"
GEMMA4_MOE_FAMILY = "gemma4_moe"
GEMMA4_MOE_MODEL_TYPE = "gemma4_text"
GEMMA4_MOE_ARCHITECTURE = "Gemma4ForConditionalGeneration"
RECEIPT_FACT_FIELDS = {
    "architecture",
    "context_length",
    "family",
    "hidden_size",
    "intermediate_size",
    "layers",
    "license_name",
    "model_type",
    "moe",
    "quantization_bits",
    "quantization_layout",
}
PROVENANCE_KINDS = {
    "measured",
    "provider-declared",
    "user-attested",
    "inferred",
    "unknown",
}
INSPECTION_PROVENANCE_KINDS = {"provider-declared", "inferred"}
COMPATIBILITY_SUBJECT_FACT_FIELDS = {
    "architecture",
    "family",
    "layers",
    "model_type",
    "moe",
    "quantization_bits",
    "quantization_layout",
}
POLICY_DECISION_SOURCES = {"provider-inspection", "user-attested"}
DENSE_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
MODEL_TARGET_MODULES = {
    family: DENSE_TARGET_MODULES
    for family in ("gemma", "gemma4", "llama", "mistral", "qwen")
}
MODEL_TARGET_MODULES[QWEN3_MOE_FAMILY] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]
MODEL_TARGET_MODULES[GEMMA4_MOE_FAMILY] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]
MLX_SPARSE_LAYER_ADAPTER_TARGETS = frozenset({"k_proj", "v_proj"})
MLX_SPARSE_ADAPTER_FAMILIES = frozenset({"gemma4", GEMMA4_MOE_FAMILY})


MLX_PACKED_CHECKPOINT_OVERHEAD_FLOOR_BYTES = 2 * 1024**2
MLX_PACKED_CHECKPOINT_OVERHEAD_RATIO = 0.0001


def mlx_packed_checkpoint_overhead_limit(expected_packed: int) -> int:
    """Return the fail-closed container-overhead ceiling.

    Gemma 4 keeps BF16 RMSNorms that the 4-bit plus groupwise-metadata formula
    does not price. The two-mebibyte floor covers that residual plus headers.
    Unused vision/audio Hub payloads are excluded from the observed bytes and
    must not be absorbed as overhead.
    """

    if not _positive_int(expected_packed):
        raise ValueError(
            "MLX packed-checkpoint overhead limit requires a positive packed size."
        )
    return max(
        MLX_PACKED_CHECKPOINT_OVERHEAD_FLOOR_BYTES,
        round(expected_packed * MLX_PACKED_CHECKPOINT_OVERHEAD_RATIO),
    )


def mlx_trainable_target_instance_total(
    planned_targets: Any,
    layer_count: Any,
    target_instance_counts: Any,
    *,
    family: Any,
) -> int:
    """Return the bound adapter-instance total for a loaded MLX model.

    Each planned target must appear at least once and at most once per
    transformer layer. Default is every planned target in every layer.
    Only Gemma 4 families may omit k_proj/v_proj together on KV-shared
    layers, and may omit v_proj alone on k-equals-v layers when k_proj
    still appears at least once and v_count does not exceed k_count.
    """

    if (
        not isinstance(planned_targets, (list, tuple))
        or not planned_targets
        or any(not isinstance(target, str) or not target for target in planned_targets)
        or not isinstance(layer_count, int)
        or isinstance(layer_count, bool)
        or layer_count <= 0
        or not isinstance(family, str)
        or not family
        or not isinstance(target_instance_counts, Mapping)
        or set(target_instance_counts) != set(planned_targets)
        or len(target_instance_counts) != len(planned_targets)
    ):
        raise ValueError("MLX trainable-target instance counts are not exact.")
    allow_sparse_kv = family in MLX_SPARSE_ADAPTER_FAMILIES
    total = 0
    for target in planned_targets:
        count = target_instance_counts.get(target)
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
            or count > layer_count
        ):
            raise ValueError("MLX trainable-target instance counts are not exact.")
        sparse_ok = allow_sparse_kv and target in MLX_SPARSE_LAYER_ADAPTER_TARGETS
        if not sparse_ok and count != layer_count:
            raise ValueError(
                "MLX adapters must cover every transformer layer unless the "
                "Gemma 4 family omits k_proj/v_proj on KV-shared layers."
            )
        total += count
    if allow_sparse_kv:
        k_count = target_instance_counts.get("k_proj")
        v_count = target_instance_counts.get("v_proj")
        if (
            isinstance(k_count, int)
            and not isinstance(k_count, bool)
            and isinstance(v_count, int)
            and not isinstance(v_count, bool)
            and v_count > k_count
        ):
            raise ValueError(
                "Gemma 4 k_proj and v_proj adapter counts cannot omit k_proj "
                "while keeping v_proj."
            )
    return total


MOE_TOPOLOGY_FIELDS = (
    "expert_count",
    "experts_per_token",
    "expert_intermediate_size",
    "decoder_sparse_step",
    "mlp_only_layers",
    "shared_expert_intermediate_size",
)
QUANTIZATION_LAYOUT_FIELDS = (
    "default_bits",
    "default_group_size",
    "module_overrides",
)
QUANTIZATION_OVERRIDE_FIELDS = ("module_path", "bits", "group_size")

# Evidence record prose is executable-plan input because it is displayed and
# copied into generated bundles. These digests mirror aptus.evidence and keep
# the portable validator independent from the host package.
EVIDENCE_RECORD_SHA256 = {
    "method.full.transformers": (
        "000e5bd35b691c5c6cc304241bb1caac902a6c82d394413cc45611e3b5f4fdb5"
    ),
    "method.lora.paper": (
        "5d4deee885b48352282ec562e8943e519825e7d793204bd550bfefc22c474d99"
    ),
    "method.qlora.paper": (
        "79bcfa63d5378e33c8d65e5ba0179c18cf1dea42ec252a84aa3bbae198a2589b"
    ),
    "method.bitsandbytes.int8": (
        "569146a786fb72236e3e23f05fa17d630588f02d5e2c9ad32a8f1f5abdf599f5"
    ),
    "estimate.memory.v2": (
        "575dfbd577b113e41b52ce88cba981cd024aa06dd3e3d9aa5c97a21ec8133552"
    ),
    "policy.qwen3-moe.mlx-qlora.v1": (
        "3f40dfd230170bed8ad89ada2969d13b5e3b711f7bfeb30f0f122c5e4f29a844"
    ),
    "admission.qwen3-30b-a3b.memory-blocked.2026-07-28": (
        "8def6f7dd2591edd1ac5dbedb49b49a4740a65b6398ad56669f32423f7c56ac2"
    ),
    "policy.qwen2-24l.mlx-qlora.v1": (
        "c7a097e7140ade3f48a9cf9cfbc01cb95dd31b5124e7e9d10d90b8b0f3b63264"
    ),
    "policy.gemma4.mlx.v1": (
        "dec03ae4eee5ea8671ebd632907ca0d1d834a73db4192654813cc1e6bad81c59"
    ),
    "policy.gemma4-unified.mlx.v1": (
        "8fba1a85e361db082f72f8b5b417b86461ae7a71bb5a99c2149065abb1868f4e"
    ),
    "policy.gemma4-moe.mlx.v1": (
        "983c9fc8258e0a1b051609cc335b970aea118dba4eca1147b085914003593781"
    ),
    "runtime.qwen2-0.5b.mlx-qlora.2026-07-27": (
        "2b1905044b84b3473d536fbe73af31841af15857523f458bb58a6d34d89447bc"
    ),
}
METHOD_EVIDENCE_IDS = {
    "full": ["method.full.transformers", "estimate.memory.v2"],
    "lora": ["method.lora.paper", "estimate.memory.v2"],
    "int8-lora": [
        "method.lora.paper",
        "method.bitsandbytes.int8",
        "estimate.memory.v2",
    ],
    "qlora": ["method.qlora.paper", "estimate.memory.v2"],
}

# This table is intentionally self-contained because plan_contract.py is copied
# into every generated bundle. It mirrors the executable RuntimeBinding entries
# in aptus.methods and lets a bundle reject invented compiler identities without
# importing the Aptus package at validation time.
RUNTIME_BINDING_IDENTITIES = {
    ("full", "transformers-peft-cuda", "cuda"): (
        "transformers.full.v2",
        "aptus-memory-v2",
        "full-model-safetensors",
        "pilot-required",
    ),
    ("lora", "transformers-peft-cuda", "cuda"): (
        "transformers.peft-lora.v2",
        "aptus-memory-v2",
        "peft-adapter-safetensors",
        "pilot-required",
    ),
    ("lora", "mlx-lm", "mps"): (
        "mlx-lm.lora.v1",
        "aptus-memory-mlx-v2",
        "mlx-lm-adapter",
        "pilot-required",
    ),
    ("int8-lora", "transformers-peft-cuda", "cuda"): (
        "transformers.peft-int8-lora.v2",
        "aptus-memory-v2",
        "peft-adapter-safetensors",
        "pilot-required",
    ),
    ("qlora", "transformers-peft-cuda", "cuda"): (
        "transformers.peft-qlora.v2",
        "aptus-memory-v2",
        "peft-adapter-safetensors",
        "pilot-required",
    ),
    ("qlora", "mlx-lm", "mps"): (
        "mlx-lm.qlora.v1",
        "aptus-memory-mlx-v2",
        "mlx-lm-adapter",
        "pilot-required",
    ),
}
UNAVAILABLE_RUNTIME_IDENTITY = (
    None,
    "unavailable",
    None,
    "implementation-required",
)


class StaleModelPolicyError(ValueError):
    """An internally valid saved plan no longer matches the current policy."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_json_object(value: str | bytes, label: str) -> dict[str, Any]:
    """Parse one portable JSON object without leaking parser resource errors."""

    try:
        parsed = json.loads(value)
    except (RecursionError, ValueError):
        raise ValueError(f"{label} is unreadable or invalid JSON.") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    """Load one portable JSON object without leaking parser resource errors."""

    try:
        value = path.read_bytes()
    except OSError:
        raise ValueError(f"{label} is unreadable or invalid JSON.") from None
    return parse_json_object(value, label)


def bundle_fingerprint(root: Path) -> str:
    manifest = root / "bundle-manifest.json"
    if manifest.is_file():
        return sha256_file(manifest)
    digest = hashlib.sha256()
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name != "validation-report.json"
    ):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\n")
    return digest.hexdigest()


def validate_bundle_manifest(root: Path) -> tuple[str, ...]:
    """Verify the immutable file set bound by bundle-manifest.json."""

    errors: list[str] = []
    if root.is_symlink():
        errors.append("Bundle root cannot be a symlink.")
    try:
        symlinks = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_symlink()
        )
    except OSError as error:
        return (f"Bundle tree could not be inspected safely: {error}",)
    if symlinks:
        errors.append("Bundle tree contains symlink(s): " + ", ".join(symlinks) + ".")
    manifest_path = root / "bundle-manifest.json"
    if not manifest_path.is_file():
        return ("Bundle manifest is missing.",)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as error:
        return (f"Bundle manifest is invalid JSON: {error}",)
    if not isinstance(manifest, dict):
        errors.append("Bundle manifest must be a JSON object.")
        return tuple(errors)
    if manifest.get("schema_version") != "aptus.bundle.v3":
        errors.append("Bundle manifest schema must be aptus.bundle.v3.")
    expected_artifact_fingerprint = os.environ.get(
        "APTUS_EXPECTED_ARTIFACT_FINGERPRINT"
    )
    if expected_artifact_fingerprint is not None and (
        not _valid_sha256(expected_artifact_fingerprint)
        or sha256_file(manifest_path) != expected_artifact_fingerprint
    ):
        errors.append(
            "Bundle manifest does not match the host-authorized artifact fingerprint."
        )
    authorized_policy_snapshot = os.environ.get(
        "APTUS_AUTHORIZED_MODEL_POLICY_SNAPSHOT_SHA256"
    )
    if authorized_policy_snapshot is not None and (
        not _valid_sha256(authorized_policy_snapshot)
        or manifest.get("policy_snapshot_sha256") != authorized_policy_snapshot
    ):
        errors.append(
            "Bundle model policy snapshot does not match the host-authorized digest."
        )
    plan_path = root / "plan.json"
    if not plan_path.is_file() or manifest.get("plan_sha256") != sha256_file(plan_path):
        errors.append("Bundle manifest plan digest does not match plan.json.")
    snapshot_relative = manifest.get("policy_snapshot_path")
    if snapshot_relative != MODEL_POLICY_SNAPSHOT_PATH:
        errors.append(
            f"Bundle manifest policy snapshot path must be {MODEL_POLICY_SNAPSHOT_PATH}."
        )
    snapshot_path = root / MODEL_POLICY_SNAPSHOT_PATH
    snapshot_digest = manifest.get("policy_snapshot_sha256")
    if not snapshot_path.is_file():
        errors.append("Bundle model policy snapshot is missing.")
    else:
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if not isinstance(snapshot, Mapping):
                raise ValueError("snapshot must be an object")
            actual_snapshot_digest = _policy_snapshot_sha256(snapshot)
            if (
                snapshot_path.read_bytes()
                != _policy_snapshot_module().model_policy_snapshot_bytes(snapshot)
            ):
                errors.append("Bundle model policy snapshot is not canonical JSON.")
            if snapshot_digest != actual_snapshot_digest:
                errors.append("Bundle model policy snapshot digest does not match.")
            if plan_path.is_file():
                plan_value = json.loads(plan_path.read_text(encoding="utf-8"))
                if (
                    not isinstance(plan_value, Mapping)
                    or plan_value.get("model_policy_snapshot_sha256")
                    != actual_snapshot_digest
                ):
                    errors.append(
                        "Bundle plan does not bind the emitted model policy snapshot."
                    )
        except (OSError, ValueError, RecursionError) as error:
            errors.append(f"Bundle model policy snapshot is malformed: {error}")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        errors.append("Bundle manifest files must be a non-empty list.")
        return tuple(errors)
    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            errors.append("Every bundle manifest entry requires a path.")
            continue
        relative = item["path"]
        relative_path = Path(relative)
        if (
            relative in seen
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            errors.append(f"Unsafe or duplicate bundle manifest path: {relative}.")
            continue
        seen.add(relative)
        path = root / relative_path
        if path.is_symlink():
            errors.append(f"Manifested file cannot be a symlink: {relative}.")
            continue
        if not path.is_file():
            errors.append(f"Manifested file is missing: {relative}.")
            continue
        if (
            item.get("sha256") != sha256_file(path)
            or item.get("size_bytes") != path.stat().st_size
        ):
            errors.append(f"Manifested file changed: {relative}.")
    mutable_files = {
        ".validation-report.lock",
        "model-data-evidence.json",
        "validation-report.json",
        "preflight-metrics.json",
    }
    mutable_prefixes = ("pilot-output/", "runs/")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "bundle-manifest.json"
    }
    unexpected = sorted(
        relative
        for relative in actual - seen
        if relative not in mutable_files and not relative.startswith(mutable_prefixes)
    )
    if unexpected:
        errors.append(
            "Bundle contains unmanifested input file(s): " + ", ".join(unexpected) + "."
        )
    return tuple(errors)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _known_text(value: Any, choices: set[str]) -> bool:
    """Return membership for an untrusted JSON scalar without raising."""

    return isinstance(value, str) and value in choices


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _content_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:20]


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_revision(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and 40 <= len(value) <= 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _valid_content_id(value: Any, *, prefix: str) -> bool:
    suffix = value[len(prefix) :] if isinstance(value, str) else ""
    return bool(
        isinstance(value, str)
        and value.startswith(prefix)
        and len(suffix) == 20
        and all(character in "0123456789abcdef" for character in suffix)
    )


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _compatibility_subject_payload(model: Mapping[str, Any]) -> dict[str, Any]:
    family = model.get("family")
    if isinstance(family, str) and family != family.lower():
        raise ValueError("Model family must use its canonical lowercase identity.")
    quantization_layout = (
        _normalized_quantization_layout(model.get("quantization_layout"))
        if model.get("quantization_layout") is not None
        else None
    )
    moe = _normalized_moe(model.get("moe")) if model.get("moe") is not None else None
    return {
        "family": family,
        "model_type": model.get("model_type"),
        "architecture": model.get("architecture"),
        "layers": model.get("layers"),
        "quantization_bits": model.get("quantization_bits"),
        "quantization_layout": quantization_layout,
        "moe": moe,
        "fact_errors": [],
    }


def _policy_decision_identity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "schema_version",
            "subject_facts_sha256",
            "kind",
            "family",
            "policy_id",
            "policy_version",
            "paths",
            "reason_codes",
            "evidence_ids",
        )
    }


def _semantic_policy_decision(value: Any) -> Any:
    """Return policy identity fields while excluding explanatory prose."""

    if not isinstance(value, Mapping):
        return value
    return {
        key: value.get(key)
        for key in (
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
        )
    }


def _semantic_inspection_receipt(value: Any) -> Any:
    """Bind the full receipt except explanatory decision prose."""

    if not isinstance(value, Mapping):
        return value
    return {
        key: (_semantic_policy_decision(item) if key == "decision" else item)
        for key, item in value.items()
    }


def _current_model_policy_decision(
    model: Mapping[str, Any],
    policy_snapshot: Mapping[str, Any] | None = None,
    *,
    confirm_unreviewed_runtime: bool = False,
) -> dict[str, Any]:
    """Evaluate model compatibility from the versioned portable snapshot."""

    snapshot = (
        policy_snapshot
        if policy_snapshot is not None
        else _policy_snapshot_for_validation()
    )
    module = _policy_snapshot_module()
    subject = _compatibility_subject_payload(model)
    decision = module.evaluate_model_policy_snapshot(snapshot, subject)
    return module.apply_operator_unreviewed_runtime_confirm(
        snapshot,
        subject,
        decision,
        confirmed=confirm_unreviewed_runtime,
    )


def _policy_snapshot_for_validation(
    *,
    root: Path | None = None,
    policy_snapshot: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if policy_snapshot is not None:
        return policy_snapshot
    if __package__:
        try:
            from .model_compatibility import current_model_policy_snapshot

            return current_model_policy_snapshot()
        except ImportError as error:
            raise ValueError(
                "Current host model policy snapshot is unavailable."
            ) from error
    if root is not None:
        path = root / MODEL_POLICY_SNAPSHOT_PATH
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                raise ValueError("Model policy snapshot must be an object.")
            return value
    raise ValueError("Portable model policy snapshot is unavailable.")


def _policy_snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    return _policy_snapshot_module().model_policy_snapshot_sha256(snapshot)


def require_current_model_policy_snapshot(
    plan_value: Any,
    *,
    historical_policy_snapshot: Mapping[str, Any] | None = None,
) -> None:
    """Require a bundle-bound snapshot to match the installed host registry.

    Bundle-manifest validation establishes that the recorded digest belongs to
    the bundle's frozen snapshot. When that digest is no longer current, the
    complete historical plan chain is checked before the mismatch is classified
    as a replanning condition rather than malformed state.
    """

    if not isinstance(plan_value, Mapping):
        raise ValueError("Persisted plan must be an object.")
    recorded_digest = plan_value.get("model_policy_snapshot_sha256")
    if not isinstance(recorded_digest, str):
        raise ValueError("Persisted plan must bind a model policy snapshot digest.")
    snapshot = _policy_snapshot_for_validation()
    if recorded_digest == _policy_snapshot_sha256(snapshot):
        return
    require_current_model_policy(
        plan_value,
        policy_snapshot=snapshot,
        historical_policy_snapshot=historical_policy_snapshot,
    )


def require_current_model_policy(
    plan_value: Any,
    *,
    policy_snapshot: Mapping[str, Any] | None = None,
    historical_policy_snapshot: Mapping[str, Any] | None = None,
) -> None:
    """Raise a distinct error when a same-schema plan needs policy replanning.

    Malformed or tampered v5 state remains a plain ``ValueError``. A decision is
    classified as stale only after its full historical receipt, candidate,
    recommendation, evidence, and plan identity chain validates independently
    of today's policy registry.
    """

    if not isinstance(plan_value, Mapping):
        raise ValueError("Persisted plan must be an object.")
    if plan_value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Persisted plan schema must be {SCHEMA_VERSION}.")
    model = plan_value.get("model")
    decision = plan_value.get("model_policy_decision")
    if not isinstance(model, Mapping) or not isinstance(decision, Mapping):
        raise ValueError("Persisted v5 plans require model policy state.")
    try:
        snapshot = _policy_snapshot_for_validation(policy_snapshot=policy_snapshot)
        expected = _current_model_policy_decision(
            model,
            snapshot,
            confirm_unreviewed_runtime=bool(
                (plan_value.get("target") or {}).get("unreviewed_runtime_confirmed")
            )
            if isinstance(plan_value.get("target"), Mapping)
            else False,
        )
    except (
        AttributeError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Saved model policy could not be recomputed from malformed model facts."
        ) from error
    reason = decision.get("reason")
    if not isinstance(reason, str) or not reason.strip() or reason != reason.strip():
        raise ValueError("Saved model policy reason must be unpadded non-empty text.")
    required_decision_fields = {
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
    }
    try:
        recorded_decision_id = (
            "compat_" + _sha256_json(_policy_decision_identity_payload(decision))[:20]
        )
    except (
        AttributeError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError("Saved model policy state is malformed.") from error
    if (
        set(decision) != required_decision_fields
        or decision.get("schema_version") != MODEL_COMPATIBILITY_SCHEMA_VERSION
        or not _valid_content_id(decision.get("decision_id"), prefix="compat_")
        or decision.get("decision_id") != recorded_decision_id
        or decision.get("subject_facts_sha256") != expected.get("subject_facts_sha256")
    ):
        raise ValueError(
            "Saved model policy state is malformed, tampered, or inconsistent "
            "with the current model facts."
        )
    try:
        snapshot_digest = _policy_snapshot_sha256(snapshot)
        decision_is_current = _semantic_policy_decision(
            decision
        ) == _semantic_policy_decision(expected)
    except (
        AttributeError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError("Saved model policy state is malformed.") from error
    if (
        plan_value.get("model_policy_snapshot_sha256") == snapshot_digest
        and decision_is_current
    ):
        return
    try:
        historical_errors = _validate_plan_payload_impl(
            plan_value,
            verify_dataset=False,
            expected_policy_decision_override=decision,
            policy_snapshot=(historical_policy_snapshot or snapshot),
            enforce_current_policy=historical_policy_snapshot is not None,
            allow_unavailable_policy_provenance=(historical_policy_snapshot is None),
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Saved model policy dependencies are malformed or tampered."
        ) from error
    if historical_errors:
        raise ValueError(
            "Saved model policy dependencies are malformed or tampered: "
            + "; ".join(historical_errors[:3])
        )
    raise StaleModelPolicyError(
        "The saved plan uses policy semantics that are no longer current; "
        "replan_required."
    )


def _receipt_fact_value(field: str, model: Mapping[str, Any]) -> Any:
    value = model.get(field)
    if field == "quantization_layout" and value is not None:
        return _normalized_quantization_layout(value)
    if field == "moe" and value is not None:
        return _normalized_moe(value)
    return value


def _validate_receipt_model_provenance(
    *,
    model: Mapping[str, Any],
    provenance_summary: list[Mapping[str, Any]],
    receipt_id: Any,
    errors: list[str],
) -> None:
    model_provenance = model.get("provenance")
    if not isinstance(model_provenance, Mapping):
        errors.append("Receipt-backed model provenance must be an object.")
        return
    expected_fields = RECEIPT_FACT_FIELDS | {"parameters", "training_allowed"}
    if set(model_provenance) != expected_fields:
        errors.append(
            "Receipt-backed model provenance must name every observed planning fact "
            "plus parameters and training_allowed."
        )
    summary_by_field = {
        item.get("field"): item
        for item in provenance_summary
        if isinstance(item.get("field"), str)
    }
    provenance_fields = {
        "kind",
        "source",
        "observed_at",
        "digest",
        "detail",
    }
    for field in expected_fields:
        value = model_provenance.get(field)
        if not isinstance(value, Mapping):
            errors.append(f"Model provenance for {field} must be an object.")
            continue
        if set(value) != provenance_fields:
            errors.append(
                f"Model provenance for {field} must contain the exact v1 fields."
            )
        observed = summary_by_field.get(field)
        if observed is not None:
            expected = {
                "kind": observed.get("kind"),
                "source": observed.get("source"),
                "observed_at": observed.get("observed_at"),
                "digest": receipt_id,
                "detail": (
                    "Provider observation at immutable revision "
                    f"{observed.get('resolved_revision')}."
                ),
            }
            if value != expected:
                errors.append(
                    f"Model provenance for {field} does not match its inspection receipt."
                )
        elif value.get("kind") != "user-attested":
            errors.append(f"Model provenance for {field} must remain user-attested.")


def _validate_user_attested_model_provenance(
    *, model: Mapping[str, Any], errors: list[str]
) -> None:
    provenance = model.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {"all"}:
        errors.append(
            "Receipt-free plans require one all-fields user-attested model provenance record."
        )
        return
    record = provenance.get("all")
    if not isinstance(record, Mapping) or set(record) != {
        "kind",
        "source",
        "observed_at",
        "digest",
        "detail",
    }:
        errors.append(
            "User-attested model provenance must contain the exact v1 fields."
        )
        return
    source = record.get("source")
    if record.get("kind") != "user-attested":
        errors.append("Receipt-free model provenance must remain user-attested.")
    if not isinstance(source, str) or not source or source != source.strip():
        errors.append("User-attested model provenance source must be unpadded text.")


def _validate_inspection_receipt(
    receipt_value: Any,
    *,
    model: Mapping[str, Any],
    expected_decision: Mapping[str, Any],
    policy_snapshot: Mapping[str, Any] | None,
    allow_unavailable_policy_provenance: bool,
    errors: list[str],
) -> Mapping[str, Any] | None:
    if receipt_value is None:
        return None
    if not isinstance(receipt_value, Mapping):
        errors.append("Plan inspection_receipt must be an object or null.")
        return None
    receipt = receipt_value
    required_receipt_fields = {
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
    }
    if set(receipt) != required_receipt_fields:
        errors.append("Plan inspection receipt must contain the exact v1 fields.")
    if receipt.get("schema_version") != MODEL_INSPECTION_RECEIPT_SCHEMA_VERSION:
        errors.append(
            "Plan inspection receipt schema must be "
            f"{MODEL_INSPECTION_RECEIPT_SCHEMA_VERSION}."
        )
    if receipt.get("model_id") != model.get("model_id"):
        errors.append("Plan inspection receipt model ID does not match model facts.")
    if (
        not _valid_revision(receipt.get("resolved_revision"))
        or str(receipt.get("resolved_revision")).lower()
        != str(model.get("revision")).lower()
    ):
        errors.append("Plan inspection receipt revision does not match model facts.")
    receipt_decision = receipt.get("decision")
    receipt_reason = (
        receipt_decision.get("reason")
        if isinstance(receipt_decision, Mapping)
        else None
    )
    if not isinstance(receipt_decision, Mapping) or set(receipt_decision) != {
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
    }:
        errors.append(
            "Plan inspection receipt decision must contain the exact v2 fields."
        )
    if (
        not isinstance(receipt_reason, str)
        or not receipt_reason.strip()
        or receipt_reason != receipt_reason.strip()
    ):
        errors.append(
            "Plan inspection receipt decision reason must be unpadded non-empty text."
        )
    if _semantic_policy_decision(receipt_decision) != _semantic_policy_decision(
        expected_decision
    ):
        errors.append(
            "Plan inspection receipt decision is stale or differs from current policy."
        )
    if not _valid_timestamp(receipt.get("evaluated_at")):
        errors.append(
            "Plan inspection receipt evaluated_at must be timezone-aware ISO-8601."
        )

    provenance_value = receipt.get("provenance_summary")
    provenance: list[Mapping[str, Any]] = []
    provenance_fields: list[str] = []
    if not isinstance(provenance_value, list) or not provenance_value:
        errors.append("Plan inspection receipt requires a non-empty provenance list.")
    else:
        for index, item in enumerate(provenance_value):
            name = f"Inspection provenance {index}"
            if not isinstance(item, Mapping):
                errors.append(f"{name} must be an object.")
                continue
            provenance.append(item)
            if set(item) != {
                "field",
                "kind",
                "source",
                "observed_at",
                "resolved_revision",
            }:
                errors.append(f"{name} must contain the exact v1 fields.")
            field = item.get("field")
            if not isinstance(field, str) or field not in RECEIPT_FACT_FIELDS:
                errors.append(f"{name} names an unsupported planning fact.")
            else:
                provenance_fields.append(field)
            if not _known_text(item.get("kind"), INSPECTION_PROVENANCE_KINDS):
                errors.append(f"{name} kind must be provider-declared or inferred.")
            source = item.get("source")
            if not isinstance(source, str) or not source or source != source.strip():
                errors.append(f"{name} source must be unpadded text.")
            if not _valid_timestamp(item.get("observed_at")):
                errors.append(f"{name} observed_at must be timezone-aware ISO-8601.")
            if (
                not _valid_revision(item.get("resolved_revision"))
                or str(item.get("resolved_revision")).lower()
                != str(receipt.get("resolved_revision")).lower()
            ):
                errors.append(f"{name} revision does not match the receipt.")
    if provenance_fields != sorted(set(provenance_fields)):
        errors.append(
            "Plan inspection receipt provenance fields must be sorted and unique."
        )
    required_subject_fields = {
        field
        for field in COMPATIBILITY_SUBJECT_FACT_FIELDS
        if model.get(field) is not None
    }
    missing_subject_fields = required_subject_fields.difference(provenance_fields)
    if missing_subject_fields:
        errors.append(
            "Plan inspection receipt provenance does not cover compatibility "
            "subject facts: " + ", ".join(sorted(missing_subject_fields)) + "."
        )
    if not any(
        item.get("field") in required_subject_fields
        and item.get("kind") == "provider-declared"
        for item in provenance
    ):
        errors.append(
            "Plan inspection receipt compatibility facts require at least one "
            "provider-declared observation."
        )

    observed_facts = {
        field: _receipt_fact_value(field, model) for field in provenance_fields
    }
    expected_observed_digest = _sha256_json(observed_facts)
    if (
        not _valid_sha256(receipt.get("observed_facts_sha256"))
        or receipt.get("observed_facts_sha256") != expected_observed_digest
    ):
        errors.append(
            "Plan inspection receipt observed-facts digest does not match model facts."
        )

    has_registered_policy = expected_decision.get("policy_id") is not None
    expected_requirement = "provider-declared" if has_registered_policy else None
    required_policy_fields: set[str] = set()
    policy_definition_unavailable = False
    if has_registered_policy:
        policies = (
            policy_snapshot.get("policies")
            if isinstance(policy_snapshot, Mapping)
            else None
        )
        policy = (
            next(
                (
                    item
                    for item in policies
                    if isinstance(item, Mapping)
                    and item.get("policy_id") == expected_decision.get("policy_id")
                    and item.get("policy_version")
                    == expected_decision.get("policy_version")
                ),
                None,
            )
            if isinstance(policies, list)
            else None
        )
        fields = (
            policy.get("required_provenance_fields")
            if isinstance(policy, Mapping)
            else None
        )
        if policy is None:
            policy_definition_unavailable = True
            if not allow_unavailable_policy_provenance:
                errors.append(
                    "Plan inspection receipt policy provenance requirements are unavailable."
                )
        elif not isinstance(fields, list) or any(
            not isinstance(field, str) for field in fields
        ):
            errors.append(
                "Plan inspection receipt policy provenance requirements are unavailable."
            )
        else:
            required_policy_fields = set(fields)
    if (
        has_registered_policy
        and policy_definition_unavailable
        and allow_unavailable_policy_provenance
    ):
        # A standalone stale plan does not carry its frozen snapshot payload.
        # Conservatively require every non-null compatibility field except the
        # normalized family alias to remain provider-declared. This matches the
        # reviewed policies and refuses a stale classification when the missing
        # historical definition cannot prove an equal or narrower requirement.
        fallback_required_fields = required_subject_fields.difference({"family"})
        expected_requirement_met = bool(
            fallback_required_fields
            and fallback_required_fields.issubset(provenance_fields)
            and all(
                item.get("kind") == "provider-declared"
                for item in provenance
                if _known_text(item.get("field"), fallback_required_fields)
            )
        )
    else:
        expected_requirement_met = bool(
            has_registered_policy
            and required_policy_fields
            and required_policy_fields.issubset(provenance_fields)
            and all(
                item.get("kind") == "provider-declared"
                for item in provenance
                if _known_text(item.get("field"), required_policy_fields)
            )
        )
    if receipt.get("provenance_requirement") != expected_requirement:
        errors.append("Plan inspection receipt provenance requirement is stale.")
    if not isinstance(receipt.get("provenance_requirement_met"), bool) or (
        receipt.get("provenance_requirement_met") is not expected_requirement_met
    ):
        errors.append("Plan inspection receipt provenance result is inconsistent.")
    if expected_decision.get("kind") == "path-matched" and not expected_requirement_met:
        errors.append(
            "A matched provider policy receipt requires provider-declared path facts."
        )

    receipt_identity = {
        "schema_version": MODEL_INSPECTION_RECEIPT_SCHEMA_VERSION,
        "model_id": receipt.get("model_id"),
        "resolved_revision": str(receipt.get("resolved_revision")).lower(),
        "observed_facts_sha256": receipt.get("observed_facts_sha256"),
        "decision_id": expected_decision.get("decision_id"),
        "provenance_summary": provenance_value,
        "provenance_requirement": receipt.get("provenance_requirement"),
        "provenance_requirement_met": receipt.get("provenance_requirement_met"),
        "evaluated_at": receipt.get("evaluated_at"),
    }
    expected_receipt_id = "receipt_" + _sha256_json(receipt_identity)[:20]
    if (
        not _valid_content_id(receipt.get("receipt_id"), prefix="receipt_")
        or receipt.get("receipt_id") != expected_receipt_id
    ):
        errors.append(
            "Plan inspection receipt immutable ID does not match its content."
        )
    _validate_receipt_model_provenance(
        model=model,
        provenance_summary=provenance,
        receipt_id=receipt.get("receipt_id"),
        errors=errors,
    )
    return receipt


def _matching_policy_path(
    decision: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    paths = decision.get("paths")
    if decision.get("kind") != "path-matched" or not isinstance(paths, list):
        return None
    return next(
        (
            path
            for path in paths
            if isinstance(path, Mapping)
            and path.get("method") == candidate.get("method")
            and path.get("distribution") == candidate.get("distribution")
            and path.get("target_modules") == candidate.get("target_modules")
            and path.get("runtime_contract") == candidate.get("runtime_contract")
        ),
        None,
    )


def _expected_policy_binding(
    *,
    decision: Mapping[str, Any],
    path: Mapping[str, Any],
    source: str,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    decision_evidence = decision.get("evidence_ids")
    path_evidence = path.get("evidence_ids")
    evidence_ids = list(
        dict.fromkeys(
            (decision_evidence if isinstance(decision_evidence, list) else [])
            + (path_evidence if isinstance(path_evidence, list) else [])
        )
    )
    return {
        "schema_version": MODEL_POLICY_BINDING_SCHEMA_VERSION,
        "decision_id": decision.get("decision_id"),
        "subject_facts_sha256": decision.get("subject_facts_sha256"),
        "policy_id": decision.get("policy_id"),
        "policy_version": decision.get("policy_version"),
        "path_id": path.get("path_id"),
        "source": source,
        "inspection_receipt_id": (
            receipt.get("receipt_id") if receipt is not None else None
        ),
        "reason_codes": decision.get("reason_codes"),
        "evidence_ids": evidence_ids,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _model_config_source(config: Mapping[str, Any]) -> Mapping[str, Any]:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, Mapping) else config


def _config_declares_moe_topology_field(source: Mapping[str, Any], name: str) -> bool:
    """Return whether a config field is a real MoE declaration.

    Empty `mlp_only_layers` is a common dense Hub default. Inspect treats it as
    not declared; the train gate must not call that topology.
    """

    if name not in source:
        return False
    value = source.get(name)
    if value is None:
        return False
    if name == "mlp_only_layers":
        return isinstance(value, list) and bool(value)
    return True


def _config_first(config: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = config.get(name)
        if value is not None:
            return value
    return None


def _config_quantization_bits(config: Mapping[str, Any]) -> Any:
    source = _model_config_source(config)
    quantization = (
        source.get("quantization")
        or source.get("quantization_config")
        or config.get("quantization")
        or config.get("quantization_config")
    )
    return quantization.get("bits") if isinstance(quantization, Mapping) else None


def _normalized_quantization_layout(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(QUANTIZATION_LAYOUT_FIELDS):
        raise ValueError(
            "Model quantization_layout must contain the exact aptus.facts.v3 "
            "layout fields."
        )
    default_bits = value.get("default_bits")
    default_group_size = value.get("default_group_size")
    if (
        not _positive_int(default_bits)
        or default_bits > 16
        or not _positive_int(default_group_size)
    ):
        raise ValueError(
            "Model quantization layout defaults require bits from 1 through 16 "
            "and a positive group_size."
        )
    raw_overrides = value.get("module_overrides")
    if not isinstance(raw_overrides, (list, tuple)):
        raise ValueError("Model quantization module_overrides must be a list.")
    overrides: list[dict[str, Any]] = []
    for item in raw_overrides:
        if not isinstance(item, Mapping) or set(item) != set(
            QUANTIZATION_OVERRIDE_FIELDS
        ):
            raise ValueError(
                "Each model quantization override must contain only module_path, "
                "bits, and group_size."
            )
        module_path = item.get("module_path")
        bits = item.get("bits")
        group_size = item.get("group_size")
        if (
            not isinstance(module_path, str)
            or not module_path
            or len(module_path) > 256
            or any(
                not part
                or any(
                    character
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
                    for character in part
                )
                for part in module_path.split(".")
            )
        ):
            raise ValueError(
                "Model quantization override module_path must be a dotted module identifier."
            )
        if not _positive_int(bits) or bits > 16 or not _positive_int(group_size):
            raise ValueError(
                "Model quantization override bits or group_size is invalid."
            )
        overrides.append(
            {
                "module_path": module_path,
                "bits": bits,
                "group_size": group_size,
            }
        )
    paths = [item["module_path"] for item in overrides]
    if paths != sorted(set(paths)):
        raise ValueError(
            "Model quantization module_overrides must be sorted and unique."
        )
    return {
        "default_bits": default_bits,
        "default_group_size": default_group_size,
        "module_overrides": overrides,
    }


def _reviewed_qwen3_moe_quantization_layout(layers: int) -> dict[str, Any]:
    if not _positive_int(layers):
        raise ValueError("Reviewed Qwen3 MoE quantization requires positive layers.")
    return {
        "default_bits": 4,
        "default_group_size": 64,
        "module_overrides": [
            {
                "module_path": f"model.layers.{index}.mlp.gate",
                "bits": 8,
                "group_size": 64,
            }
            for index in sorted(range(layers), key=lambda value: str(value))
        ],
    }


def _reviewed_gemma4_moe_quantization_layout(layers: int) -> dict[str, Any]:
    if not _positive_int(layers):
        raise ValueError("Reviewed Gemma 4 MoE quantization requires positive layers.")
    return {
        "default_bits": 4,
        "default_group_size": 64,
        "module_overrides": [
            {
                "module_path": f"model.layers.{index}.router.proj",
                "bits": 8,
                "group_size": 64,
            }
            for index in sorted(range(layers), key=lambda value: str(value))
        ],
    }


def _strip_language_model_prefix(module_path: str) -> str:
    prefix = "language_model."
    if module_path.startswith(prefix):
        return module_path[len(prefix) :]
    return module_path


def _is_reviewed_moe_identity(
    model: Mapping[str, Any], *, model_type: Any, architecture: Any
) -> str | None:
    family = model.get("family")
    if (
        family == QWEN3_MOE_FAMILY
        and model_type == QWEN3_MOE_MODEL_TYPE
        and architecture == QWEN3_MOE_ARCHITECTURE
    ):
        return QWEN3_MOE_FAMILY
    if (
        family == GEMMA4_MOE_FAMILY
        and model_type == GEMMA4_MOE_MODEL_TYPE
        and architecture == GEMMA4_MOE_ARCHITECTURE
    ):
        return GEMMA4_MOE_FAMILY
    return None


def _canonical_config_quantization_layout(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    layout = {
        "default_bits": value.get("bits"),
        "default_group_size": value.get("group_size"),
        "module_overrides": [
            {
                "module_path": _strip_language_model_prefix(key),
                "bits": item.get("bits") if isinstance(item, Mapping) else None,
                "group_size": (
                    item.get("group_size") if isinstance(item, Mapping) else None
                ),
            }
            for key, item in value.items()
            if key not in {"bits", "group_size"} and isinstance(item, Mapping)
        ],
    }
    layout["module_overrides"].sort(key=lambda item: item["module_path"])
    try:
        normalized = _normalized_quantization_layout(layout)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Pinned model quantization layout contains unsupported fields or values."
        ) from error
    assert normalized is not None
    for key, item in value.items():
        if key in {"bits", "group_size"} or not isinstance(item, Mapping):
            continue
        if set(item) != {"bits", "group_size"}:
            raise ValueError(
                "Pinned model quantization layout contains unsupported override fields."
            )
    return normalized


def _config_quantization_layout(config: Mapping[str, Any]) -> dict[str, Any] | None:
    source = _model_config_source(config)
    candidates: list[Mapping[str, Any]] = []
    for container in (source, config):
        for name in ("quantization", "quantization_config"):
            value = container.get(name)
            if isinstance(value, Mapping) and all(
                value is not item for item in candidates
            ):
                candidates.append(value)
    if not candidates:
        return None
    layouts = [_canonical_config_quantization_layout(item) for item in candidates]
    if any(item != layouts[0] for item in layouts[1:]):
        raise ValueError(
            "Pinned model quantization and quantization_config layouts disagree."
        )
    return layouts[0]


def _normalized_moe(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(MOE_TOPOLOGY_FIELDS):
        raise ValueError(
            "Model moe must contain the exact aptus.facts.v3 topology fields."
        )
    positive_names = (
        "expert_count",
        "experts_per_token",
        "expert_intermediate_size",
        "decoder_sparse_step",
    )
    if any(not _positive_int(value.get(name)) for name in positive_names):
        raise ValueError("Model MoE topology integer facts must be positive.")
    if value["experts_per_token"] > value["expert_count"]:
        raise ValueError("Model experts_per_token cannot exceed expert_count.")
    shared = value.get("shared_expert_intermediate_size")
    if shared is not None and not _positive_int(shared):
        raise ValueError(
            "Model shared_expert_intermediate_size must be positive when supplied."
        )
    mlp_only = value.get("mlp_only_layers")
    if (
        not isinstance(mlp_only, (list, tuple))
        or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in mlp_only
        )
        or list(mlp_only) != sorted(set(mlp_only))
    ):
        raise ValueError("Model mlp_only_layers must be a sorted unique integer list.")
    return {
        "expert_count": value["expert_count"],
        "experts_per_token": value["experts_per_token"],
        "expert_intermediate_size": value["expert_intermediate_size"],
        "decoder_sparse_step": value["decoder_sparse_step"],
        "mlp_only_layers": list(mlp_only),
        "shared_expert_intermediate_size": shared,
    }


def _derived_model_topology(
    model: Mapping[str, Any], moe: Mapping[str, Any] | None
) -> tuple[int, int]:
    parameters = model.get("parameters")
    layers = model.get("layers")
    hidden_size = model.get("hidden_size")
    if not all(_positive_int(value) for value in (parameters, layers, hidden_size)):
        raise ValueError("Model structural facts are incomplete.")
    assert isinstance(parameters, int)
    assert isinstance(layers, int)
    assert isinstance(hidden_size, int)
    if moe is None:
        return 0, parameters
    mlp_only = set(moe["mlp_only_layers"])
    if any(index >= layers for index in mlp_only):
        raise ValueError("Model mlp_only_layers references a missing layer.")
    sparse_layers = sum(
        1
        for index in range(layers)
        if (index + 1) % moe["decoder_sparse_step"] == 0 and index not in mlp_only
    )
    if sparse_layers <= 0:
        raise ValueError("Model MoE topology must contain at least one sparse layer.")
    inactive = (
        sparse_layers
        * (moe["expert_count"] - moe["experts_per_token"])
        * 3
        * hidden_size
        * moe["expert_intermediate_size"]
    )
    active = parameters - inactive
    if active <= 0 or active > parameters:
        raise ValueError(
            "Model derived active_parameters must be positive and no greater than total parameters."
        )
    return sparse_layers, active


def expected_model_architecture_contract(
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic model architecture contract carried by runtime proof."""

    if not isinstance(model, Mapping):
        raise ValueError("Model architecture contract requires a model object.")
    architecture = model.get("architecture")
    model_type = model.get("model_type")
    if not isinstance(architecture, str) or not architecture.strip():
        raise ValueError("Model architecture is required.")
    if model_type is not None and (
        not isinstance(model_type, str) or not model_type.strip()
    ):
        raise ValueError("Model model_type must be non-empty when supplied.")
    for name in ("parameters", "hidden_size", "layers", "context_length"):
        if not _positive_int(model.get(name)):
            raise ValueError(f"Model {name} must be positive.")
    intermediate_size = model.get("intermediate_size")
    if intermediate_size is not None and not _positive_int(intermediate_size):
        raise ValueError("Model intermediate_size must be positive when supplied.")
    quantization_bits = model.get("quantization_bits")
    if quantization_bits is not None and (
        not _positive_int(quantization_bits) or quantization_bits > 16
    ):
        raise ValueError("Model quantization_bits must be between 1 and 16.")
    quantization_layout = _normalized_quantization_layout(
        model.get("quantization_layout")
    )
    if quantization_layout is not None and (
        quantization_bits != quantization_layout["default_bits"]
    ):
        raise ValueError(
            "Model quantization_bits must equal quantization_layout default_bits."
        )
    moe = _normalized_moe(model.get("moe"))
    sparse_layers, active_parameters = _derived_model_topology(model, moe)
    if model.get("sparse_layer_count") != sparse_layers:
        raise ValueError("Model sparse_layer_count does not match its topology.")
    if model.get("active_parameters") != active_parameters:
        raise ValueError("Model active_parameters does not match its topology.")
    if moe is not None:
        moe_family = _is_reviewed_moe_identity(
            model, model_type=model_type, architecture=architecture
        )
        if moe_family is None:
            raise ValueError(
                "Model MoE topology requires an exact reviewed MoE identity."
            )
        if moe["shared_expert_intermediate_size"] is not None:
            raise ValueError(
                "The reviewed MoE runtime does not support a shared expert."
            )
        if quantization_bits != 4:
            raise ValueError(
                "The reviewed MoE runtime requires explicit four-bit metadata."
            )
        expected_layout = (
            _reviewed_qwen3_moe_quantization_layout(model["layers"])
            if moe_family == QWEN3_MOE_FAMILY
            else _reviewed_gemma4_moe_quantization_layout(model["layers"])
        )
        if quantization_layout != expected_layout:
            raise ValueError(
                "The reviewed MoE runtime requires the exact four-bit "
                "group-64 layout with one eight-bit group-64 router override "
                "for every layer."
            )
    quantization_layout_sha256 = (
        hashlib.sha256(
            json.dumps(
                quantization_layout, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if quantization_layout is not None
        else None
    )
    payload = {
        "schema_version": "aptus.model-architecture-contract.v1",
        "model_type": model_type,
        "architecture": architecture,
        "parameters": model["parameters"],
        "hidden_size": model["hidden_size"],
        "intermediate_size": intermediate_size,
        "layers": model["layers"],
        "context_length": model["context_length"],
        "quantization_bits": quantization_bits,
        "quantization_layout": quantization_layout,
        "quantization_layout_sha256": quantization_layout_sha256,
        "moe": moe,
        "sparse_layer_count": sparse_layers,
        "active_parameters": active_parameters,
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def validate_model_config_against_plan(
    model: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Match a pinned provider config to the plan and return its proof payload."""

    expected = expected_model_architecture_contract(model)
    if not isinstance(config, Mapping):
        raise ValueError("Pinned model config must be an object.")
    source = _model_config_source(config)
    expected_model_type = expected["model_type"]
    observed_model_type = source.get("model_type") or config.get("model_type")
    if expected_model_type is not None and observed_model_type != expected_model_type:
        raise ValueError("Pinned model_type does not match the plan.")
    raw_architectures = config.get("architectures") or source.get("architectures")
    observed_architecture = (
        raw_architectures[0]
        if isinstance(raw_architectures, list)
        and raw_architectures
        and isinstance(raw_architectures[0], str)
        else observed_model_type
    )
    if (
        expected["architecture"] != "causal-lm"
        and observed_architecture != expected["architecture"]
    ):
        raise ValueError("Pinned model architecture does not match the plan.")
    structural_names = {
        "hidden_size": ("hidden_size", "d_model", "n_embd"),
        "intermediate_size": ("intermediate_size", "ffn_dim", "n_inner"),
        "layers": ("num_hidden_layers", "n_layer", "num_layers"),
        "context_length": (
            "max_position_embeddings",
            "n_positions",
            "seq_length",
        ),
    }
    for planned_name, config_names in structural_names.items():
        planned = expected[planned_name]
        observed = _config_first(source, *config_names)
        if planned is not None and (
            (expected["moe"] is not None and observed != planned)
            or (observed is not None and observed != planned)
        ):
            raise ValueError(f"Pinned model {planned_name} does not match the plan.")
    observed_bits = _config_quantization_bits(config)
    if (
        expected["quantization_bits"] is not None
        and observed_bits != expected["quantization_bits"]
    ):
        raise ValueError("Pinned model quantization bits do not match the plan.")
    expected_layout = expected["quantization_layout"]
    if expected_layout is not None:
        observed_layout = _config_quantization_layout(config)
        if observed_layout != expected_layout:
            raise ValueError(
                "Pinned model quantization layout does not match the plan."
            )
    moe_config_names = (
        "num_experts",
        "num_experts_per_tok",
        "top_k_experts",
        "moe_intermediate_size",
        "decoder_sparse_step",
        "mlp_only_layers",
        "shared_expert_intermediate_size",
    )
    expected_moe = expected["moe"]
    if expected_moe is None and any(
        _config_declares_moe_topology_field(source, name) for name in moe_config_names
    ):
        raise ValueError("Pinned model unexpectedly declares MoE topology.")
    if expected_moe is not None:
        experts_per_token = source.get("num_experts_per_tok")
        if experts_per_token is None:
            experts_per_token = source.get("top_k_experts")
        decoder_sparse_step = source.get("decoder_sparse_step")
        mlp_only_layers = source.get("mlp_only_layers")
        gemma_style = (
            source.get("enable_moe_block") is True
            or source.get("top_k_experts") is not None
        )
        if gemma_style:
            if decoder_sparse_step is None:
                decoder_sparse_step = 1
            if mlp_only_layers is None:
                mlp_only_layers = []
        observed_moe = {
            "expert_count": source.get("num_experts"),
            "experts_per_token": experts_per_token,
            "expert_intermediate_size": source.get("moe_intermediate_size"),
            "decoder_sparse_step": decoder_sparse_step,
            "mlp_only_layers": mlp_only_layers,
            "shared_expert_intermediate_size": source.get(
                "shared_expert_intermediate_size"
            ),
        }
        if observed_moe != expected_moe:
            raise ValueError("Pinned model MoE topology does not match the plan.")
    return expected


def _select(value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    source = _mapping(value)
    return {key: source.get(key) for key in keys}


def _normalized_model(value: Any) -> dict[str, Any]:
    model = _select(
        value,
        (
            "model_id",
            "revision",
            "family",
            "parameters",
            "hidden_size",
            "intermediate_size",
            "layers",
            "context_length",
            "architecture",
            "model_type",
            "quantization_bits",
            "quantization_layout",
            "sparse_layer_count",
            "active_parameters",
            "tokenizer_id",
            "license_name",
            "training_allowed",
        ),
    )
    moe = _mapping(value).get("moe")
    quantization_layout = _mapping(value).get("quantization_layout")
    model["quantization_layout"] = (
        _normalized_quantization_layout(quantization_layout)
        if quantization_layout is not None
        else None
    )
    model["moe"] = (
        _select(moe, MOE_TOPOLOGY_FIELDS) if isinstance(moe, Mapping) else None
    )
    return model


def _normalized_dataset(value: Any) -> dict[str, Any]:
    dataset = _select(
        value,
        (
            "source_sha256",
            "source_format",
            "schema_name",
            "example_count",
            "total_estimated_tokens",
            "sequence_p50",
            "sequence_p95",
            "sequence_max",
            "measurement",
            "sampled_examples",
            "sample_indices",
            "duplicate_count",
            "empty_count",
            "truncation_count",
            "truncation_rate",
            "source_size_bytes",
            "canonical_size_bytes",
            "max_canonical_row_bytes",
        ),
    )
    dataset["schema_counts"] = dict(
        sorted(_mapping(_mapping(value).get("schema_counts")).items())
    )
    return dataset


def _normalized_hardware(value: Any) -> dict[str, Any]:
    hardware = _mapping(value)
    devices = hardware.get("devices")
    normalized_devices = (
        [
            _select(
                item,
                (
                    "name",
                    "backend",
                    "total_vram_bytes",
                    "free_vram_bytes",
                    "supports_bf16",
                    "supports_4bit",
                    "supports_8bit",
                    "compute_capability",
                    "driver_version",
                ),
            )
            for item in devices
        ]
        if isinstance(devices, (list, tuple))
        else []
    )
    return {
        "devices": normalized_devices,
        **_select(
            hardware,
            (
                "host_ram_bytes",
                "host_ram_free_bytes",
                "reserve_per_device_bytes",
                "disk_free_bytes",
                "cuda_version",
                "interconnect",
            ),
        ),
    }


def _normalized_target(value: Any) -> dict[str, Any]:
    return _select(
        value,
        (
            "objective",
            "sequence_length",
            "effective_batch_size",
            "max_epochs",
            "method_preference",
            "task",
            "evaluation_fraction",
            "packing",
            "checkpoint_steps",
            "max_wall_time_minutes",
            "training_runtime",
            "optimizer_steps",
            "split_seed",
            "training_seed",
            "data_order_seed",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "unreviewed_runtime_confirmed",
        ),
    )


def _mlx_adapter_parameter_count(
    model: Mapping[str, Any], *, rank: int, target_modules: list[str] | tuple[str, ...]
) -> int:
    """Return the deterministic adapter cardinality used by the MLX estimator."""

    hidden_size = model.get("hidden_size")
    layers = model.get("layers")
    if not _positive_int(hidden_size) or not _positive_int(layers):
        raise ValueError("MLX memory recomputation requires positive model dimensions.")
    if not _positive_int(rank):
        raise ValueError("MLX memory recomputation requires a positive adapter rank.")
    intermediate_size = model.get("intermediate_size")
    if intermediate_size is None and any(
        module in {"gate_proj", "up_proj", "down_proj"} for module in target_modules
    ):
        raise ValueError(
            "MLX memory recomputation refuses invented 4 × hidden_size for MLP adapters."
        )
    if intermediate_size is None:
        intermediate_size = 0
    if not _positive_int(intermediate_size) and any(
        module in {"gate_proj", "up_proj", "down_proj"} for module in target_modules
    ):
        raise ValueError(
            "MLX memory recomputation requires a positive intermediate dimension."
        )
    moe = model.get("moe")
    if moe is not None and any(
        module in ("gate_proj", "up_proj", "down_proj") for module in target_modules
    ):
        raise ValueError(
            "MLX MoE memory recomputation refuses topology-free expert adapters."
        )
    per_layer = 0
    for module in target_modules:
        if module in ("gate_proj", "up_proj", "down_proj"):
            per_layer += hidden_size + intermediate_size
        elif module in ("q_proj", "k_proj", "v_proj", "o_proj"):
            per_layer += hidden_size * 2
        else:
            raise ValueError(
                f"MLX memory recomputation does not recognize target module {module!r}."
            )
    return layers * rank * per_layer


def mlx_quantized_storage_bytes_for_contract(
    model: Mapping[str, Any], *, logical_parameters: int | None = None
) -> tuple[int, int]:
    """Price the bound MLX affine layout, including reviewed router overrides."""

    parameters = (
        logical_parameters
        if logical_parameters is not None
        else model.get("parameters")
    )
    if not _positive_int(parameters):
        raise ValueError(
            "MLX memory recomputation requires a positive parameter count."
        )
    layout = model.get("quantization_layout")
    if layout is None:
        # Dense aptus.facts.v3 records do not require an exact layout binding.
        # Preserve their analytical prior; runtime still needs real 4-bit metadata.
        return round(parameters * 0.5), round(parameters * 0.0625)
    if not isinstance(layout, Mapping):
        raise ValueError("MLX quantization layout must be an object.")
    default_bits = layout.get("default_bits")
    default_group_size = layout.get("default_group_size")
    overrides = layout.get("module_overrides")
    if (
        not _positive_int(default_bits)
        or not _positive_int(default_group_size)
        or not isinstance(overrides, list)
    ):
        raise ValueError("MLX quantization layout is incomplete.")
    if not overrides:
        return (
            round(parameters * default_bits / 8),
            round(parameters * 4 / default_group_size),
        )

    moe = model.get("moe")
    hidden_size = model.get("hidden_size")
    if not isinstance(moe, Mapping) or not _positive_int(hidden_size):
        raise ValueError(
            "MLX quantization overrides require a bound MoE topology and hidden size."
        )
    expert_count = moe.get("expert_count")
    if not _positive_int(expert_count):
        raise ValueError("MLX quantization overrides require a positive expert count.")

    overridden_parameters = 0
    weighted_storage_bits = 0
    weighted_metadata_bytes = 0.0
    seen_paths: set[str] = set()
    for override in overrides:
        if not isinstance(override, Mapping):
            raise ValueError("MLX quantization overrides must be objects.")
        module_path = override.get("module_path")
        bits = override.get("bits")
        group_size = override.get("group_size")
        if (
            not isinstance(module_path, str)
            or not module_path
            or module_path in seen_paths
            or not module_path.startswith("model.layers.")
            or not (
                module_path.endswith(".mlp.gate")
                or module_path.endswith(".router.proj")
            )
            or not _positive_int(bits)
            or not _positive_int(group_size)
        ):
            raise ValueError("MLX quantization override is not a unique router gate.")
        seen_paths.add(module_path)
        parameter_count = hidden_size * expert_count
        overridden_parameters += parameter_count
        weighted_storage_bits += parameter_count * bits
        # MLX affine quantization stores one half-precision scale and bias per group.
        weighted_metadata_bytes += parameter_count * 4 / group_size
    if overridden_parameters > parameters:
        raise ValueError("MLX quantization overrides exceed the model parameter count.")
    default_parameters = parameters - overridden_parameters
    storage_bytes = round(
        (default_parameters * default_bits + weighted_storage_bits) / 8
    )
    metadata_bytes = round(
        default_parameters * 4 / default_group_size + weighted_metadata_bytes
    )
    return storage_bytes, metadata_bytes


def mlx_memory_breakdown_for_contract(
    *,
    model: Mapping[str, Any],
    target: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the portable MLX memory contract from normalized plan facts."""

    method = candidate.get("method")
    if method not in ("lora", "qlora"):
        raise ValueError("The MLX estimator supports LoRA and QLoRA only.")
    parameters = model.get("parameters")
    hidden_size = model.get("hidden_size")
    layers = model.get("layers")
    sequence_length = target.get("sequence_length")
    micro_batch_size = candidate.get("micro_batch_size")
    rank = candidate.get("rank")
    target_modules = candidate.get("target_modules")
    if not all(
        _positive_int(value)
        for value in (
            parameters,
            hidden_size,
            layers,
            sequence_length,
            micro_batch_size,
            rank,
        )
    ) or not isinstance(target_modules, list):
        raise ValueError("MLX memory recomputation requires complete positive facts.")

    if method == "lora":
        base_weights = round(parameters * 2.0)
        quantization_metadata = 0
    else:
        base_weights, quantization_metadata = mlx_quantized_storage_bytes_for_contract(
            model
        )
    trainable = _mlx_adapter_parameter_count(
        model, rank=rank, target_modules=target_modules
    )
    adapter_weights = round(trainable * 4)
    gradients = round(trainable * 4)
    optimizer = round(trainable * 8)
    dense_activations = round(
        micro_batch_size * sequence_length * hidden_size * layers * 2 * 3.0
    )
    routed_expert_activations = 0
    moe = model.get("moe")
    if isinstance(moe, Mapping):
        sparse_layer_count = model.get("sparse_layer_count")
        experts_per_token = moe.get("experts_per_token")
        expert_intermediate_size = moe.get("expert_intermediate_size")
        if not all(
            _positive_int(value)
            for value in (
                sparse_layer_count,
                experts_per_token,
                expert_intermediate_size,
            )
        ):
            raise ValueError(
                "MLX MoE memory recomputation requires complete derived topology."
            )
        routed_expert_activations = round(
            micro_batch_size
            * sequence_length
            * sparse_layer_count
            * experts_per_token
            * expert_intermediate_size
            * 2
            * 3.0
        )
    activations = dense_activations + routed_expert_activations
    resident_bytes = base_weights + quantization_metadata
    workspace = max(round(0.75 * 1024**3), round(resident_bytes * 0.04))
    temporary = max(round(0.75 * 1024**3), round(resident_bytes * 0.08))
    load_transient = round(resident_bytes * 0.30)
    point_components = {
        "base_weights_bytes": base_weights,
        "quantization_metadata_bytes": quantization_metadata,
        "adapter_weights_bytes": adapter_weights,
        "adapter_gradients_bytes": gradients,
        "optimizer_states_bytes": optimizer,
        "activations_bytes": activations,
        "communication_bytes": 0,
        "workspace_bytes": workspace,
        "temporary_overhead_bytes": temporary,
        "load_transient_bytes": load_transient,
    }
    allocator = round(sum(point_components.values()) * 0.15)
    point_components["allocator_bytes"] = allocator
    point = sum(point_components.values())
    safety = round(point * 0.25)
    component_upper_bounds = {
        **point_components,
        "activations_bytes": round(activations * 1.60),
        "workspace_bytes": round(workspace * 1.75),
        "temporary_overhead_bytes": round(temporary * 1.75),
        "allocator_bytes": round(allocator * 1.75),
        "load_transient_bytes": round(load_transient * 1.50),
        "uncertainty_bytes": safety,
    }
    return {
        **point_components,
        "safety_margin_bytes": safety,
        "point_estimate_bytes": point,
        "estimated_peak_bytes": point,
        "upper_estimate_bytes": sum(component_upper_bounds.values()),
        "uncertainty_bytes": safety,
        "formula_version": MLX_FORMULA_VERSION,
        "component_upper_bounds": component_upper_bounds,
    }


def _normalized_memory(value: Any) -> dict[str, Any]:
    memory = _select(
        value,
        (
            "base_weights_bytes",
            "quantization_metadata_bytes",
            "adapter_weights_bytes",
            "adapter_gradients_bytes",
            "optimizer_states_bytes",
            "activations_bytes",
            "temporary_overhead_bytes",
            "safety_margin_bytes",
            "communication_bytes",
            "workspace_bytes",
            "allocator_bytes",
            "load_transient_bytes",
            "point_estimate_bytes",
            "estimated_peak_bytes",
            "upper_estimate_bytes",
            "uncertainty_bytes",
            "formula_version",
        ),
    )
    memory["component_upper_bounds"] = dict(
        sorted(_mapping(_mapping(value).get("component_upper_bounds")).items())
    )
    return memory


def candidate_id_for_payload(
    candidate: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    dataset: Mapping[str, Any],
    hardware: Mapping[str, Any],
    target: Mapping[str, Any],
) -> str:
    """Derive the portable content ID for an executable candidate contract."""

    target_modules = candidate.get("target_modules")
    runtime_contract = candidate.get("runtime_contract")
    identity = {
        "strategy": {
            **_select(
                candidate,
                (
                    "method",
                    "distribution",
                    "precision",
                    "quantization",
                    "micro_batch_size",
                    "gradient_accumulation_steps",
                    "effective_batch_size",
                    "world_size",
                    "device_indices",
                    "rank",
                    "alpha",
                    "learning_rate",
                    "user_reserve_bytes",
                    "required_host_ram_bytes",
                    "required_disk_bytes",
                    "checkpoint_retention_bytes",
                    "final_export_bytes",
                    "status",
                    "feasible",
                ),
            ),
            "target_modules": sorted(target_modules)
            if isinstance(target_modules, (list, tuple))
            else [],
            "memory": _normalized_memory(candidate.get("memory")),
            **(
                {
                    "runtime_contract": _select(
                        runtime_contract,
                        (
                            "schema_version",
                            "compute_backend",
                            "training_runtime",
                            "compiler_id",
                            "estimator_id",
                            "evidence_requirement",
                            "export_kind",
                        ),
                    )
                }
                if isinstance(runtime_contract, Mapping)
                else {}
            ),
            "model_policy_decision_id": candidate.get("model_policy_decision_id"),
            "policy_binding": candidate.get("policy_binding"),
        },
        "facts": {
            "model": _normalized_model(model),
            "dataset": _normalized_dataset(dataset),
            "hardware": _normalized_hardware(hardware),
            "target": _normalized_target(target),
        },
    }
    return _content_id("cand_", identity)


def plan_id_for_payload(plan: Mapping[str, Any]) -> str:
    """Derive the plan ID from normalized facts, candidates, and recommendation."""

    candidates = plan.get("candidates")
    candidate_ids = (
        [
            item.get("candidate_id") if isinstance(item, Mapping) else None
            for item in candidates
        ]
        if isinstance(candidates, (list, tuple))
        else []
    )
    recommended = _mapping(plan.get("recommended"))
    identity = {
        "schema_version": plan.get("schema_version"),
        "formula_version": plan.get("formula_version"),
        "training_policy_version": plan.get("training_policy_version"),
        "facts": {
            "model": _normalized_model(plan.get("model")),
            "dataset": _normalized_dataset(plan.get("dataset")),
            "hardware": _normalized_hardware(plan.get("hardware")),
            "target": _normalized_target(plan.get("target")),
        },
        "candidate_ids": candidate_ids,
        "recommended_candidate_id": recommended.get("candidate_id"),
        "model_policy_decision": _semantic_policy_decision(
            plan.get("model_policy_decision")
        ),
        "model_policy_decision_source": plan.get("model_policy_decision_source"),
        "model_policy_snapshot_sha256": plan.get("model_policy_snapshot_sha256"),
        "inspection_receipt": _semantic_inspection_receipt(
            plan.get("inspection_receipt")
        ),
        "evidence_records": plan.get("evidence_records"),
    }
    return _content_id("plan_", identity)


def _validate_plan_payload_impl(
    plan_value: Any,
    *,
    root: Path | None = None,
    verify_dataset: bool = True,
    expected_policy_decision_override: Mapping[str, Any] | None = None,
    policy_snapshot: Mapping[str, Any] | None = None,
    enforce_current_policy: bool = True,
    allow_unavailable_policy_provenance: bool = False,
) -> tuple[str, ...]:
    errors: list[str] = []
    if not isinstance(plan_value, dict) or not plan_value:
        return ("Plan must be a non-empty object.",)
    plan = plan_value
    if _contains_nonfinite(plan):
        errors.append("Plan numbers must be finite JSON values.")
    if plan.get("schema_version") == "aptus.training-plan.v2" and any(
        key in _mapping(plan.get("model"))
        for key in (
            "model_type",
            "quantization_bits",
            "quantization_layout",
            "moe",
            "sparse_layer_count",
            "active_parameters",
        )
    ):
        errors.append("A v2 plan cannot contain v3 model architecture or MoE fields.")
    if plan.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"Plan schema_version must be {SCHEMA_VERSION}; replan_required.")
    if plan.get("formula_version") != FORMULA_VERSION:
        errors.append(f"Plan formula_version must be {FORMULA_VERSION}.")
    if plan.get("training_policy_version") != TRAINING_POLICY_VERSION:
        errors.append(
            f"Plan training_policy_version must be {TRAINING_POLICY_VERSION}."
        )
    for key in (
        "model",
        "dataset",
        "hardware",
        "target",
        "recommended",
        "candidates",
        "evidence_records",
        "model_policy_decision",
        "model_policy_decision_source",
        "inspection_receipt",
        "model_policy_snapshot_sha256",
    ):
        if key not in plan:
            errors.append(f"Plan requires {key}.")

    model = plan.get("model") if isinstance(plan.get("model"), dict) else {}
    dataset = plan.get("dataset") if isinstance(plan.get("dataset"), dict) else {}
    hardware = plan.get("hardware") if isinstance(plan.get("hardware"), dict) else {}
    target = plan.get("target") if isinstance(plan.get("target"), dict) else {}
    for key in ("model_id", "revision", "family", "license_name"):
        if not isinstance(model.get(key), str) or not model.get(key, "").strip():
            errors.append(f"Model {key} is required.")
    if (
        isinstance(model.get("family"), str)
        and model["family"] != model["family"].lower()
    ):
        errors.append("Model family must use its canonical lowercase identity.")
    revision = model.get("revision")
    if (
        not isinstance(revision, str)
        or not (40 <= len(revision) <= 64)
        or any(c not in "0123456789abcdefABCDEF" for c in revision)
    ):
        errors.append("Model revision must be an immutable hexadecimal commit ID.")
    if model.get("training_allowed") is not True:
        errors.append("Model training permission must be explicitly true.")
    for key in ("parameters", "hidden_size", "layers", "context_length"):
        if not _positive_int(model.get(key)):
            errors.append(f"Model {key} must be positive.")
    if model.get("intermediate_size") is not None and not _positive_int(
        model.get("intermediate_size")
    ):
        errors.append("Model intermediate_size must be positive when supplied.")
    architecture = model.get("architecture")
    if not isinstance(architecture, str) or not architecture.strip():
        errors.append("Model architecture is required.")
    model_type = model.get("model_type")
    if model_type is not None and (
        not isinstance(model_type, str) or not model_type.strip()
    ):
        errors.append("Model model_type must be non-empty when supplied.")
    moe_identity = bool(
        model.get("family") == QWEN3_MOE_FAMILY
        or model.get("family") == GEMMA4_MOE_FAMILY
        or model_type == QWEN3_MOE_MODEL_TYPE
        or architecture == QWEN3_MOE_ARCHITECTURE
        or model.get("moe") is not None
    )
    if moe_identity and model.get("moe") is None:
        errors.append("MoE plans require complete expert topology facts.")
    try:
        expected_model_architecture_contract(model)
    except ValueError as error:
        errors.append(str(error))

    snapshot: Mapping[str, Any] | None = None
    current_policy_decision: Mapping[str, Any] | None = None
    try:
        snapshot = _policy_snapshot_for_validation(
            root=root,
            policy_snapshot=policy_snapshot,
        )
        current_policy_decision = _current_model_policy_decision(
            model,
            snapshot,
            confirm_unreviewed_runtime=bool(
                (plan.get("target") or {}).get("unreviewed_runtime_confirmed")
            )
            if isinstance(plan.get("target"), Mapping)
            else False,
        )
    except (TypeError, ValueError) as error:
        errors.append(
            "Model compatibility decision could not be recomputed from bound facts: "
            f"{error}"
        )
    snapshot_digest = plan.get("model_policy_snapshot_sha256")
    if (
        not isinstance(snapshot_digest, str)
        or len(snapshot_digest) != 64
        or any(character not in "0123456789abcdef" for character in snapshot_digest)
    ):
        errors.append(
            "Plan model_policy_snapshot_sha256 must be a lowercase SHA-256 digest."
        )
    elif enforce_current_policy and snapshot is not None:
        try:
            if snapshot_digest != _policy_snapshot_sha256(snapshot):
                errors.append(
                    "Plan model policy snapshot digest is stale or tampered; replan_required."
                )
        except ValueError as error:
            errors.append(f"Model policy snapshot is malformed: {error}")
    expected_policy_decision = (
        expected_policy_decision_override
        if expected_policy_decision_override is not None
        else current_policy_decision
    )
    policy_decision_value = plan.get("model_policy_decision")
    if not isinstance(policy_decision_value, Mapping):
        errors.append("Plan model_policy_decision must be an object.")
        policy_decision: Mapping[str, Any] = {}
    else:
        policy_decision = policy_decision_value
        if set(policy_decision) != {
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
        }:
            errors.append(
                "Plan model_policy_decision must contain the exact v2 fields."
            )
        if not _valid_content_id(policy_decision.get("decision_id"), prefix="compat_"):
            errors.append("Plan model policy decision ID is invalid.")
        elif policy_decision.get("decision_id") != (
            "compat_"
            + _sha256_json(_policy_decision_identity_payload(policy_decision))[:20]
        ):
            errors.append(
                "Plan model policy decision ID does not match its semantic content."
            )
        if not _valid_sha256(policy_decision.get("subject_facts_sha256")):
            errors.append("Plan model policy subject digest is invalid.")
        reason = policy_decision.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or reason != reason.strip()
        ):
            errors.append("Plan model policy reason must be unpadded non-empty text.")
        policy_comparison_target = (
            current_policy_decision
            if enforce_current_policy
            else expected_policy_decision
        )
        if policy_comparison_target is not None and _semantic_policy_decision(
            policy_decision
        ) != _semantic_policy_decision(policy_comparison_target):
            errors.append(
                "Plan model compatibility decision is stale, tampered, or differs "
                "from the current registered policy."
            )

    policy_source = plan.get("model_policy_decision_source")
    if not _known_text(policy_source, POLICY_DECISION_SOURCES):
        errors.append(
            "Plan model_policy_decision_source must be provider-inspection or "
            "user-attested."
        )
    inspection_receipt = (
        _validate_inspection_receipt(
            plan.get("inspection_receipt"),
            model=model,
            expected_decision=expected_policy_decision,
            policy_snapshot=snapshot,
            allow_unavailable_policy_provenance=allow_unavailable_policy_provenance,
            errors=errors,
        )
        if expected_policy_decision is not None
        else None
    )
    if policy_source == "provider-inspection" and inspection_receipt is None:
        errors.append("Provider-inspection plans require a valid inspection receipt.")
    if policy_source == "user-attested" and plan.get("inspection_receipt") is not None:
        errors.append("User-attested plans cannot carry an inspection receipt.")
    if policy_source == "user-attested" and plan.get("inspection_receipt") is None:
        _validate_user_attested_model_provenance(model=model, errors=errors)

    if not _known_text(
        dataset.get("schema_name"),
        {"text", "prompt-completion", "instruction-output", "messages", "mixed"},
    ):
        errors.append("Dataset schema is unsupported.")
    if (
        not isinstance(dataset.get("source_path"), str)
        or not dataset.get("source_path", "").strip()
    ):
        errors.append("Dataset source_path is required.")
    if not _known_text(dataset.get("source_format"), {"jsonl", "json", "csv", "txt"}):
        errors.append("Dataset source_format is unsupported.")
    for key in (
        "example_count",
        "total_estimated_tokens",
        "sequence_p50",
        "sequence_p95",
        "sequence_max",
        "sampled_examples",
    ):
        if not _positive_int(dataset.get(key)):
            errors.append(f"Dataset {key} must be positive.")
    if all(
        _positive_int(dataset.get(key))
        for key in ("sequence_p50", "sequence_p95", "sequence_max")
    ) and not (
        dataset["sequence_p50"] <= dataset["sequence_p95"] <= dataset["sequence_max"]
    ):
        errors.append("Dataset sequence percentiles must be ordered.")
    digest = dataset.get("source_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in digest)
    ):
        errors.append("Dataset source_sha256 is invalid.")
    for key in (
        "source_size_bytes",
        "canonical_size_bytes",
        "max_canonical_row_bytes",
    ):
        if not _positive_int(dataset.get(key)):
            errors.append(f"Dataset {key} must be positive.")
    if verify_dataset and isinstance(dataset.get("source_path"), str):
        source = Path(dataset["source_path"])
        if root is not None:
            root = root.resolve()
            if source.is_absolute():
                errors.append("Bundled dataset source_path must be relative.")
            else:
                unresolved = root / source
                if unresolved.is_symlink():
                    errors.append("Bundled dataset source_path cannot be a symlink.")
                try:
                    resolved = unresolved.resolve(strict=True)
                except FileNotFoundError:
                    resolved = unresolved.resolve()
                if resolved != root and root not in resolved.parents:
                    errors.append(
                        "Bundled dataset source_path escapes the bundle root."
                    )
                source = resolved
        if not source.is_file():
            errors.append("Dataset source file is unavailable.")
        elif isinstance(digest, str) and sha256_file(source) != digest:
            errors.append("Dataset source hash does not match the plan.")

    devices = hardware.get("devices")
    if not isinstance(devices, list) or not devices:
        errors.append("Hardware requires at least one device.")
    elif any(
        not isinstance(item, dict)
        or not _known_text(item.get("backend"), {"cuda", "mps"})
        for item in devices
    ):
        errors.append("Aptus execution plans support CUDA or MPS compute devices.")
    if isinstance(devices, list):
        for index, device in enumerate(devices):
            if not isinstance(device, dict):
                continue
            for capability in ("supports_bf16", "supports_8bit", "supports_4bit"):
                if not isinstance(device.get(capability), bool):
                    errors.append(
                        f"Hardware device {index} {capability} must be boolean."
                    )
            total_vram = device.get("total_vram_bytes")
            free_vram = device.get("free_vram_bytes")
            if not _positive_int(total_vram):
                errors.append(
                    f"Hardware device {index} total_vram_bytes must be positive."
                )
            if free_vram is not None and (
                not _positive_int(free_vram)
                or (_positive_int(total_vram) and free_vram > total_vram)
            ):
                errors.append(
                    f"Hardware device {index} free_vram_bytes must be positive and no greater than total VRAM."
                )
    reserve = hardware.get("reserve_per_device_bytes")
    if not isinstance(reserve, int) or isinstance(reserve, bool) or reserve < 0:
        errors.append(
            "Hardware reserve_per_device_bytes must be a non-negative integer."
        )
    if not _positive_int(hardware.get("host_ram_bytes")):
        errors.append("Hardware host_ram_bytes must be positive.")
    host_free = hardware.get("host_ram_free_bytes")
    if host_free is not None and (
        not _positive_int(host_free)
        or (
            _positive_int(hardware.get("host_ram_bytes"))
            and host_free > hardware["host_ram_bytes"]
        )
    ):
        errors.append(
            "Hardware host_ram_free_bytes must be positive and no greater than host RAM."
        )
    if hardware.get("disk_free_bytes") is not None and not _positive_int(
        hardware.get("disk_free_bytes")
    ):
        errors.append("Hardware disk_free_bytes must be positive when supplied.")
    if (
        isinstance(devices, list)
        and isinstance(reserve, int)
        and not isinstance(reserve, bool)
        and reserve >= 0
    ):
        for index, device in enumerate(devices):
            if (
                isinstance(device, dict)
                and _positive_int(device.get("total_vram_bytes"))
                and reserve >= device["total_vram_bytes"]
            ):
                errors.append(
                    f"Hardware reserve must be smaller than device {index} total VRAM."
                )
    if not _positive_int(target.get("sequence_length")) or not _positive_int(
        target.get("effective_batch_size")
    ):
        errors.append("Target sequence length and effective batch must be positive.")
    if not _positive_int(target.get("max_epochs")):
        errors.append("Target max_epochs must be positive.")
    if not _positive_int(target.get("checkpoint_steps")):
        errors.append("Target checkpoint_steps must be positive.")
    if target.get("optimizer_steps") is not None and not _positive_int(
        target.get("optimizer_steps")
    ):
        errors.append("Target optimizer_steps must be positive when supplied.")
    for seed_name in ("split_seed", "training_seed", "data_order_seed"):
        seed_value = target.get(seed_name)
        if (
            not isinstance(seed_value, int)
            or isinstance(seed_value, bool)
            or seed_value < 0
        ):
            errors.append(f"Target {seed_name} must be a non-negative integer.")
    if (
        all(
            isinstance(target.get(name), int) and not isinstance(target.get(name), bool)
            for name in ("training_seed", "data_order_seed")
        )
        and target["data_order_seed"] != 1_000_000 + target["training_seed"]
    ):
        errors.append("Target data_order_seed must equal 1000000 + training_seed.")
    explicit_micro = target.get("micro_batch_size")
    explicit_accumulation = target.get("gradient_accumulation_steps")
    if (explicit_micro is None) != (explicit_accumulation is None):
        errors.append("Target explicit batch controls must be supplied together.")
    elif explicit_micro is not None and not all(
        _positive_int(value) for value in (explicit_micro, explicit_accumulation)
    ):
        errors.append("Target explicit batch controls must be positive.")
    if target.get("max_wall_time_minutes") is not None and not _positive_int(
        target.get("max_wall_time_minutes")
    ):
        errors.append("Target max_wall_time_minutes must be positive when supplied.")
    elif target.get("max_wall_time_minutes") is not None:
        errors.append(
            "Target max_wall_time_minutes must be null in Aptus v0.2 because execution does not enforce it."
        )
    if target.get("task") != "sft":
        errors.append("Aptus v0.2 target task must be sft.")
    if not _known_text(target.get("objective"), {"quality", "memory", "speed"}):
        errors.append("Target objective is invalid.")
    if target.get("method_preference") is not None and not _known_text(
        target.get("method_preference"), METHODS
    ):
        errors.append("Target method_preference is invalid.")
    if target.get("training_runtime") is not None and not _known_text(
        target.get("training_runtime"), TRAINING_RUNTIMES
    ):
        errors.append("Target training_runtime is invalid.")
    if target.get("packing") is not False:
        errors.append("Aptus v0.2 target packing must be false.")
    evaluation_fraction = target.get("evaluation_fraction")
    if not _finite_number(evaluation_fraction) or not 0 <= evaluation_fraction < 1:
        errors.append("Target evaluation_fraction must be in [0, 1).")
    if _positive_int(model.get("context_length")) and _positive_int(
        target.get("sequence_length")
    ):
        if target["sequence_length"] > model["context_length"]:
            errors.append("Target sequence length exceeds model context length.")

    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append("Plan candidates must be a non-empty list.")
        candidates = []
    historical_policy_validation = bool(
        expected_policy_decision_override is not None and not enforce_current_policy
    )
    historical_adapter_targets: list[str] | None = None
    historical_adapter_targets_registered = False
    if historical_policy_validation:
        assert expected_policy_decision_override is not None
        historical_kind = expected_policy_decision_override.get("kind")
        historical_policy_id = expected_policy_decision_override.get("policy_id")
        target_variants: list[tuple[str, ...]] = []
        if historical_kind == "path-matched":
            historical_paths = expected_policy_decision_override.get("paths")
            if isinstance(historical_paths, list):
                for path in historical_paths:
                    targets = (
                        path.get("target_modules") if isinstance(path, dict) else None
                    )
                    if (
                        isinstance(targets, list)
                        and targets
                        and all(isinstance(item, str) for item in targets)
                    ):
                        variant = tuple(targets)
                        if variant not in target_variants:
                            target_variants.append(variant)
        elif historical_kind == "family-recognized" or (
            historical_kind == "blocked" and historical_policy_id is not None
        ):
            for candidate in candidates:
                if not isinstance(candidate, dict) or not _known_text(
                    candidate.get("method"), METHODS - {"full"}
                ):
                    continue
                targets = candidate.get("target_modules")
                if (
                    isinstance(targets, list)
                    and targets
                    and all(isinstance(item, str) for item in targets)
                ):
                    variant = tuple(targets)
                    if variant not in target_variants:
                        target_variants.append(variant)
        if _known_text(historical_kind, {"path-matched", "family-recognized"}) or (
            historical_kind == "blocked" and historical_policy_id is not None
        ):
            if len(target_variants) != 1:
                errors.append(
                    "Historical model policy must establish one internally "
                    "consistent adapter-target set."
                )
            else:
                historical_adapter_targets = list(target_variants[0])
                historical_adapter_targets_registered = True
    candidate_ids: set[str] = set()
    candidate_by_id: dict[str, dict[str, Any]] = {}
    strategy_pairs: set[tuple[str, str]] = set()
    evidence_records = plan.get("evidence_records")
    if not isinstance(evidence_records, list):
        errors.append("Plan evidence_records must be a list.")
        evidence_records = []
    evidence_ids: set[str] = set()
    evidence_id_order: list[str] = []
    required_evidence_fields = {
        "evidence_id",
        "claim",
        "source",
        "source_kind",
        "scope",
        "confidence",
        "revision",
    }
    for index, record in enumerate(evidence_records):
        name = f"Evidence record {index}"
        if not isinstance(record, dict):
            errors.append(f"{name} must be an object.")
            continue
        if set(record) != required_evidence_fields:
            errors.append(f"{name} must contain the exact v1 fields.")
        for field in required_evidence_fields - {"revision"}:
            if not isinstance(record.get(field), str) or not record[field].strip():
                errors.append(f"{name} requires non-empty string {field}.")
        evidence_id = record.get("evidence_id")
        if isinstance(evidence_id, str) and evidence_id:
            evidence_id_order.append(evidence_id)
            if evidence_id in evidence_ids:
                errors.append(f"Duplicate evidence ID: {evidence_id}.")
            else:
                evidence_ids.add(evidence_id)
            canonical_digest = EVIDENCE_RECORD_SHA256.get(evidence_id)
            if canonical_digest is None:
                errors.append(f"{name} uses an unknown evidence ID.")
            elif _sha256_json(record) != canonical_digest:
                errors.append(
                    f"{name} content does not match the canonical evidence registry."
                )
        revision = record.get("revision")
        if revision is not None and (
            not isinstance(revision, str) or not revision.strip()
        ):
            errors.append(f"{name} revision must be null or a non-empty string.")
    if evidence_id_order != sorted(set(evidence_id_order)):
        errors.append("Plan evidence records must be sorted by unique evidence ID.")
    referenced_evidence_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        name = f"Candidate {index}"
        if not isinstance(candidate, dict):
            errors.append(f"{name} must be an object.")
            continue
        expected_policy_binding: Mapping[str, Any] | None = None
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            errors.append(f"{name} requires candidate_id.")
        elif candidate_id in candidate_ids:
            errors.append(f"Duplicate candidate ID: {candidate_id}.")
        else:
            candidate_ids.add(candidate_id)
            candidate_by_id[candidate_id] = candidate
        expected_decision_id = (
            expected_policy_decision.get("decision_id")
            if expected_policy_decision is not None
            else policy_decision.get("decision_id")
        )
        if candidate.get("model_policy_decision_id") != expected_decision_id:
            errors.append(f"{name} must link to the current model policy decision ID.")
        if "policy_binding" not in candidate:
            errors.append(f"{name} requires an explicit policy_binding field.")
        binding_value = candidate.get("policy_binding")
        matching_path = (
            _matching_policy_path(expected_policy_decision, candidate)
            if expected_policy_decision is not None
            else None
        )
        if matching_path is None:
            if binding_value is not None:
                errors.append(
                    f"{name} policy_binding must be null because no registered "
                    "policy path matches the candidate."
                )
        else:
            expected_policy_binding = _expected_policy_binding(
                decision=expected_policy_decision,
                path=matching_path,
                source=str(policy_source),
                receipt=inspection_receipt,
            )
            if not isinstance(binding_value, Mapping):
                errors.append(
                    f"{name} requires a policy_binding for its registered policy path."
                )
            elif binding_value != expected_policy_binding:
                errors.append(
                    f"{name} policy_binding is stale, tampered, or does not match "
                    "the current registered path."
                )
        candidate_method = candidate.get("method")
        if not _known_text(candidate_method, METHODS):
            errors.append(f"{name} method is invalid.")
        runtime_contract = candidate.get("runtime_contract")
        runtime_id = "transformers-peft-cuda"
        runtime_backend = "cuda"
        runtime_estimator = FORMULA_VERSION
        if not isinstance(runtime_contract, dict):
            errors.append(f"{name} runtime_contract must be an object.")
        elif isinstance(runtime_contract, dict):
            runtime_id = runtime_contract.get("training_runtime")
            runtime_backend = runtime_contract.get("compute_backend")
            runtime_estimator = runtime_contract.get("estimator_id")
            if runtime_contract.get("schema_version") != RUNTIME_CONTRACT_VERSION:
                errors.append(
                    f"{name} runtime contract schema must be {RUNTIME_CONTRACT_VERSION}."
                )
            if not _known_text(runtime_id, TRAINING_RUNTIMES):
                errors.append(f"{name} training runtime is invalid.")
            if not _known_text(runtime_backend, {"cuda", "mps"}):
                errors.append(f"{name} runtime compute backend is invalid.")
            expected_runtime_backend = (
                {
                    "transformers-peft-cuda": "cuda",
                    "mlx-lm": "mps",
                    "pytorch-mps": "mps",
                }.get(runtime_id)
                if isinstance(runtime_id, str)
                else None
            )
            if expected_runtime_backend and runtime_backend != expected_runtime_backend:
                errors.append(f"{name} runtime and compute backend do not match.")
            if not _known_text(
                runtime_contract.get("evidence_requirement"), EVIDENCE_REQUIREMENTS
            ):
                errors.append(f"{name} runtime evidence requirement is invalid.")
            expected_runtime_identity = (
                RUNTIME_BINDING_IDENTITIES.get(
                    (candidate_method, runtime_id, runtime_backend)
                )
                if all(
                    isinstance(item, str)
                    for item in (candidate_method, runtime_id, runtime_backend)
                )
                else None
            )
            actual_runtime_identity = (
                runtime_contract.get("compiler_id"),
                runtime_contract.get("estimator_id"),
                runtime_contract.get("export_kind"),
                runtime_contract.get("evidence_requirement"),
            )
            if expected_runtime_identity is None:
                if actual_runtime_identity != UNAVAILABLE_RUNTIME_IDENTITY:
                    errors.append(
                        f"{name} unregistered method/runtime/backend contract must use the exact unavailable identity."
                    )
            elif actual_runtime_identity != expected_runtime_identity:
                errors.append(
                    f"{name} runtime contract does not match its registered compiler, estimator, export, and evidence identity."
                )
            viable_runtime = _known_text(
                candidate.get("status"), {"feasible", "conditional"}
            )
            if viable_runtime:
                if expected_runtime_identity is None:
                    errors.append(
                        f"{name} viable runtime requires a registered method/runtime/backend compiler binding."
                    )
                elif runtime_contract.get("evidence_requirement") != "pilot-required":
                    errors.append(f"{name} viable runtime must remain pilot-required.")
        if not _known_text(candidate.get("precision"), {"bf16", "fp16"}):
            errors.append(f"{name} precision is invalid.")
        learning_rate = candidate.get("learning_rate")
        if not _finite_number(learning_rate) or learning_rate <= 0:
            errors.append(f"{name} learning_rate must be positive and finite.")
        if (
            candidate.get("method") == "full"
            and candidate.get("precision") == "fp16"
            and _known_text(candidate.get("status"), {"feasible", "conditional"})
        ):
            errors.append(
                f"{name} full-parameter FP16 execution is unsupported in Aptus v0.2."
            )
        if not _known_text(candidate.get("distribution"), DISTRIBUTIONS):
            errors.append(f"{name} distribution is invalid.")
        elif _known_text(candidate.get("method"), METHODS):
            strategy_pairs.add((candidate["method"], candidate["distribution"]))
        if not _known_text(candidate.get("status"), CANDIDATE_STATUSES):
            errors.append(f"{name} status is invalid.")
        elif candidate.get("feasible") is not (
            _known_text(candidate["status"], {"feasible", "conditional"})
        ):
            errors.append(f"{name} feasible flag does not match status.")
        for key in (
            "micro_batch_size",
            "gradient_accumulation_steps",
            "effective_batch_size",
            "world_size",
        ):
            if not _positive_int(candidate.get(key)):
                errors.append(f"{name} {key} must be positive.")
        device_indices = candidate.get("device_indices")
        selected_devices: list[dict[str, Any]] = []
        if (
            not isinstance(device_indices, list)
            or len(device_indices) != candidate.get("world_size")
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in device_indices
            )
            or len(set(device_indices)) != len(device_indices)
        ):
            errors.append(
                f"{name} device_indices must contain one unique non-negative integer per world rank."
            )
        elif not isinstance(devices, list) or any(
            item >= len(devices) for item in device_indices
        ):
            errors.append(f"{name} device_indices reference unavailable hardware.")
        elif not all(isinstance(devices[item], dict) for item in device_indices):
            errors.append(f"{name} device_indices reference invalid hardware facts.")
        else:
            selected_devices = [devices[item] for item in device_indices]
            selected_backends = {item.get("backend") for item in selected_devices}
            if len(selected_backends) != 1:
                errors.append(f"{name} cannot mix compute backends.")
            elif runtime_backend not in selected_backends:
                errors.append(
                    f"{name} runtime compute backend does not match selected hardware."
                )
        zero_checkpoint_allowed = bool(
            _known_text(candidate_method, METHODS)
            and candidate_method != "full"
            and candidate.get("status") == "unsupported"
            and candidate.get("target_modules") in ([], ())
        )
        for key in (
            "required_host_ram_bytes",
            "required_disk_bytes",
            "checkpoint_retention_bytes",
            "final_export_bytes",
        ):
            value = candidate.get(key)
            if key == "checkpoint_retention_bytes" and zero_checkpoint_allowed:
                if not (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                ):
                    errors.append(
                        f"{name} {key} must be non-negative for an unsupported "
                        "zero-target adapter candidate."
                    )
            elif not _positive_int(value):
                errors.append(f"{name} {key} must be positive.")
        if all(
            _positive_int(candidate.get(key))
            for key in ("micro_batch_size", "gradient_accumulation_steps", "world_size")
        ):
            calculated = (
                candidate["micro_batch_size"]
                * candidate["gradient_accumulation_steps"]
                * candidate["world_size"]
            )
            if calculated != candidate.get(
                "effective_batch_size"
            ) or calculated != target.get("effective_batch_size"):
                errors.append(f"{name} global batch arithmetic is invalid.")
            if target.get("micro_batch_size") is not None and (
                candidate["micro_batch_size"] != target["micro_batch_size"]
                or candidate["gradient_accumulation_steps"]
                != target["gradient_accumulation_steps"]
            ):
                errors.append(f"{name} does not bind the explicit batch controls.")
            expected_world = (
                1
                if candidate.get("distribution") == "single"
                else len(devices)
                if isinstance(devices, list)
                else 0
            )
            if candidate["world_size"] != expected_world:
                errors.append(
                    f"{name} world_size does not match its distribution and hardware."
                )
        memory = candidate.get("memory")
        if not isinstance(memory, dict):
            errors.append(f"{name} memory must be an object.")
        else:
            component_names = (
                "base_weights_bytes",
                "quantization_metadata_bytes",
                "adapter_weights_bytes",
                "adapter_gradients_bytes",
                "optimizer_states_bytes",
                "activations_bytes",
                "temporary_overhead_bytes",
                "communication_bytes",
                "workspace_bytes",
                "allocator_bytes",
                "load_transient_bytes",
            )
            if all(
                isinstance(memory.get(key), int)
                and not isinstance(memory[key], bool)
                and memory[key] >= 0
                for key in component_names
            ):
                point = sum(memory[key] for key in component_names)
                if (
                    memory.get("point_estimate_bytes") != point
                    or memory.get("estimated_peak_bytes") != point
                ):
                    errors.append(f"{name} point memory does not equal its components.")
                bounds = memory.get("component_upper_bounds")
                if not isinstance(bounds, dict) or not bounds:
                    errors.append(
                        f"{name} requires transparent component_upper_bounds."
                    )
                elif not all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in bounds.values()
                ):
                    errors.append(
                        f"{name} upper memory components must be non-negative integers."
                    )
                else:
                    upper = sum(bounds.values())
                    if memory.get("upper_estimate_bytes") != upper or upper < point:
                        errors.append(
                            f"{name} upper memory must equal its component upper bounds and cover the point estimate."
                        )
                    if bounds.get("uncertainty_bytes") != memory.get(
                        "safety_margin_bytes"
                    ):
                        errors.append(
                            f"{name} uncertainty_bytes must equal the named safety margin."
                        )
                if memory.get("uncertainty_bytes") != memory.get("safety_margin_bytes"):
                    errors.append(
                        f"{name} uncertainty alias must equal the named safety margin."
                    )
                expected_memory_formula = (
                    MLX_FORMULA_VERSION
                    if runtime_id == "mlx-lm"
                    and _known_text(candidate.get("method"), {"lora", "qlora"})
                    else FORMULA_VERSION
                )
                if memory.get("formula_version") != expected_memory_formula:
                    errors.append(
                        f"{name} memory formula must be {expected_memory_formula}."
                    )
                if runtime_id == "mlx-lm" and _known_text(
                    candidate.get("method"), {"lora", "qlora"}
                ):
                    try:
                        recomputed_memory = mlx_memory_breakdown_for_contract(
                            model=model,
                            target=target,
                            candidate=candidate,
                        )
                    except (OverflowError, TypeError, ValueError) as error:
                        errors.append(
                            f"{name} MLX memory could not be recomputed from bound facts: {error}"
                        )
                    else:
                        observed_memory = {
                            key: memory.get(key) for key in recomputed_memory
                        }
                        if observed_memory != recomputed_memory:
                            errors.append(
                                f"{name} MLX memory does not match deterministic recomputation from bound facts."
                            )
                if (
                    _known_text(candidate.get("status"), {"feasible", "conditional"})
                    and runtime_estimator != expected_memory_formula
                ):
                    errors.append(
                        f"{name} runtime estimator does not match its memory formula."
                    )
                if (
                    _known_text(candidate.get("status"), {"feasible", "conditional"})
                    and selected_devices
                    and isinstance(reserve, int)
                ):
                    if runtime_backend == "mps":
                        if not _positive_int(host_free):
                            errors.append(
                                f"{name} viable MLX status requires measured host RAM free."
                            )
                            capacities = []
                        else:
                            capacities = [host_free]
                    else:
                        frees = [
                            item.get("free_vram_bytes") for item in selected_devices
                        ]
                        if any(not _positive_int(value) for value in frees):
                            errors.append(
                                f"{name} viable status requires measured free per-device memory."
                            )
                            capacities = []
                        else:
                            capacities = [int(value) for value in frees]
                    usable = (
                        min(capacity - reserve for capacity in capacities)
                        if capacities
                        else 0
                    )
                    if point > usable:
                        errors.append(
                            f"{name} viable status exceeds usable per-device memory at its point estimate."
                        )
                    if (
                        candidate.get("status") == "feasible"
                        and memory.get("upper_estimate_bytes", 0) > usable
                    ):
                        errors.append(
                            f"{name} feasible status exceeds usable per-device memory at its heuristic upper envelope."
                        )
            else:
                errors.append(
                    f"{name} memory components must be non-negative integers."
                )
        method = candidate.get("method")
        quantization = candidate.get("quantization")
        mlx_qlora_bits = model.get("quantization_bits")
        expected_quantization = (
            f"mlx-{mlx_qlora_bits}bit-groupwise"
            if runtime_id == "mlx-lm"
            and method == "qlora"
            and isinstance(mlx_qlora_bits, int)
            and not isinstance(mlx_qlora_bits, bool)
            and 1 <= mlx_qlora_bits <= 16
            else (
                "mlx-4bit-groupwise"
                if runtime_id == "mlx-lm" and method == "qlora"
                else (
                    {
                        "full": None,
                        "lora": None,
                        "int8-lora": "int8-bitsandbytes",
                        "qlora": "nf4-double-quant",
                    }.get(method)
                    if isinstance(method, str)
                    else None
                )
            )
        )
        if _known_text(method, METHODS) and quantization != expected_quantization:
            errors.append(f"{name} quantization does not match method.")
        model_family = model.get("family")
        current_expected_targets = (
            MODEL_TARGET_MODULES.get(model_family)
            if isinstance(model_family, str)
            else None
        )
        expected_targets = (
            historical_adapter_targets
            if historical_policy_validation
            else current_expected_targets
        )
        adapter_targets_registered = (
            historical_adapter_targets_registered
            if historical_policy_validation
            else current_expected_targets is not None
        )
        if method != "full" and _known_text(method, METHODS):
            if not adapter_targets_registered:
                if candidate.get("target_modules") not in ([], ()):
                    errors.append(
                        f"{name} unregistered model families cannot carry adapter targets."
                    )
                if (
                    candidate.get("status") != "unsupported"
                    or candidate.get("feasible") is not False
                ):
                    errors.append(
                        f"{name} adapter path must be unsupported for an unregistered model family."
                    )
            elif candidate.get("target_modules") != expected_targets:
                errors.append(
                    f"{name} target modules do not match the exact model-family policy."
                )
        if method == "full" and (
            candidate.get("rank") != 0
            or candidate.get("alpha") != 0
            or candidate.get("target_modules") not in ([], ())
        ):
            errors.append(f"{name} full fine-tuning cannot carry adapter fields.")
        if (
            method != "full"
            and _known_text(method, METHODS)
            and (
                not _positive_int(candidate.get("rank"))
                or not _positive_int(candidate.get("alpha"))
                or (adapter_targets_registered and not candidate.get("target_modules"))
            )
        ):
            errors.append(
                f"{name} adapter method requires rank, alpha, and target modules."
            )
        if moe_identity:
            moe_family = _is_reviewed_moe_identity(
                model, model_type=model_type, architecture=architecture
            )
            expected_layout = None
            if moe_family == QWEN3_MOE_FAMILY:
                expected_layout = _reviewed_qwen3_moe_quantization_layout(
                    model.get("layers")
                )
            elif moe_family == GEMMA4_MOE_FAMILY:
                expected_layout = _reviewed_gemma4_moe_quantization_layout(
                    model.get("layers")
                )
            reviewed_moe_runtime = (
                moe_family is not None
                and method == "qlora"
                and candidate.get("distribution") == "single"
                and runtime_id == "mlx-lm"
                and runtime_backend == "mps"
                and quantization == "mlx-4bit-groupwise"
                and model.get("quantization_bits") == 4
                and model.get("quantization_layout") == expected_layout
                and isinstance(model.get("moe"), dict)
                and model["moe"].get("shared_expert_intermediate_size") is None
            )
            if not reviewed_moe_runtime and candidate.get("status") != "unsupported":
                errors.append(
                    f"{name} violates the exact single-device MLX-LM QLoRA MoE policy."
                )
            if reviewed_moe_runtime and candidate.get("status") == "feasible":
                errors.append(
                    f"{name} MoE execution must remain conditional pending its measured pilot."
                )
        if (
            _known_text(candidate.get("status"), {"feasible", "conditional"})
            and selected_devices
        ):
            if candidate.get("precision") == "bf16" and any(
                not item.get("supports_bf16") for item in selected_devices
            ):
                errors.append(f"{name} uses bf16 without device support.")
            if (
                runtime_id != "mlx-lm"
                and method == "qlora"
                and any(not item.get("supports_4bit") for item in selected_devices)
            ):
                errors.append(
                    f"{name} uses four-bit quantization without device support."
                )
            if method == "int8-lora" and any(
                not item.get("supports_8bit") for item in selected_devices
            ):
                errors.append(
                    f"{name} uses eight-bit quantization without device support."
                )
            if candidate.get("distribution") == "fsdp" and _known_text(
                method, {"int8-lora", "qlora"}
            ):
                errors.append(f"{name} uses an unsupported quantized FSDP combination.")
            if runtime_id == "mlx-lm":
                if not _known_text(method, {"lora", "qlora"}):
                    errors.append(f"{name} MLX-LM method is unsupported.")
                if candidate.get("distribution") != "single":
                    errors.append(f"{name} MLX-LM distribution must be single.")
                if candidate.get("status") != "conditional":
                    errors.append(
                        f"{name} MLX-LM status must remain conditional until pilot evidence."
                    )
                if runtime_contract and "bitsandbytes" in json.dumps(
                    runtime_contract, sort_keys=True
                ):
                    errors.append(
                        f"{name} MLX-LM contract cannot use bitsandbytes identity."
                    )
        if isinstance(candidate_id, str) and candidate_id != candidate_id_for_payload(
            candidate,
            model=model,
            dataset=dataset,
            hardware=hardware,
            target=target,
        ):
            errors.append(
                f"{name} immutable candidate ID does not match its normalized execution contract."
            )
        candidate_evidence = candidate.get("evidence")
        if not isinstance(candidate_evidence, list) or any(
            not isinstance(item, str) or not item for item in candidate_evidence or ()
        ):
            errors.append(f"{name} evidence must be a list of non-empty IDs.")
        else:
            if len(candidate_evidence) != len(set(candidate_evidence)):
                errors.append(f"{name} evidence IDs must be unique.")
            expected_candidate_evidence = list(
                METHOD_EVIDENCE_IDS.get(candidate_method, ())
                if isinstance(candidate_method, str)
                else ()
            )
            if expected_policy_binding is not None:
                expected_candidate_evidence.extend(
                    expected_policy_binding["evidence_ids"]
                )
            if _known_text(candidate_method, METHODS) and (
                candidate_evidence != expected_candidate_evidence
            ):
                errors.append(
                    f"{name} evidence must match its method and policy contract."
                )
            if expected_policy_binding is not None and any(
                evidence_id not in candidate_evidence
                for evidence_id in expected_policy_binding["evidence_ids"]
            ):
                errors.append(
                    f"{name} evidence must include every bound model-policy record."
                )
            for evidence_id in candidate_evidence:
                referenced_evidence_ids.add(evidence_id)
                if evidence_id not in evidence_ids:
                    errors.append(
                        f"{name} references missing evidence ID {evidence_id}."
                    )

    if evidence_ids != referenced_evidence_ids:
        errors.append(
            "Plan evidence records must exactly match the candidate evidence IDs."
        )

    expected_pairs = {
        (method, distribution) for method in METHODS for distribution in DISTRIBUTIONS
    }
    if strategy_pairs != expected_pairs or len(candidates) != len(expected_pairs):
        errors.append(
            "Plan must contain exactly one candidate for every method and distribution pair."
        )

    recommended = plan.get("recommended")
    if not isinstance(recommended, dict):
        errors.append("Recommended candidate must be an object.")
    else:
        recommended_id = recommended.get("candidate_id")
        listed = candidate_by_id.get(recommended_id)
        if listed is None:
            errors.append("Recommended candidate must appear in candidates.")
        elif listed != recommended:
            errors.append(
                "Recommended candidate must exactly match its listed candidate."
            )
        if not _known_text(recommended.get("status"), {"feasible", "conditional"}):
            errors.append("Recommended candidate must be feasible or conditional.")
    if plan.get("plan_id") != plan_id_for_payload(plan):
        errors.append(
            "Plan immutable ID does not match its normalized facts, candidates, and recommendation."
        )
    return tuple(errors)


def validate_plan_payload(
    plan_value: Any,
    *,
    root: Path | None = None,
    verify_dataset: bool = True,
    policy_snapshot: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Validate a plan against the current portable policy contract."""

    try:
        return _validate_plan_payload_impl(
            plan_value,
            root=root,
            verify_dataset=verify_dataset,
            policy_snapshot=policy_snapshot,
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        return (f"Plan structure is malformed: {error}",)
