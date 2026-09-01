"""Portable, package-independent model-policy snapshot primitives."""

from __future__ import annotations

import hashlib
import json
from string import Formatter
from typing import Any, Mapping


MODEL_POLICY_SNAPSHOT_SCHEMA_VERSION = "aptus.model-policy-snapshot.v1"
_CONSTRAINT_KINDS = {
    "exact_identity",
    "quantization_layout",
    "sparse_topology",
    "no_shared_expert",
    "field_equals",
    "compiler_contract",
}
_REQUIRED_REASONS = {
    "identity",
    "layout",
    "topology",
    "shared",
    "four_bit",
    "invalid",
    "matched",
    "dense",
    "sparse",
    "unknown",
}
_COMPATIBILITY_SUBJECT_FIELDS = (
    "family",
    "model_type",
    "architecture",
    "layers",
    "quantization_bits",
    "quantization_layout",
    "moe",
)
_IDENTITY_FIELDS = {"family", "model_type", "architecture"}
_QUANTIZATION_INTEGER_OPERANDS = (
    "default_bits",
    "default_group_size",
    "override_bits",
    "override_group_size",
)
_PATH_FIELDS = {
    "path_id",
    "method",
    "distribution",
    "adapter_profile_id",
    "target_modules",
    "runtime_contract",
    "required_validation_levels",
    "evidence_ids",
}
_RUNTIME_CONTRACT_FIELDS = {
    "compute_backend",
    "training_runtime",
    "compiler_id",
    "estimator_id",
    "evidence_requirement",
    "export_kind",
    "schema_version",
}


def _canonical_json(value: Any, *, newline: bool) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def model_policy_snapshot_payload(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached, validated snapshot from primitive registry data."""

    if not isinstance(registry, Mapping):
        raise ValueError("Model policy registry must be a mapping.")
    if "schema_version" in registry:
        raise ValueError("Registry data must not supply the snapshot schema version.")
    try:
        detached = json.loads(_canonical_json(dict(registry), newline=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Model policy registry must contain JSON values.") from error
    snapshot = {"schema_version": MODEL_POLICY_SNAPSHOT_SCHEMA_VERSION, **detached}
    validate_model_policy_snapshot(snapshot)
    return snapshot


def model_policy_snapshot_bytes(snapshot: Mapping[str, Any]) -> bytes:
    validate_model_policy_snapshot(snapshot)
    return _canonical_json(snapshot, newline=True)


def model_policy_snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    return hashlib.sha256(model_policy_snapshot_bytes(snapshot)).hexdigest()


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(
            f"Model policy snapshot {label} must be non-empty unpadded text."
        )
    return value


def _require_text_list(
    value: Any,
    label: str,
    *,
    non_empty: bool,
) -> list[str]:
    if (
        not isinstance(value, list)
        or (non_empty and not value)
        or any(
            not isinstance(item, str) or not item.strip() or item != item.strip()
            for item in value
        )
    ):
        qualifier = "non-empty " if non_empty else ""
        raise ValueError(
            f"Model policy snapshot {label} must be a {qualifier}unpadded text list."
        )
    if len(set(value)) != len(value):
        raise ValueError(f"Model policy snapshot {label} must be unique.")
    return value


def _require_positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Model policy snapshot {label} must be a positive integer.")
    return value


def _validate_override_module_template(value: Any) -> None:
    _require_string(value, "override module template")
    remainder = value.replace("{layer}", "", 1)
    if "{" in remainder or "}" in remainder:
        raise ValueError(
            "Model policy snapshot override module template may only contain "
            "the {layer} placeholder."
        )
    try:
        parsed = list(Formatter().parse(value))
    except ValueError as error:
        raise ValueError(
            "Model policy snapshot override module template is malformed."
        ) from error
    placeholders = [
        (field_name, format_spec, conversion)
        for _, field_name, format_spec, conversion in parsed
        if field_name is not None
    ]
    if placeholders != [("layer", "", None)]:
        raise ValueError(
            "Model policy snapshot override module template must contain exactly "
            "one plain {layer} placeholder."
        )


def _validate_path(path: Any, path_ids: set[str]) -> None:
    if not isinstance(path, Mapping) or set(path) != _PATH_FIELDS:
        raise ValueError(
            "Each model policy snapshot path must have the exact portable shape."
        )
    path_id = _require_string(path.get("path_id"), "path id")
    if path_id in path_ids:
        raise ValueError("Model policy snapshot path ids must be unique.")
    path_ids.add(path_id)

    for field in ("method", "distribution"):
        _require_string(path[field], f"path {field}")
    if path["adapter_profile_id"] is not None:
        _require_string(path["adapter_profile_id"], "path adapter profile id")
    for field in ("target_modules", "required_validation_levels", "evidence_ids"):
        _require_text_list(
            path[field],
            f"path {field.replace('_', ' ')}",
            non_empty=True,
        )

    contract = path["runtime_contract"]
    if not isinstance(contract, Mapping) or set(contract) != _RUNTIME_CONTRACT_FIELDS:
        raise ValueError(
            "Model policy snapshot path runtime contract must have the exact "
            "portable shape."
        )
    for field in _RUNTIME_CONTRACT_FIELDS - {"compiler_id", "export_kind"}:
        _require_string(contract[field], f"path runtime contract {field}")
    for field in ("compiler_id", "export_kind"):
        if contract[field] is not None:
            _require_string(contract[field], f"path runtime contract {field}")


def validate_model_policy_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Reject incomplete or non-portable policy snapshot structures."""

    if not isinstance(snapshot, Mapping):
        raise ValueError("Model policy snapshot must be a mapping.")
    try:
        _canonical_json(snapshot, newline=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Model policy snapshot must contain JSON values.") from error
    if snapshot.get("schema_version") != MODEL_POLICY_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("Unsupported model policy snapshot schema version.")
    _require_string(
        snapshot.get("compatibility_schema_version"), "compatibility schema"
    )
    dense = snapshot.get("dense_families")
    markers = snapshot.get("sparse_identity_markers")
    _require_text_list(dense, "dense families", non_empty=False)
    if dense != sorted(set(dense)):
        raise ValueError(
            "Model policy snapshot dense families must be sorted and unique."
        )
    _require_text_list(markers, "sparse markers", non_empty=True)
    if markers != sorted(set(markers)):
        raise ValueError(
            "Model policy snapshot sparse markers must be sorted and unique."
        )
    reasons = snapshot.get("reasons")
    if not isinstance(reasons, Mapping) or not _REQUIRED_REASONS.issubset(reasons):
        raise ValueError("Model policy snapshot reasons are incomplete.")
    for key, value in reasons.items():
        _require_string(key, "reason key")
        _require_string(value, "reason")
    policies = snapshot.get("policies")
    if not isinstance(policies, list):
        raise ValueError("Model policy snapshot policies must be a list.")
    identities: set[str] = set()
    path_ids: set[str] = set()
    for policy in policies:
        if not isinstance(policy, Mapping):
            raise ValueError("Each model policy snapshot policy must be a mapping.")
        policy_id = _require_string(policy.get("policy_id"), "policy id")
        if policy_id in identities:
            raise ValueError("Model policy snapshot policy ids must be unique.")
        identities.add(policy_id)
        for field in ("policy_version", "family", "matched_reason"):
            _require_string(policy.get(field), field.replace("_", " "))
        claims = policy.get("claims")
        if not isinstance(claims, Mapping) or set(claims) != {"any_identity"}:
            raise ValueError(
                "Model policy snapshot claims must contain only any_identity data."
            )
        any_identity = claims.get("any_identity")
        if not isinstance(any_identity, Mapping) or not any_identity:
            raise ValueError("Model policy snapshot claims require any_identity data.")
        for field, values in any_identity.items():
            if field not in _IDENTITY_FIELDS:
                raise ValueError("Model policy snapshot identity claims are malformed.")
            _require_text_list(
                values,
                f"identity claims for {field}",
                non_empty=True,
            )
        constraints = policy.get("constraints")
        if not isinstance(constraints, list) or not constraints:
            raise ValueError(
                "Model policy snapshot constraints must be a non-empty list."
            )
        exact_identities: list[Mapping[str, Any]] = []
        for constraint in constraints:
            if not isinstance(constraint, Mapping):
                raise ValueError(
                    "Model policy snapshot contains an unknown constraint."
                )
            kind = constraint.get("kind")
            if not isinstance(kind, str) or kind not in _CONSTRAINT_KINDS:
                raise ValueError(
                    "Model policy snapshot contains an unknown constraint."
                )
            _require_string(constraint.get("reason"), "constraint reason key")
            if constraint["reason"] not in reasons:
                raise ValueError(
                    "Model policy snapshot constraint reason is undefined."
                )
            _require_string(constraint.get("reason_code"), "constraint reason code")
            operands = {
                "exact_identity": {"values"},
                "quantization_layout": {
                    "default_bits",
                    "default_group_size",
                    "override_module_template",
                    "override_bits",
                    "override_group_size",
                },
                "sparse_topology": set(),
                "no_shared_expert": set(),
                "field_equals": {"field", "value"},
                "compiler_contract": set(),
            }[kind]
            expected_fields = {"kind", "reason", "reason_code"} | operands
            if set(constraint) != expected_fields:
                raise ValueError(
                    f"Model policy snapshot {kind} constraint shape is malformed."
                )
            if kind == "exact_identity":
                values = constraint["values"]
                if not isinstance(values, Mapping) or set(values) != _IDENTITY_FIELDS:
                    raise ValueError(
                        "Model policy snapshot exact identity constraint is malformed."
                    )
                for value in values.values():
                    _require_string(value, "exact identity value")
                exact_identities.append(constraint)
            elif kind == "quantization_layout":
                for field in _QUANTIZATION_INTEGER_OPERANDS:
                    _require_positive_integer(
                        constraint[field],
                        f"quantization {field.replace('_', ' ')}",
                    )
                _validate_override_module_template(
                    constraint["override_module_template"]
                )
            elif kind == "field_equals":
                field = _require_string(constraint["field"], "field equals field")
                if field not in _COMPATIBILITY_SUBJECT_FIELDS:
                    raise ValueError(
                        "Model policy snapshot field equals constraint names an "
                        "unsupported compatibility fact."
                    )

        if len(exact_identities) != 1:
            raise ValueError(
                "Each model policy snapshot policy must contain exactly one "
                "exact identity constraint."
            )
        exact_values = exact_identities[0]["values"]
        if any(
            exact_values[field] not in values for field, values in any_identity.items()
        ):
            raise ValueError(
                "Model policy snapshot claimed exact identity values must be "
                "represented by the policy claims."
            )

        paths = policy.get("paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError("Model policy snapshot paths must be a non-empty list.")
        for path in paths:
            _validate_path(path, path_ids)
        _require_text_list(
            policy.get("matched_reason_codes"),
            "matched reason codes",
            non_empty=True,
        )
        _require_text_list(
            policy.get("evidence_ids"),
            "evidence ids",
            non_empty=True,
        )
        if "required_provenance_fields" in policy:
            _require_text_list(
                policy["required_provenance_fields"],
                "required provenance fields",
                non_empty=True,
            )
        if policy["matched_reason"] not in reasons:
            raise ValueError("Model policy snapshot matched reason is undefined.")


def _compatibility_subject_payload(subject: Mapping[str, Any]) -> dict[str, Any]:
    """Return the host-compatible fixed facts used by decision identity."""

    return {
        **{field: subject.get(field) for field in _COMPATIBILITY_SUBJECT_FIELDS},
        "fact_errors": sorted(subject.get("fact_errors", [])),
    }


def _subject_digest(subject: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(_compatibility_subject_payload(subject), newline=False)
    ).hexdigest()


def _claims(policy: Mapping[str, Any], subject: Mapping[str, Any]) -> bool:
    return any(
        subject.get(field) in values
        for field, values in policy["claims"]["any_identity"].items()
    )


def _has_sparse_identity(
    snapshot: Mapping[str, Any], subject: Mapping[str, Any]
) -> bool:
    return any(
        marker in value.lower()
        for value in (
            subject.get("family"),
            subject.get("model_type"),
            subject.get("architecture"),
        )
        if isinstance(value, str)
        for marker in snapshot["sparse_identity_markers"]
    )


def _policy_has_sparse_identity(
    snapshot: Mapping[str, Any], policy: Mapping[str, Any]
) -> bool:
    identity = next(
        constraint
        for constraint in policy["constraints"]
        if constraint["kind"] == "exact_identity"
    )
    return any(
        marker in value.lower()
        for value in identity["values"].values()
        for marker in snapshot["sparse_identity_markers"]
    )


def _eligible_claim(
    snapshot: Mapping[str, Any],
    policy: Mapping[str, Any],
    subject: Mapping[str, Any],
) -> bool:
    if not _claims(policy, subject):
        return False
    return not (
        _has_sparse_identity(snapshot, subject)
        and not _policy_has_sparse_identity(snapshot, policy)
    )


def _constraint_matches(
    constraint: Mapping[str, Any], subject: Mapping[str, Any]
) -> bool:
    kind = constraint["kind"]
    if kind == "exact_identity":
        return all(
            subject.get(field) == value for field, value in constraint["values"].items()
        )
    if kind == "field_equals":
        return subject.get(constraint["field"]) == constraint["value"]
    if kind == "no_shared_expert":
        moe = subject.get("moe")
        return (
            isinstance(moe, Mapping)
            and moe.get("shared_expert_intermediate_size") is None
        )
    if kind == "compiler_contract":
        return False
    if kind == "sparse_topology":
        moe, layers = subject.get("moe"), subject.get("layers")
        if (
            not isinstance(moe, Mapping)
            or not isinstance(layers, int)
            or isinstance(layers, bool)
        ):
            return False
        step, dense_layers = moe.get("decoder_sparse_step"), moe.get("mlp_only_layers")
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or step <= 0
            or not isinstance(dense_layers, list)
        ):
            return False
        if any(
            not isinstance(index, int) or index < 0 or index >= layers
            for index in dense_layers
        ):
            return False
        dense = set(dense_layers)
        return any(
            (index + 1) % step == 0 and index not in dense for index in range(layers)
        )
    if kind == "quantization_layout":
        layout, layers = subject.get("quantization_layout"), subject.get("layers")
        if (
            not isinstance(layout, Mapping)
            or not isinstance(layers, int)
            or isinstance(layers, bool)
            or layers <= 0
        ):
            return False
        expected = [
            {
                "module_path": constraint["override_module_template"].format(
                    layer=index
                ),
                "bits": constraint["override_bits"],
                "group_size": constraint["override_group_size"],
            }
            for index in sorted(range(layers), key=str)
        ]
        return (
            layout.get("default_bits") == constraint["default_bits"]
            and layout.get("default_group_size") == constraint["default_group_size"]
            and layout.get("module_overrides") == expected
        )
    raise ValueError("Unknown model policy snapshot constraint.")


def _decision(
    snapshot: Mapping[str, Any],
    subject: Mapping[str, Any],
    *,
    kind: str,
    family: Any,
    policy: Mapping[str, Any] | None,
    paths: list[Any],
    reason_codes: list[str],
    evidence_ids: list[str],
    reason: str,
) -> dict[str, Any]:
    subject_digest = _subject_digest(subject)
    identity = {
        "schema_version": snapshot["compatibility_schema_version"],
        "subject_facts_sha256": subject_digest,
        "kind": kind,
        "family": family,
        "policy_id": policy["policy_id"] if policy else None,
        "policy_version": policy["policy_version"] if policy else None,
        "paths": paths,
        "reason_codes": reason_codes,
        "evidence_ids": evidence_ids,
    }
    return {
        "schema_version": snapshot["compatibility_schema_version"],
        "decision_id": "compat_"
        + hashlib.sha256(_canonical_json(identity, newline=False)).hexdigest()[:20],
        "subject_facts_sha256": subject_digest,
        "kind": kind,
        "family": family,
        "policy_id": identity["policy_id"],
        "policy_version": identity["policy_version"],
        "paths": paths,
        "reason_codes": reason_codes,
        "evidence_ids": evidence_ids,
        "reason": reason,
    }


def evaluate_model_policy_snapshot(
    snapshot: Mapping[str, Any], subject: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate primitive compatibility facts using only snapshot rule data."""

    validate_model_policy_snapshot(snapshot)
    if not isinstance(subject, Mapping):
        raise ValueError("Model compatibility subject must be a mapping.")
    subject = _compatibility_subject_payload(subject)
    reasons = snapshot["reasons"]
    if subject["fact_errors"]:
        policy = next(
            (
                candidate
                for candidate in snapshot["policies"]
                if _eligible_claim(snapshot, candidate, subject)
            ),
            None,
        )
        if policy is None:
            return _decision(
                snapshot,
                subject,
                kind="blocked",
                family=subject.get("family"),
                policy=None,
                paths=[],
                reason_codes=["invalid-compatibility-facts"],
                evidence_ids=[],
                reason=reasons["invalid"],
            )
        constraints = policy["constraints"]
        identity = next(
            item for item in constraints if item["kind"] == "exact_identity"
        )
        failed = None if _constraint_matches(identity, subject) else identity
        key = "invalid" if failed is None else failed["reason"]
        code = (
            "invalid-compatibility-facts" if failed is None else failed["reason_code"]
        )
        return _decision(
            snapshot,
            subject,
            kind="blocked",
            family=policy["family"],
            policy=policy,
            paths=[],
            reason_codes=[code],
            evidence_ids=list(policy["evidence_ids"]),
            reason=reasons[key],
        )
    for policy in snapshot["policies"]:
        if not _eligible_claim(snapshot, policy, subject):
            continue
        constraints = policy["constraints"]
        for constraint in constraints:
            if not _constraint_matches(constraint, subject):
                return _decision(
                    snapshot,
                    subject,
                    kind="blocked",
                    family=policy["family"],
                    policy=policy,
                    paths=[],
                    reason_codes=[constraint["reason_code"]],
                    evidence_ids=list(policy["evidence_ids"]),
                    reason=reasons[constraint["reason"]],
                )
        return _decision(
            snapshot,
            subject,
            kind="path-matched",
            family=policy["family"],
            policy=policy,
            paths=list(policy["paths"]),
            reason_codes=list(policy["matched_reason_codes"]),
            evidence_ids=list(policy["evidence_ids"]),
            reason=reasons[policy["matched_reason"]],
        )

    sparse = (
        subject.get("moe") is not None
        or _has_sparse_identity(snapshot, subject)
        or any(str(item).startswith("moe:") for item in subject.get("fact_errors", []))
    )
    if sparse:
        return _decision(
            snapshot,
            subject,
            kind="blocked",
            family=subject.get("family"),
            policy=None,
            paths=[],
            reason_codes=["unreviewed-sparse-model"],
            evidence_ids=[],
            reason=reasons["sparse"],
        )
    family = subject.get("family")
    if isinstance(family, str) and family.lower() in snapshot["dense_families"]:
        return _decision(
            snapshot,
            subject,
            kind="family-recognized",
            family=family.lower(),
            policy=None,
            paths=[],
            reason_codes=["family-recognized"],
            evidence_ids=[],
            reason=reasons["dense"],
        )
    return _decision(
        snapshot,
        subject,
        kind="unknown",
        family=family,
        policy=None,
        paths=[],
        evidence_ids=[],
        reason=reasons["unknown"],
        reason_codes=["no-policy-match"],
    )


UNREVIEWED_RUNTIME_OPERATOR_ATTESTED_REASON = (
    "This Qwen2 four-bit dense MLX-LM QLoRA shape is not the reviewed 24-layer "
    "Path Alpha footprint. The operator attested an unreviewed runtime. This is "
    "not a reviewed 7B identity. Pilot remains required."
)


def apply_operator_unreviewed_runtime_confirm(
    snapshot: Mapping[str, Any],
    subject: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    confirmed: bool,
) -> dict[str, Any]:
    """Promote a layer-count catalog miss to an attested unreviewed path.

    Resource feasibility is unchanged. Identity, four-bit, layout, and dense
    constraints still fail closed. Only ``layer-count-mismatch`` on the Qwen2
    MLX QLoRA policy may be promoted, and only after an explicit confirm.
    """

    if not confirmed:
        return dict(decision)
    if decision.get("kind") != "blocked":
        return dict(decision)
    if tuple(decision.get("reason_codes") or ()) != ("layer-count-mismatch",):
        return dict(decision)
    if decision.get("policy_id") != "model.qwen2-24l.mlx-qlora":
        return dict(decision)
    policy = next(
        (
            item
            for item in snapshot["policies"]
            if item.get("policy_id") == "model.qwen2-24l.mlx-qlora"
        ),
        None,
    )
    if policy is None:
        return dict(decision)
    subject_payload = _compatibility_subject_payload(subject)
    for constraint in policy["constraints"]:
        if (
            constraint.get("kind") == "field_equals"
            and constraint.get("field") == "layers"
        ):
            continue
        if not _constraint_matches(constraint, subject_payload):
            return dict(decision)
    return _decision(
        snapshot,
        subject_payload,
        kind="path-matched",
        family=policy["family"],
        policy=policy,
        paths=list(policy["paths"]),
        reason_codes=[
            "unreviewed-runtime-operator-attested",
            "pilot-not-yet-proven",
        ],
        evidence_ids=list(policy["evidence_ids"]),
        reason=UNREVIEWED_RUNTIME_OPERATOR_ATTESTED_REASON,
    )
