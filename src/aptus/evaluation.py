"""Optional post-train evaluation contract and scoring.

``aptus.evaluation-contract.v1`` and ``aptus.evaluation-result.v1`` keep
training completion distinct from a task-metric decision. The v1 metric is
deterministic exact match against operator-supplied predictions. Callers may
attach a contract onto a plan payload for presentation; it must never enter
``plan_id`` material.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .plan_contract import sha256_file


CONTRACT_SCHEMA_VERSION = "aptus.evaluation-contract.v1"
RESULT_SCHEMA_VERSION = "aptus.evaluation-result.v1"
EXACT_MATCH_IMPLEMENTATION = "aptus.exact-match.v1"
SUPPORTED_METRICS = frozenset({"exact_match"})
SUPPORTED_GOLD_FIELDS = frozenset({"completion", "output", "gold"})
SUPPORTED_PREDICTION_FIELDS = frozenset({"prediction", "output", "completion"})
SUPPORTED_EXPORT_KINDS = frozenset({"adapter", "final-export"})
DEFAULT_NON_CLAIMS = (
    "Training finished is not an evaluation pass.",
    "Train loss and split evaluation loss are not this decision.",
    "This result is not general model quality, safety, or human preference.",
    "This result is not a leaderboard ranking.",
)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class EvaluationNormalization:
    strip: bool = True
    collapse_whitespace: bool = True
    casefold: bool = False

    def to_primitive(self) -> dict[str, object]:
        return {
            "strip": self.strip,
            "collapse_whitespace": self.collapse_whitespace,
            "casefold": self.casefold,
        }

    def apply(self, text: str) -> str:
        value = text
        if self.strip:
            value = value.strip()
        if self.collapse_whitespace:
            value = _WHITESPACE.sub(" ", value)
            if self.strip:
                value = value.strip()
        if self.casefold:
            value = value.casefold()
        return value


@dataclass(frozen=True)
class EvaluationDatasetBinding:
    sha256: str
    format: str
    gold_field: str
    row_count: int
    id_field: str | None = None
    path: str | None = None

    def to_primitive(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "sha256": self.sha256,
            "format": self.format,
            "gold_field": self.gold_field,
            "row_count": self.row_count,
            "id_field": self.id_field,
        }
        if self.path is not None:
            payload["path"] = self.path
        return payload

    def identity(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "format": self.format,
            "gold_field": self.gold_field,
            "row_count": self.row_count,
            "id_field": self.id_field,
        }


@dataclass(frozen=True)
class EvaluationMetric:
    name: str = "exact_match"
    direction: str = "higher_is_better"
    implementation_version: str = EXACT_MATCH_IMPLEMENTATION
    normalization: EvaluationNormalization = EvaluationNormalization()

    def to_primitive(self) -> dict[str, object]:
        return {
            "name": self.name,
            "direction": self.direction,
            "implementation_version": self.implementation_version,
            "normalization": self.normalization.to_primitive(),
        }


@dataclass(frozen=True)
class EvaluationThreshold:
    minimum: float
    comparison: str = "gte"

    def to_primitive(self) -> dict[str, object]:
        return {"minimum": self.minimum, "comparison": self.comparison}


@dataclass(frozen=True)
class EvaluationArtifactBinding:
    plan_id: str | None = None
    candidate_id: str | None = None
    job_id: str | None = None
    export_digest: str | None = None
    export_kind: str | None = None

    def to_primitive(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "candidate_id": self.candidate_id,
            "job_id": self.job_id,
            "export_digest": self.export_digest,
            "export_kind": self.export_kind,
        }


@dataclass(frozen=True)
class EvaluationContract:
    claim: str
    dataset: EvaluationDatasetBinding
    metric: EvaluationMetric
    threshold: EvaluationThreshold
    artifact_binding: EvaluationArtifactBinding
    non_claims: tuple[str, ...] = DEFAULT_NON_CLAIMS
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def to_primitive(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "claim": self.claim,
            "dataset": self.dataset.to_primitive(),
            "metric": self.metric.to_primitive(),
            "threshold": self.threshold.to_primitive(),
            "artifact_binding": self.artifact_binding.to_primitive(),
            "non_claims": list(self.non_claims),
        }

    def digest(self) -> str:
        return _sha256_json(
            {
                "schema_version": self.schema_version,
                "claim": self.claim,
                "dataset": self.dataset.identity(),
                "metric": self.metric.to_primitive(),
                "threshold": self.threshold.to_primitive(),
                "artifact_binding": self.artifact_binding.to_primitive(),
                "non_claims": list(self.non_claims),
            }
        )


@dataclass(frozen=True)
class EvaluationResult:
    contract_sha256: str
    gold_sha256: str
    predictions_sha256: str
    artifact_binding: EvaluationArtifactBinding
    metric: str
    score: float | None
    threshold: float
    n_gold: int
    n_predictions: int
    n_scored: int
    n_missing: int
    n_extra: int
    decision: str
    decision_reasons: tuple[str, ...]
    non_claims: tuple[str, ...] = DEFAULT_NON_CLAIMS
    evaluated_at: str = ""
    schema_version: str = RESULT_SCHEMA_VERSION

    def to_primitive(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_sha256": self.contract_sha256,
            "gold_sha256": self.gold_sha256,
            "predictions_sha256": self.predictions_sha256,
            "artifact_binding": self.artifact_binding.to_primitive(),
            "metric": self.metric,
            "score": self.score,
            "threshold": self.threshold,
            "n_gold": self.n_gold,
            "n_predictions": self.n_predictions,
            "n_scored": self.n_scored,
            "n_missing": self.n_missing,
            "n_extra": self.n_extra,
            "decision": self.decision,
            "decision_reasons": list(self.decision_reasons),
            "non_claims": list(self.non_claims),
            "evaluated_at": self.evaluated_at,
        }


def build_evaluation_contract(
    *,
    dataset_path: Path,
    claim: str,
    threshold: float,
    metric: str = "exact_match",
    gold_field: str = "completion",
    id_field: str | None = "id",
    casefold: bool = False,
    strip: bool = True,
    collapse_whitespace: bool = True,
    plan_id: str | None = None,
    candidate_id: str | None = None,
    job_id: str | None = None,
    export_digest: str | None = None,
    export_kind: str | None = None,
) -> EvaluationContract:
    path = Path(dataset_path)
    cleaned_claim = _require_text(claim, "claim")
    if metric not in SUPPORTED_METRICS:
        raise ValueError("v1 evaluation supports only the exact_match metric.")
    if not _finite_unit_interval(threshold):
        raise ValueError("threshold must be a finite number in [0, 1].")
    if gold_field not in SUPPORTED_GOLD_FIELDS:
        raise ValueError(
            "gold_field must be one of: "
            + ", ".join(sorted(SUPPORTED_GOLD_FIELDS))
            + "."
        )
    cleaned_id_field = _optional_text(id_field)
    rows = _load_jsonl_objects(path, "Gold dataset")
    if not rows:
        raise ValueError("Gold dataset must contain at least one JSONL object.")
    if cleaned_id_field == "id" and all("id" not in row for row in rows):
        cleaned_id_field = None
    if export_kind is not None and export_kind not in SUPPORTED_EXPORT_KINDS:
        raise ValueError("export_kind must be adapter or final-export.")
    if export_digest is not None and not _valid_sha256(export_digest):
        raise ValueError("export_digest must be a 64-character hexadecimal SHA-256.")
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if _row_text(row, gold_field) is None:
            raise ValueError(f"Gold row {index} is missing field {gold_field!r}.")
        if cleaned_id_field is not None:
            row_id = _row_text(row, cleaned_id_field)
            if row_id is None:
                raise ValueError(
                    f"Gold row {index} is missing identity field {cleaned_id_field!r}."
                )
            if row_id in seen_ids:
                raise ValueError(f"Gold identity {row_id!r} is repeated.")
            seen_ids.add(row_id)
    return EvaluationContract(
        claim=cleaned_claim,
        dataset=EvaluationDatasetBinding(
            sha256=sha256_file(path),
            format="jsonl",
            gold_field=gold_field,
            row_count=len(rows),
            id_field=cleaned_id_field,
            path=str(path),
        ),
        metric=EvaluationMetric(
            name=metric,
            normalization=EvaluationNormalization(
                strip=strip,
                collapse_whitespace=collapse_whitespace,
                casefold=casefold,
            ),
        ),
        threshold=EvaluationThreshold(minimum=float(threshold)),
        artifact_binding=EvaluationArtifactBinding(
            plan_id=_optional_text(plan_id),
            candidate_id=_optional_text(candidate_id),
            job_id=_optional_text(job_id),
            export_digest=_optional_text(export_digest),
            export_kind=_optional_text(export_kind),
        ),
    )


def evaluation_contract_from_primitive(
    payload: Mapping[str, Any],
) -> EvaluationContract:
    if not isinstance(payload, Mapping):
        raise ValueError("Evaluation contract must be a JSON object.")
    if payload.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONTRACT_SCHEMA_VERSION}.")
    dataset = payload.get("dataset")
    metric = payload.get("metric")
    threshold = payload.get("threshold")
    binding = payload.get("artifact_binding")
    if not isinstance(dataset, Mapping):
        raise ValueError("Evaluation contract dataset must be an object.")
    if not isinstance(metric, Mapping):
        raise ValueError("Evaluation contract metric must be an object.")
    if not isinstance(threshold, Mapping):
        raise ValueError("Evaluation contract threshold must be an object.")
    if binding is None:
        binding = {}
    if not isinstance(binding, Mapping):
        raise ValueError("Evaluation contract artifact_binding must be an object.")
    normalization = metric.get("normalization") or {}
    if not isinstance(normalization, Mapping):
        raise ValueError("Evaluation contract normalization must be an object.")
    name = metric.get("name")
    if name not in SUPPORTED_METRICS:
        raise ValueError("v1 evaluation supports only the exact_match metric.")
    minimum = threshold.get("minimum")
    if not _finite_unit_interval(minimum):
        raise ValueError("threshold must be a finite number in [0, 1].")
    if dataset.get("format") not in {None, "jsonl"}:
        raise ValueError("dataset.format must be jsonl.")
    if metric.get("direction") not in {None, "higher_is_better"}:
        raise ValueError("metric.direction must be higher_is_better.")
    if threshold.get("comparison") not in {None, "gte"}:
        raise ValueError("threshold.comparison must be gte.")
    implementation = metric.get("implementation_version") or EXACT_MATCH_IMPLEMENTATION
    if implementation != EXACT_MATCH_IMPLEMENTATION:
        raise ValueError(
            f"metric.implementation_version must be {EXACT_MATCH_IMPLEMENTATION}."
        )
    gold_field = dataset.get("gold_field")
    if gold_field not in SUPPORTED_GOLD_FIELDS:
        raise ValueError("gold_field must be completion, output, or gold.")
    digest = dataset.get("sha256")
    if not _valid_sha256(digest):
        raise ValueError("dataset.sha256 must be a 64-character hexadecimal SHA-256.")
    row_count = dataset.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 1:
        raise ValueError("dataset.row_count must be a positive integer.")
    export_digest = binding.get("export_digest")
    if export_digest is not None and not _valid_sha256(export_digest):
        raise ValueError("export_digest must be a 64-character hexadecimal SHA-256.")
    export_kind = binding.get("export_kind")
    if export_kind is not None and export_kind not in SUPPORTED_EXPORT_KINDS:
        raise ValueError("export_kind must be adapter or final-export.")
    non_claims = payload.get("non_claims") or DEFAULT_NON_CLAIMS
    if not isinstance(non_claims, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in non_claims
    ):
        raise ValueError("non_claims must be a list of non-empty strings.")
    missing_non_claims = [item for item in DEFAULT_NON_CLAIMS if item not in non_claims]
    if missing_non_claims:
        raise ValueError("Evaluation contract is missing required non_claims.")
    return EvaluationContract(
        claim=_require_text(payload.get("claim"), "claim"),
        dataset=EvaluationDatasetBinding(
            sha256=digest,
            format="jsonl",
            gold_field=gold_field,
            row_count=row_count,
            id_field=_optional_text(dataset.get("id_field")),
            path=_optional_text(dataset.get("path")),
        ),
        metric=EvaluationMetric(
            name=name,
            direction="higher_is_better",
            implementation_version=EXACT_MATCH_IMPLEMENTATION,
            normalization=EvaluationNormalization(
                strip=bool(normalization.get("strip", True)),
                collapse_whitespace=bool(
                    normalization.get("collapse_whitespace", True)
                ),
                casefold=bool(normalization.get("casefold", False)),
            ),
        ),
        threshold=EvaluationThreshold(minimum=float(minimum)),
        artifact_binding=EvaluationArtifactBinding(
            plan_id=_optional_text(binding.get("plan_id")),
            candidate_id=_optional_text(binding.get("candidate_id")),
            job_id=_optional_text(binding.get("job_id")),
            export_digest=_optional_text(export_digest),
            export_kind=_optional_text(export_kind),
        ),
        non_claims=tuple(str(item) for item in non_claims),
    )


def attach_evaluation_contract(
    payload: Mapping[str, Any],
    contract: EvaluationContract | Mapping[str, Any],
) -> dict[str, Any]:
    """Return a shallow copy with presentation-only ``evaluation_contract``."""

    body = dict(payload)
    if isinstance(contract, EvaluationContract):
        body["evaluation_contract"] = contract.to_primitive()
    else:
        body["evaluation_contract"] = dict(contract)
    return body


def evaluate_predictions(
    contract: EvaluationContract,
    gold_path: Path,
    predictions_path: Path,
    *,
    expected_export_digest: str | None = None,
) -> EvaluationResult:
    gold = Path(gold_path)
    predictions = Path(predictions_path)
    gold_digest = sha256_file(gold)
    predictions_digest = sha256_file(predictions)
    gold_rows = _load_jsonl_objects(gold, "Gold dataset")
    prediction_rows = _load_jsonl_objects(predictions, "Predictions")
    reasons: list[str] = []
    if contract.metric.name not in SUPPORTED_METRICS:
        reasons.append("unsupported metric")
    if gold_digest != contract.dataset.sha256:
        reasons.append("gold digest does not match the evaluation contract")
    if len(gold_rows) != contract.dataset.row_count:
        reasons.append("gold row count does not match the evaluation contract")
    if not gold_rows:
        reasons.append("gold dataset is empty")
    expected_digest = _optional_text(expected_export_digest)
    bound_digest = contract.artifact_binding.export_digest
    if bound_digest is not None and expected_digest is None:
        reasons.append("export digest is required because the contract binds one")
    elif expected_digest is not None and bound_digest is None:
        reasons.append("export digest was supplied but the contract does not bind one")
    elif (
        expected_digest is not None
        and bound_digest is not None
        and expected_digest != bound_digest
    ):
        reasons.append("export digest does not match the evaluation contract")
    gold_index = _index_rows(
        gold_rows,
        id_field=contract.dataset.id_field,
        text_fields=(contract.dataset.gold_field,),
        label="gold",
    )
    prediction_index = _index_rows(
        prediction_rows,
        id_field=contract.dataset.id_field,
        text_fields=tuple(SUPPORTED_PREDICTION_FIELDS),
        label="predictions",
    )
    missing = sorted(set(gold_index) - set(prediction_index))
    extra = sorted(set(prediction_index) - set(gold_index))
    if missing:
        reasons.append(f"missing {len(missing)} prediction(s)")
    if extra:
        reasons.append(f"extra {len(extra)} prediction(s)")
    n_scored = 0
    matches = 0
    if not reasons:
        for key, gold_text in gold_index.items():
            predicted = prediction_index[key]
            n_scored += 1
            if contract.metric.normalization.apply(
                gold_text
            ) == contract.metric.normalization.apply(predicted):
                matches += 1
    score = (matches / n_scored) if n_scored else None
    if reasons:
        decision = "abstain"
        score = None
        n_scored = 0
    elif score is not None and score >= contract.threshold.minimum:
        decision = "pass"
    else:
        decision = "fail"
        if score is not None:
            reasons.append(
                f"exact_match {score:.6f} is below threshold {contract.threshold.minimum:.6f}"
            )
    return EvaluationResult(
        contract_sha256=contract.digest(),
        gold_sha256=gold_digest,
        predictions_sha256=predictions_digest,
        artifact_binding=contract.artifact_binding,
        metric=contract.metric.name,
        score=score,
        threshold=contract.threshold.minimum,
        n_gold=len(gold_rows),
        n_predictions=len(prediction_rows),
        n_scored=n_scored,
        n_missing=len(missing),
        n_extra=len(extra),
        decision=decision,
        decision_reasons=tuple(reasons),
        non_claims=contract.non_claims,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )


def _index_rows(
    rows: list[dict[str, Any]],
    *,
    id_field: str | None,
    text_fields: tuple[str, ...],
    label: str,
) -> dict[str, str]:
    indexed: dict[str, str] = {}
    for index, row in enumerate(rows):
        key = _row_text(row, id_field) if id_field else str(index)
        if key is None:
            raise ValueError(
                f"{label} row {index} is missing identity field {id_field!r}."
            )
        if key in indexed:
            raise ValueError(f"{label} identity {key!r} is repeated.")
        text = None
        for field in text_fields:
            text = _row_text(row, field)
            if text is not None:
                break
        if text is None:
            raise ValueError(
                f"{label} row {index} is missing one of: "
                + ", ".join(text_fields)
                + "."
            )
        indexed[key] = text
    return indexed


def _load_jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"{label} is unreadable: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except (RecursionError, ValueError) as error:
            raise ValueError(
                f"{label} line {line_number} is not valid JSON."
            ) from error
        if not isinstance(parsed, dict):
            raise ValueError(f"{label} line {line_number} must be a JSON object.")
        rows.append(parsed)
    return rows


def _row_text(row: Mapping[str, Any], field: str | None) -> str | None:
    if not field:
        return None
    value = row.get(field)
    if isinstance(value, str):
        return value
    return None


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return " ".join(value.split())


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _finite_unit_interval(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 <= value <= 1
    )


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
