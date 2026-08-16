"""Training-policy presentation and pure instruction-SFT classification.

Surfaces existing rank/alpha/LR/completions-mask knobs as labeled priors.
Classifies dataset-size / epoch priors for task=sft without rewriting
operator facts. Classifies recorded train/validation loss series into a
presentation-only run-correction object (regularization heuristic, not
quality). Callers attach presentation at the API/CLI/UI boundary and must
never feed presentation prose into ``plan_id`` materialization.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .domain import TrainingPlan
from .plan_contract import TRAINING_POLICY_VERSION


TRAINING_POLICY_SCHEMA_VERSION = "aptus.training-policy.v1"
RUN_CORRECTION_SCHEMA_VERSION = "aptus.run-correction.v1"
RUN_CORRECTION_SOURCE = "train_loss_observations+validation_loss_observations"
DEFAULT_TRUNCATION_POLICY = (
    "completion-first; left-truncate-prompt-to-fit; refuse-empty-supervision"
)

# Loss-curve heuristics (TP4). Order: eval-rose, loss-collapsed, loss-flat, none.
_LOSS_COLLAPSE_RATIO = 0.2
_LOSS_COLLAPSE_ABS = 0.2
_LOSS_FLAT_RATIO = 0.85

INSTRUCTION_SFT_MIN_ROWS = 100
INSTRUCTION_SFT_EPOCH_CAP = 3
INSTRUCTION_SFT_SMALL_CORPUS_MAX = 299
INSTRUCTION_SFT_PARROT_EPOCHS = 10

_REASON_BELOW_SUPERVISION_PRIOR = (
    "Dataset example_count is below the instruction-SFT supervision prior of "
    "100 rows; this is not a justified domain adaptation."
)
_REASON_TOO_SMALL_FOR_LONG_TRAIN = (
    "Dataset example_count is below 100 rows; Aptus will not endorse training "
    "longer than 3 epochs on that set."
)
_REASON_EPOCH_CAP_EXCEEDED = (
    "Requested max_epochs exceeds the instruction-SFT epoch-cap prior of 3; "
    "Aptus will not rewrite the requested epoch count."
)
_REASON_PARROT_OVERTRAINING = (
    "Small instruction corpus (under 300 rows) with max_epochs >= 10 matches "
    "the parrot/sycophancy over-training prior."
)

_NON_CLAIM_QUALITY = "These knobs are not a prediction of model quality."


@dataclass(frozen=True)
class TrainingKnob:
    name: str
    value: str
    prior_kind: str
    rationale: str

    def to_primitive(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "prior_kind": self.prior_kind,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class TrainingPolicyPresentation:
    schema_version: str
    policy_version: str
    knobs: tuple[TrainingKnob, ...]
    non_claims: tuple[str, ...]

    def to_primitive(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "knobs": [item.to_primitive() for item in self.knobs],
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class TrainingPolicyVerdict:
    status: str  # "none" | "conditional" | "infeasible"
    reasons: tuple[str, ...]


def classify_instruction_sft_policy(
    *,
    example_count: int,
    max_epochs: int,
    task: str,
) -> TrainingPolicyVerdict:
    """Classify instruction-SFT dataset/epoch priors. Never rewrites max_epochs."""

    if task != "sft":
        return TrainingPolicyVerdict(status="none", reasons=())

    reasons: list[str] = []
    status = "none"

    if (
        example_count < INSTRUCTION_SFT_MIN_ROWS
        and max_epochs <= INSTRUCTION_SFT_EPOCH_CAP
    ):
        reasons.append(_REASON_BELOW_SUPERVISION_PRIOR)
        status = "conditional"
    if (
        example_count < INSTRUCTION_SFT_MIN_ROWS
        and max_epochs > INSTRUCTION_SFT_EPOCH_CAP
    ):
        reasons.append(_REASON_TOO_SMALL_FOR_LONG_TRAIN)
        status = "infeasible"
    if (
        example_count >= INSTRUCTION_SFT_MIN_ROWS
        and max_epochs > INSTRUCTION_SFT_EPOCH_CAP
    ):
        reasons.append(_REASON_EPOCH_CAP_EXCEEDED)
        if status != "infeasible":
            status = "conditional"
    if (
        INSTRUCTION_SFT_MIN_ROWS <= example_count <= INSTRUCTION_SFT_SMALL_CORPUS_MAX
        and max_epochs >= INSTRUCTION_SFT_PARROT_EPOCHS
    ):
        reasons.append(_REASON_PARROT_OVERTRAINING)
        status = "infeasible"

    return TrainingPolicyVerdict(status=status, reasons=tuple(reasons))


def build_training_policy_presentation(
    *,
    method: str,
    rank: int,
    alpha: int,
    learning_rate: float,
    target_modules: tuple[str, ...],
    example_count: int,
    max_epochs: int,
    truncation_policy: str,
    task: str = "sft",
) -> TrainingPolicyPresentation:
    """Explain knobs as labeled priors, including dataset/epoch instruction-SFT rules."""

    del method, target_modules  # reserved for later surfaces
    lr_text = f"{learning_rate:g}"
    verdict = classify_instruction_sft_policy(
        example_count=example_count,
        max_epochs=max_epochs,
        task=task,
    )
    if verdict.status != "none" and verdict.reasons:
        dataset_epoch_rationale = " ".join(verdict.reasons)
    else:
        dataset_epoch_rationale = (
            "The requested dataset size and epoch count are within the "
            "instruction-SFT prior."
        )
    knobs = (
        TrainingKnob(
            name="rank",
            value=str(rank),
            prior_kind="objective-and-token-volume-prior",
            rationale=(
                f"Adapter rank {rank} is the Aptus v0.2 objective and "
                "dataset-volume prior, not a tuned optimum."
            ),
        ),
        TrainingKnob(
            name="alpha",
            value=str(alpha),
            prior_kind="method-class-prior",
            rationale=(
                f"Adapter alpha {alpha} follows the Aptus v0.2 alpha=2*rank policy."
            ),
        ),
        TrainingKnob(
            name="learning_rate",
            value=lr_text,
            prior_kind="method-class-prior",
            rationale=(
                f"Learning rate {lr_text} is an Aptus v0.2 method-class prior, "
                "not a tuned optimum."
            ),
        ),
        TrainingKnob(
            name="completions_mask",
            value=truncation_policy,
            prior_kind="compiler-contract",
            rationale=(
                "Loss is computed on assistant/completion tokens only; prompt "
                "tokens are masked. Empty supervision is refused."
            ),
        ),
        TrainingKnob(
            name="epochs",
            value=str(max_epochs),
            prior_kind="method-class-prior",
            rationale=dataset_epoch_rationale,
        ),
        TrainingKnob(
            name="dataset_size",
            value=str(example_count),
            prior_kind="method-class-prior",
            rationale=dataset_epoch_rationale,
        ),
    )
    return TrainingPolicyPresentation(
        schema_version=TRAINING_POLICY_SCHEMA_VERSION,
        policy_version=TRAINING_POLICY_VERSION,
        knobs=knobs,
        non_claims=(_NON_CLAIM_QUALITY,),
    )


def build_training_policy_for_plan(plan: TrainingPlan) -> TrainingPolicyPresentation:
    """Build presentation knobs from a plan's recommended candidate and facts."""

    if not isinstance(plan, TrainingPlan):
        raise TypeError("build_training_policy_for_plan requires a TrainingPlan.")
    recommended = plan.recommended
    return build_training_policy_presentation(
        method=recommended.method.value,
        rank=recommended.rank,
        alpha=recommended.alpha,
        learning_rate=recommended.learning_rate,
        target_modules=recommended.target_modules,
        example_count=plan.dataset.example_count,
        max_epochs=plan.target.max_epochs,
        truncation_policy=DEFAULT_TRUNCATION_POLICY,
        task=plan.target.task,
    )


def attach_training_policy(
    payload: Mapping[str, Any],
    policy: TrainingPolicyPresentation | Mapping[str, Any],
) -> dict[str, Any]:
    """Return a shallow copy of *payload* with presentation-only ``training_policy``."""

    body = dict(payload)
    if isinstance(policy, TrainingPolicyPresentation):
        body["training_policy"] = policy.to_primitive()
    else:
        body["training_policy"] = dict(policy)
    return body


_RUN_CORRECTION_NON_CLAIMS = (
    "Training loss is not model quality.",
    "Validation split loss is not an aptus.evaluation-result.v1 decision.",
)

_RUN_CORRECTION_DISALLOWED: tuple[tuple[str, str], ...] = (
    (
        "no_automl",
        "Do not start a hyperparameter search.",
    ),
    (
        "no_quality_from_loss",
        "Do not treat this signal as model quality or an M8 eval decision.",
    ),
    (
        "no_weight_decay_as_sycophancy_fix",
        "Do not add weight decay as a sycophancy cure.",
    ),
)


@dataclass(frozen=True)
class RunPlanHint:
    fact: str
    direction: str
    why: str

    def to_primitive(self) -> dict[str, object]:
        return {
            "fact": self.fact,
            "direction": self.direction,
            "why": self.why,
        }


@dataclass(frozen=True)
class RunDisallowedSuggestion:
    code: str
    message: str

    def to_primitive(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class RunOperatorNextStep:
    action: str
    label: str

    def to_primitive(self) -> dict[str, object]:
        return {"action": self.action, "label": self.label}


@dataclass(frozen=True)
class RunCorrection:
    kind: str
    summary: str
    next_plan_hints: tuple[RunPlanHint, ...]
    operator_next_step: RunOperatorNextStep
    schema_version: str = RUN_CORRECTION_SCHEMA_VERSION
    source: str = RUN_CORRECTION_SOURCE
    disallowed_suggestions: tuple[RunDisallowedSuggestion, ...] = ()
    non_claims: tuple[str, ...] = _RUN_CORRECTION_NON_CLAIMS

    def to_primitive(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "summary": self.summary,
            "source": self.source,
            "next_plan_hints": [item.to_primitive() for item in self.next_plan_hints],
            "disallowed_suggestions": [
                item.to_primitive() for item in self.disallowed_suggestions
            ],
            "operator_next_step": self.operator_next_step.to_primitive(),
            "non_claims": list(self.non_claims),
        }


def _finite_series(values: Sequence[Any] | None) -> list[float] | None:
    """Return finite floats, or None when the input is missing/unusable."""

    if values is None:
        return None
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    series: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        series.append(number)
    return series


def _always_disallowed() -> tuple[RunDisallowedSuggestion, ...]:
    return tuple(
        RunDisallowedSuggestion(code=code, message=message)
        for code, message in _RUN_CORRECTION_DISALLOWED
    )


def classify_run_loss_signal(
    train_loss_observations: Sequence[Any] | None,
    validation_loss_observations: Sequence[Any] | None = None,
) -> RunCorrection:
    """Classify a recorded loss series into one presentation-only run correction.

    Detection order (first match wins): eval-rose, loss-collapsed, loss-flat,
    else none. Never rewrites plans, stops training, or emits evaluation results.
    """

    train = _finite_series(train_loss_observations)
    validation = _finite_series(validation_loss_observations)
    disallowed = _always_disallowed()

    if (
        train is not None
        and validation is not None
        and len(train) >= 2
        and len(validation) >= 2
    ):
        t0, t_n = train[0], train[-1]
        v0, v_n = validation[0], validation[-1]
        if t_n < t0 and v_n > v0:
            return RunCorrection(
                kind="eval-rose",
                summary=(
                    "Train loss fell while validation loss rose; consider fewer "
                    "epochs on the next plan. This is a regularization heuristic, "
                    "not an evaluation pass or fail."
                ),
                next_plan_hints=(
                    RunPlanHint(
                        fact="target.max_epochs",
                        direction="decrease",
                        why=(
                            "Train loss fell while validation loss rose; next plan "
                            "should use fewer epochs. This is not an evaluation "
                            "pass or fail."
                        ),
                    ),
                ),
                disallowed_suggestions=disallowed,
                operator_next_step=RunOperatorNextStep(
                    action="replan-with-fact-hints",
                    label="Replan with fewer epochs",
                ),
            )

    if train is not None and len(train) >= 2:
        t0, t_n = train[0], train[-1]
        if t_n < t0 * _LOSS_COLLAPSE_RATIO and t_n < _LOSS_COLLAPSE_ABS:
            return RunCorrection(
                kind="loss-collapsed",
                summary=(
                    "Train loss collapsed relative to the start; consider fewer "
                    "epochs on the next plan (or alpha equal to rank)."
                ),
                next_plan_hints=(
                    RunPlanHint(
                        fact="target.max_epochs",
                        direction="decrease",
                        why=(
                            "Train loss collapsed; decrease max_epochs toward the "
                            "instruction-SFT epoch cap on the next plan."
                        ),
                    ),
                ),
                disallowed_suggestions=disallowed,
                operator_next_step=RunOperatorNextStep(
                    action="replan-with-fact-hints",
                    label="Replan with fewer epochs",
                ),
            )
        if t_n > t0 * _LOSS_FLAT_RATIO:
            return RunCorrection(
                kind="loss-flat",
                summary=(
                    "Train loss stayed relatively flat; consider more epochs "
                    "(only up to 3) or review rank on the next plan."
                ),
                next_plan_hints=(
                    RunPlanHint(
                        fact="target.max_epochs",
                        direction="increase",
                        why=(
                            "Train loss stayed flat; increase max_epochs only up "
                            "to 3, or review rank. Never add weight decay as a fix."
                        ),
                    ),
                ),
                disallowed_suggestions=disallowed,
                operator_next_step=RunOperatorNextStep(
                    action="replan-with-fact-hints",
                    label="Replan with more epochs or review rank",
                ),
            )

    return RunCorrection(
        kind="none",
        summary="No training-signal correction from the recorded loss series.",
        next_plan_hints=(),
        disallowed_suggestions=disallowed,
        operator_next_step=RunOperatorNextStep(
            action="none",
            label="No next-plan change from this signal",
        ),
    )


def build_run_correction_from_metrics(
    metrics: Mapping[str, Any] | None,
) -> RunCorrection | None:
    """Build run-correction from a metrics mapping, or None when unreadable."""

    if not isinstance(metrics, Mapping):
        return None
    return classify_run_loss_signal(
        metrics.get("train_loss_observations"),
        metrics.get("validation_loss_observations"),
    )


def build_run_correction_from_metrics_path(path: Path | str) -> RunCorrection | None:
    """Read metrics.json and classify; return None if the file is missing/unusable."""

    metrics_path = Path(path)
    if not metrics_path.is_file():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return build_run_correction_from_metrics(payload)


def attach_run_correction(
    payload: Mapping[str, Any],
    correction: RunCorrection | Mapping[str, Any],
) -> dict[str, Any]:
    """Return a shallow copy of *payload* with presentation-only ``run_correction``."""

    body = dict(payload)
    if isinstance(correction, RunCorrection):
        body["run_correction"] = correction.to_primitive()
    else:
        body["run_correction"] = dict(correction)
    return body
