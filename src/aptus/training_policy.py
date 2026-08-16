"""Training-policy presentation and pure instruction-SFT classification.

Surfaces existing rank/alpha/LR/completions-mask knobs as labeled priors.
Classifies dataset-size / epoch priors for task=sft without rewriting
operator facts. Callers attach presentation at the API/CLI/UI boundary and
must never feed presentation prose into ``plan_id`` materialization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .domain import TrainingPlan


TRAINING_POLICY_SCHEMA_VERSION = "aptus.training-policy.v1"
TRAINING_POLICY_VERSION = "aptus-training-policy-v1"
DEFAULT_TRUNCATION_POLICY = (
    "completion-first; left-truncate-prompt-to-fit; refuse-empty-supervision"
)

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

    if example_count < INSTRUCTION_SFT_MIN_ROWS and max_epochs <= INSTRUCTION_SFT_EPOCH_CAP:
        reasons.append(_REASON_BELOW_SUPERVISION_PRIOR)
        status = "conditional"
    if example_count < INSTRUCTION_SFT_MIN_ROWS and max_epochs > INSTRUCTION_SFT_EPOCH_CAP:
        reasons.append(_REASON_TOO_SMALL_FOR_LONG_TRAIN)
        status = "infeasible"
    if example_count >= INSTRUCTION_SFT_MIN_ROWS and max_epochs > INSTRUCTION_SFT_EPOCH_CAP:
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
) -> TrainingPolicyPresentation:
    """Explain existing knobs as labeled priors. Does not classify status."""

    del method, target_modules, example_count, max_epochs  # reserved for later surfaces
    lr_text = f"{learning_rate:g}"
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
