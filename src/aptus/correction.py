"""Plan-level correction summaries (presentation only; not plan identity).

Builds ``aptus.plan-correction.v1`` objects from planner outcomes using the M2
refusal catalog. Callers attach the result at the API/CLI/UI boundary and must
never feed it into ``plan_id`` materialization.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Mapping

from .domain import CandidatePlan, CandidateStatus, Objective, TrainingPlan
from .refusal import RefusalGuidance, guide_rejection_reason


CORRECTION_SCHEMA_VERSION = "aptus.plan-correction.v1"
_MAX_REASON_CODES = 5
_MAX_FACT_HINTS = 5

_DISALLOWED_BY_CODE: dict[str, tuple[str, str]] = {
    "full_fsdp": (
        "no_fsdp",
        "Do not enable full FSDP; unsupported in v0.2.",
    ),
    "quantized_fsdp": (
        "no_fsdp",
        "Do not enable FSDP with quantized methods; unsupported in v0.2.",
    ),
    "multi_gpu_on_single": (
        "no_multi_gpu",
        "Do not enable DDP/FSDP on a single-device inventory.",
    ),
    "mlx_full": (
        "no_mlx_full",
        "Do not switch to full fine-tuning on MLX; no full compiler is registered.",
    ),
    "packing_unsupported": (
        "no_packing",
        "Do not enable sequence packing; closed in v0.2.",
    ),
}

_ALWAYS_NO_PATH_DISALLOWED: tuple[tuple[str, str], ...] = (
    (
        "no_new_method",
        "Do not invent a training method outside Full/LoRA/int8-LoRA/QLoRA.",
    ),
    (
        "no_fsdp",
        "Do not enable FSDP as a workaround; unsupported in v0.2.",
    ),
)


@dataclass(frozen=True)
class FactHint:
    fact: str
    direction: str
    why: str
    source_reason_codes: tuple[str, ...]

    def to_primitive(self) -> dict[str, object]:
        return {
            "fact": self.fact,
            "direction": self.direction,
            "why": self.why,
            "source_reason_codes": list(self.source_reason_codes),
        }


@dataclass(frozen=True)
class DisallowedSuggestion:
    code: str
    message: str

    def to_primitive(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class OperatorNextStep:
    action: str
    label: str

    def to_primitive(self) -> dict[str, object]:
        return {"action": self.action, "label": self.label}


@dataclass(frozen=True)
class PlanCorrection:
    kind: str
    summary: str
    primary_reason_codes: tuple[str, ...]
    recommended_candidate_id: str | None
    recommended_status: str | None
    pilot_required: bool
    ranking_objective: str | None
    fact_hints: tuple[FactHint, ...]
    disallowed_suggestions: tuple[DisallowedSuggestion, ...]
    operator_next_step: OperatorNextStep
    schema_version: str = CORRECTION_SCHEMA_VERSION

    def to_primitive(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "summary": self.summary,
            "primary_reason_codes": list(self.primary_reason_codes),
            "recommended_candidate_id": self.recommended_candidate_id,
            "recommended_status": self.recommended_status,
            "pilot_required": self.pilot_required,
            "ranking_objective": self.ranking_objective,
            "fact_hints": [item.to_primitive() for item in self.fact_hints],
            "disallowed_suggestions": [
                item.to_primitive() for item in self.disallowed_suggestions
            ],
            "operator_next_step": self.operator_next_step.to_primitive(),
        }


def build_plan_correction(plan: TrainingPlan) -> PlanCorrection:
    """Build correction for a successful plan (feasible or conditional recommend)."""

    if not isinstance(plan, TrainingPlan):
        raise TypeError("build_plan_correction requires a TrainingPlan.")
    recommended = plan.recommended
    if not recommended.feasible or recommended.status not in {
        CandidateStatus.FEASIBLE,
        CandidateStatus.CONDITIONAL,
    }:
        raise ValueError(
            "Plan correction for a successful plan requires a viable recommended candidate."
        )
    pilot_required = _candidate_pilot_required(recommended)
    status = recommended.status.value
    objective = plan.target.objective.value
    if recommended.status is CandidateStatus.CONDITIONAL:
        codes = _primary_codes_from_candidates((recommended,))
        if not codes:
            codes = ("conditional_pilot_required",)
        summary = (
            f"Use {recommended.method.value} {recommended.distribution.value}; "
            "pilot is required before full train."
        )
        next_step = OperatorNextStep(
            action="confirm-pilot-then-train",
            label="Run pilot, then confirm full train",
        )
    else:
        codes = ()
        summary = (
            f"Use {recommended.method.value} {recommended.distribution.value} "
            f"under the {objective} objective among viable candidates."
        )
        if pilot_required:
            next_step = OperatorNextStep(
                action="confirm-pilot-then-train",
                label="Run pilot, then confirm full train",
            )
        else:
            next_step = OperatorNextStep(
                action="compile-recommended",
                label="Compile recommended bundle",
            )
    disallowed = _disallowed_from_codes(
        _codes_from_candidates(plan.candidates),
        include_no_path_defaults=False,
    )
    # Always discourage inventing multi-GPU when inventory is single-device.
    if plan.hardware.devices and len(plan.hardware.devices) < 2:
        disallowed = _merge_disallowed(
            disallowed,
            (
                DisallowedSuggestion(
                    code="no_multi_gpu",
                    message="Do not enable DDP/FSDP on a single-device inventory.",
                ),
            ),
        )
    return PlanCorrection(
        kind="select-candidate",
        summary=_clip_summary(summary),
        primary_reason_codes=codes,
        recommended_candidate_id=recommended.candidate_id,
        recommended_status=status,
        pilot_required=pilot_required,
        ranking_objective=objective,
        fact_hints=(),
        disallowed_suggestions=disallowed,
        operator_next_step=next_step,
    )


def build_no_path_correction(
    candidates: Sequence[CandidatePlan],
    *,
    ranking_objective: str | Objective | None = None,
) -> PlanCorrection:
    """Build correction when the planner returns no viable candidates."""

    if not candidates:
        raise ValueError("No-path correction requires at least one candidate.")
    if any(item.feasible for item in candidates):
        raise ValueError("No-path correction cannot include viable candidates.")
    objective: str | None
    if isinstance(ranking_objective, Objective):
        objective = ranking_objective.value
    elif ranking_objective is None:
        objective = None
    else:
        objective = str(ranking_objective)
    codes = _primary_codes_from_candidates(candidates)
    hints = _fact_hints_from_candidates(candidates)
    if codes:
        summary = (
            "No supported training path fits these facts; "
            "change the listed facts and replan."
        )
    else:
        summary = "No supported training path fits these facts."
    if hints:
        # Prefer the first hint fact in the summary when space allows.
        lead = hints[0].fact
        summary = (
            f"No supported training path fits these facts; start by adjusting {lead}."
        )
    return PlanCorrection(
        kind="no-path",
        summary=_clip_summary(summary),
        primary_reason_codes=codes,
        recommended_candidate_id=None,
        recommended_status=None,
        pilot_required=False,
        ranking_objective=objective,
        fact_hints=hints,
        disallowed_suggestions=_disallowed_from_codes(
            _codes_from_candidates(candidates),
            include_no_path_defaults=True,
        ),
        operator_next_step=OperatorNextStep(
            action="change-facts",
            label="Change facts and replan",
        ),
    )


def attach_correction(
    payload: Mapping[str, Any],
    correction: PlanCorrection | Mapping[str, Any],
) -> dict[str, Any]:
    """Return a shallow copy of *payload* with presentation-only ``correction``."""

    body = dict(payload)
    if isinstance(correction, PlanCorrection):
        body["correction"] = correction.to_primitive()
    else:
        body["correction"] = dict(correction)
    return body


def _candidate_pilot_required(candidate: CandidatePlan) -> bool:
    if candidate.status is CandidateStatus.CONDITIONAL:
        return True
    contract = candidate.runtime_contract
    if contract is not None and contract.evidence_requirement.value == "pilot-required":
        return True
    confidence = (candidate.confidence or "").lower()
    return "pilot" in confidence


def _codes_from_candidates(
    candidates: Sequence[CandidatePlan],
) -> list[str]:
    codes: list[str] = []
    for candidate in candidates:
        for reason in candidate.rejection_reasons:
            guided = guide_rejection_reason(reason)
            codes.append(guided.reason_code)
    return codes


def _primary_codes_from_candidates(
    candidates: Sequence[CandidatePlan],
) -> tuple[str, ...]:
    counter = Counter(_codes_from_candidates(candidates))
    # Frequency desc, then stable code order.
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return tuple(code for code, _count in ordered[:_MAX_REASON_CODES])


def _fact_hints_from_candidates(
    candidates: Sequence[CandidatePlan],
) -> tuple[FactHint, ...]:
    # fact -> (direction, why fragments, codes)
    bucket: dict[str, tuple[str, list[str], list[str]]] = {}
    for candidate in candidates:
        for reason in candidate.rejection_reasons:
            guided = guide_rejection_reason(reason)
            if not guided.operator_actionable or guided.none_in_catalog:
                continue
            for fact in guided.changeable_facts:
                direction = _direction_for_fact(fact, guided)
                why = f"{guided.title}: {guided.explanation}"
                if fact not in bucket:
                    bucket[fact] = (direction, [why], [guided.reason_code])
                else:
                    existing_direction, whys, codes = bucket[fact]
                    if why not in whys:
                        whys.append(why)
                    if guided.reason_code not in codes:
                        codes.append(guided.reason_code)
                    # Prefer decrease/increase over review when mixed.
                    if existing_direction == "review" and direction != "review":
                        existing_direction = direction
                    bucket[fact] = (existing_direction, whys, codes)
    hints: list[FactHint] = []
    for fact, (direction, whys, codes) in bucket.items():
        hints.append(
            FactHint(
                fact=fact,
                direction=direction,
                why=_clip_summary("; ".join(whys[:2])),
                source_reason_codes=tuple(codes),
            )
        )
    # Prefer memory-related facts first.
    priority = (
        "sequence_length",
        "effective_batch",
        "micro_batch",
        "vram",
        "host_ram",
        "disk",
    )

    def sort_key(hint: FactHint) -> tuple[int, str]:
        for index, token in enumerate(priority):
            if token in hint.fact:
                return (index, hint.fact)
        return (len(priority), hint.fact)

    hints.sort(key=sort_key)
    return tuple(hints[:_MAX_FACT_HINTS])


def _direction_for_fact(fact: str, guided: RefusalGuidance) -> str:
    lowered = fact.lower()
    code = guided.reason_code
    if "packing" in lowered:
        return "set"
    if any(
        token in lowered
        for token in ("sequence_length", "effective_batch", "micro_batch", "reserve")
    ):
        if code in {
            "infeasible_memory",
            "conditional_upper_envelope",
            "sequence_length_exceeds_context",
            "host_ram_infeasible",
            "disk_infeasible",
        }:
            return "decrease"
        return "review"
    if any(
        token in lowered
        for token in ("vram", "free memory", "host_ram", "disk", "hardware.devices")
    ):
        if code in {
            "infeasible_memory",
            "conditional_upper_envelope",
            "host_ram_infeasible",
            "disk_infeasible",
            "multi_gpu_on_single",
            "no_compute_device",
        }:
            return "increase"
        return "review"
    return "review"


def _disallowed_from_codes(
    codes: Iterable[str],
    *,
    include_no_path_defaults: bool,
) -> tuple[DisallowedSuggestion, ...]:
    items: list[DisallowedSuggestion] = []
    seen: set[str] = set()
    if include_no_path_defaults:
        for code, message in _ALWAYS_NO_PATH_DISALLOWED:
            if code not in seen:
                seen.add(code)
                items.append(DisallowedSuggestion(code=code, message=message))
    for reason_code in codes:
        mapped = _DISALLOWED_BY_CODE.get(reason_code)
        if mapped is None:
            continue
        code, message = mapped
        if code not in seen:
            seen.add(code)
            items.append(DisallowedSuggestion(code=code, message=message))
    return tuple(items)


def _merge_disallowed(
    existing: Sequence[DisallowedSuggestion],
    extra: Sequence[DisallowedSuggestion],
) -> tuple[DisallowedSuggestion, ...]:
    seen = {item.code for item in existing}
    merged = list(existing)
    for item in extra:
        if item.code not in seen:
            seen.add(item.code)
            merged.append(item)
    return tuple(merged)


def _clip_summary(text: str, limit: int = 240) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"
