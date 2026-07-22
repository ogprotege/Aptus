from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MethodLifecycle(StrEnum):
    """Product status, kept separate from a method's research identity."""

    GATED_EXECUTABLE = "gated-executable"
    EXPERIMENTAL = "experimental"
    RESEARCH_ONLY = "research-only"


@dataclass(frozen=True)
class RuntimeBinding:
    """One executable method compiler on one concrete training runtime."""

    training_runtime: str
    compute_backend: str
    compiler_id: str
    estimator_id: str
    export_kind: str
    supported_distributions: tuple[str, ...]
    evidence_requirement: str = "pilot-required"
    schema_version: str = "aptus.runtime-binding.v1"

    def __post_init__(self) -> None:
        values = (
            self.training_runtime,
            self.compute_backend,
            self.compiler_id,
            self.estimator_id,
            self.export_kind,
            self.evidence_requirement,
        )
        if any(not value.strip() for value in values):
            raise ValueError("Runtime binding identities must be non-empty.")
        if not self.supported_distributions:
            raise ValueError("Runtime bindings require distribution support.")


@dataclass(frozen=True)
class MethodDescriptor:
    """Versioned method metadata used by the API and workbench.

    A descriptor does not make a method executable. Only ``selectable=True``
    plus a registered compiler ID may enter the planner's current candidate
    matrix.
    """

    method_id: str
    display_name: str
    summary: str
    lifecycle: MethodLifecycle
    selectable: bool
    parameter_scope: str
    parameterization: str
    base_storage: str
    compiler_id: str | None
    export_kind: str | None
    supported_backends: tuple[str, ...]
    supported_distributions: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    pilot_requirement: str
    blocker: str | None = None
    aliases: tuple[str, ...] = ()
    runtime_bindings: tuple[RuntimeBinding, ...] = ()
    schema_version: str = "aptus.method-descriptor.v1"

    def __post_init__(self) -> None:
        if not self.method_id or self.method_id != self.method_id.strip().lower():
            raise ValueError("method_id must be a non-empty lowercase identifier.")
        if self.selectable:
            if self.lifecycle != MethodLifecycle.GATED_EXECUTABLE:
                raise ValueError("Selectable methods must be gated executable.")
            if not self.compiler_id or not self.export_kind:
                raise ValueError(
                    "Selectable methods require compiler and export contracts."
                )
            if not self.supported_backends or not self.supported_distributions:
                raise ValueError(
                    "Selectable methods require explicit backend and distribution support."
                )
            if not self.runtime_bindings:
                raise ValueError("Selectable methods require runtime bindings.")
        elif self.blocker is None:
            raise ValueError("Non-selectable methods require an explicit blocker.")
