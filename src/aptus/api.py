from __future__ import annotations

import json
import os
import secrets
import shutil
import stat
import sys
import threading
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Self
from urllib.parse import urlencode

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import __version__
from .api_contracts import (
    API_CONTRACT_VERSION,
    BootstrapResponse,
    CompileResponse,
    ErrorResponse,
    EvaluationContractResponse,
    EvaluationResultResponse,
    HardwareProbeResponse,
    HealthResponse,
    InferenceGenerateResponse,
    InferenceModelsResponse,
    InferenceServicesResponse,
    JobResponse,
    ModelInspectionReceiptResponse,
    ModelInspectionResponse,
    NoFeasiblePlanResponse,
    PlatformResponse,
    ProfileResponse,
    ProjectRecoveryResponse,
    ProjectResponse,
    ProjectRevisionResponse,
    ProjectRevisionSummary,
    ProjectSummaryResponse,
    RuntimeConfiguredResponse,
    RuntimeInventoryResponse,
    TrainingPlanResponse,
    ValidationResponse,
)
from .catalog import STACK_VERSIONS, TARGET_MODULES
from .domain import (
    Backend,
    Method,
    Objective,
    SCHEMA_VERSION,
    TrainingRuntime,
    TrainingPlan,
    TrainingTarget,
    UnsupportedPlanSchemaError,
    model_inspection_receipt_from_primitive,
    to_primitive,
    training_plan_from_primitive,
)
from .evidence import EVIDENCE_REGISTRY
from .integrations import (
    LMStudioClient,
    OMLXClient,
    LocalInferenceError,
    discover_local_inference_services,
)
from .local_store import atomic_write_json, private_directory, read_json_object
from .methods import method_descriptors, selectable_method_ids
from .plan_contract import (
    StaleModelPolicyError,
    require_current_model_policy,
    sha256_file,
    validate_bundle_manifest,
    validate_plan_payload,
)
from .correction import (
    attach_correction,
    build_no_path_correction,
    build_plan_correction,
)
from .planning import NoFeasiblePlanError, plan_training, select_candidate
from .profiling import (
    build_hardware_spec,
    build_model_spec,
    probe_local_hardware,
    profile_dataset,
    probe_apple_platform,
)
from .runtime_env import (
    runtime_environment_key,
    runtime_inventory,
    validate_runtime_configuration,
)
from .projects import ProjectRepository


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _replan_required_payload(
    plan_payload: Mapping[str, Any],
    *,
    source: str,
    project_id: str | None = None,
    project_revision_id: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    return {
        "status": "replan_required",
        "plan_id": (
            plan_payload.get("plan_id")
            if isinstance(plan_payload.get("plan_id"), str)
            else None
        ),
        "found_schema": (
            plan_payload.get("schema_version")
            if isinstance(plan_payload.get("schema_version"), str)
            else None
        ),
        "required_schema": SCHEMA_VERSION,
        "source": source,
        "project_id": project_id,
        "project_revision_id": project_revision_id,
        "message": message
        or (
            "This saved plan predates the current executable contract. Create a "
            "new plan from its preserved facts before compiling or recovering it."
        ),
    }


def _stale_policy_error_payload(
    *,
    message: str,
    plan_id: str | None = None,
    project_id: str | None = None,
    project_revision_id: str | None = None,
) -> dict[str, Any]:
    return {
        "error": "replan_required",
        "plan_id": plan_id,
        "found_schema": SCHEMA_VERSION,
        "required_schema": SCHEMA_VERSION,
        "project_id": project_id,
        "project_revision_id": project_revision_id,
        "message": message,
    }


class MoETopologyRequest(StrictModel):
    expert_count: Annotated[int, Field(strict=True, gt=0)]
    experts_per_token: Annotated[int, Field(strict=True, gt=0)]
    expert_intermediate_size: Annotated[int, Field(strict=True, gt=0)]
    decoder_sparse_step: Annotated[int, Field(strict=True, gt=0)]
    mlp_only_layers: list[Annotated[int, Field(strict=True, ge=0)]] = Field(
        default_factory=list
    )
    shared_expert_intermediate_size: Annotated[int, Field(strict=True, gt=0)] | None = (
        None
    )


class QuantizationOverrideRequest(StrictModel):
    module_path: str = Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$",
    )
    bits: Annotated[int, Field(strict=True, ge=1, le=16)]
    group_size: Annotated[int, Field(strict=True, gt=0)]


class QuantizationLayoutRequest(StrictModel):
    default_bits: Annotated[int, Field(strict=True, ge=1, le=16)]
    default_group_size: Annotated[int, Field(strict=True, gt=0)]
    module_overrides: list[QuantizationOverrideRequest] = Field(default_factory=list)


class ModelFactsRequest(StrictModel):
    model_id: str
    revision: str
    family: str
    parameters_b: float = Field(gt=0)
    hidden_size: int = Field(gt=0)
    layers: int = Field(gt=0)
    context_length: int = Field(gt=0)
    license_name: str
    training_allowed: bool
    intermediate_size: int | None = Field(default=None, gt=0)
    model_type: str | None = None
    architecture: str | None = None
    quantization_bits: Annotated[int, Field(strict=True, ge=1, le=16)] | None = None
    quantization_layout: QuantizationLayoutRequest | None = None
    moe: MoETopologyRequest | None = None


class HardwareFactsRequest(StrictModel):
    discovery: Literal["manual", "local-scan"] = "manual"
    backend: Backend = Backend.CUDA
    gpu_count: int = Field(gt=0)
    vram_gib: float = Field(gt=0)
    supports_bf16: bool = False
    supports_4bit: bool = False
    supports_8bit: bool = False
    free_vram_gib: float | None = Field(default=None, gt=0)
    host_ram_gib: float = Field(gt=0)
    host_ram_free_gib: float | None = Field(default=None, gt=0)
    reserve_gib: float = Field(default=2.0, ge=0)
    disk_free_gib: float | None = Field(default=None, gt=0)


class TargetRequest(StrictModel):
    objective: Objective
    sequence_length: int = Field(gt=0)
    effective_batch_size: int = Field(default=16, gt=0)
    max_epochs: int = Field(default=3, gt=0)
    method_preference: Method | None = None
    training_runtime: TrainingRuntime | None = None
    task: str = "sft"
    evaluation_fraction: float = Field(default=0.1, ge=0, lt=1)
    packing: bool = False
    checkpoint_steps: int = Field(default=100, gt=0)
    optimizer_steps: int | None = Field(default=None, gt=0)
    split_seed: int = Field(default=424242, ge=0)
    training_seed: int = Field(default=17, ge=0)
    data_order_seed: int = Field(default=1000017, ge=0)
    micro_batch_size: int | None = Field(default=None, gt=0)
    gradient_accumulation_steps: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_phase3_control_consistency(self) -> Self:
        if self.data_order_seed != 1_000_000 + self.training_seed:
            raise ValueError("data_order_seed must equal 1000000 + training_seed.")
        if (self.micro_batch_size is None) != (
            self.gradient_accumulation_steps is None
        ):
            raise ValueError(
                "micro_batch_size and gradient_accumulation_steps must be supplied together."
            )
        return self


class ProfileRequest(StrictModel):
    dataset_path: str
    sample_limit: int | None = Field(default=512, gt=0)
    sequence_length: int | None = Field(default=None, gt=0)


class EvaluationContractRequest(StrictModel):
    dataset_path: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    threshold: float = Field(ge=0, le=1)
    metric: Literal["exact_match"] = "exact_match"
    gold_field: Literal["completion", "output", "gold"] = "completion"
    id_field: str | None = "id"
    casefold: bool = False
    plan_id: str | None = None
    candidate_id: str | None = None
    job_id: str | None = None
    export_digest: str | None = None
    export_kind: Literal["adapter", "final-export"] | None = None


class EvaluationScoreRequest(StrictModel):
    contract: EvaluationContractResponse
    gold_path: str = Field(min_length=1)
    predictions_path: str = Field(min_length=1)
    export_digest: str | None = None


class PlanRequest(StrictModel):
    model: ModelFactsRequest
    hardware: HardwareFactsRequest
    target: TargetRequest
    dataset_path: str
    sample_limit: int | None = Field(default=512, gt=0)
    inspection_receipt: ModelInspectionReceiptResponse | None = None
    project_id: str | None = Field(default=None, pattern=r"^project_[0-9a-f]{32}$")
    project_name: str | None = Field(default=None, min_length=1, max_length=120)


class CompileRequest(StrictModel):
    plan_id: str
    output_dir: str
    project_id: str = Field(pattern=r"^project_[0-9a-f]{32}$")
    expected_project_revision_id: str = Field(pattern=r"^revision_[0-9a-f]{32}$")


class SelectCandidateRequest(StrictModel):
    plan_id: str = Field(pattern=r"^plan_[0-9a-f]{20}$")
    candidate_id: str = Field(pattern=r"^cand_[0-9a-f]{20}$")
    project_id: str = Field(pattern=r"^project_[0-9a-f]{32}$")
    expected_project_revision_id: str = Field(pattern=r"^revision_[0-9a-f]{32}$")


class ValidateRequest(StrictModel):
    bundle_dir: str
    project_id: str = Field(pattern=r"^project_[0-9a-f]{32}$")
    expected_project_revision_id: str = Field(pattern=r"^revision_[0-9a-f]{32}$")
    level: Literal[
        "contract", "static", "dependency", "model-data", "measured-preflight", "pilot"
    ] = "static"
    run: bool = False


class JobRequest(StrictModel):
    bundle_dir: str
    project_id: str = Field(pattern=r"^project_[0-9a-f]{32}$")
    expected_project_revision_id: str = Field(pattern=r"^revision_[0-9a-f]{32}$")
    action: Literal["dependency", "model-data", "preflight", "pilot", "train"] = (
        "preflight"
    )
    confirm_full_train: bool = False


class ModelInspectRequest(StrictModel):
    model_id: str
    revision: str
    timeout_seconds: float = Field(default=10.0, gt=0, le=30)


class InferenceServiceRequest(StrictModel):
    service: Literal["lm-studio", "omlx"]
    endpoint: str | None = None
    timeout_seconds: float = Field(default=5.0, gt=0, le=30)


class InferenceMessage(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1, max_length=131_072)


class InferenceGenerateRequest(InferenceServiceRequest):
    model: str = Field(min_length=1, max_length=256)
    messages: list[InferenceMessage] = Field(min_length=1, max_length=256)
    max_tokens: int = Field(default=256, ge=1, le=32_768)
    temperature: float = Field(default=0.0, ge=0, le=2)


class RuntimeConfigureRequest(StrictModel):
    runtime_id: Literal["mlx-lm", "pytorch-mps", "transformers-peft-cuda"]
    interpreter_path: str = Field(min_length=1, max_length=4096)


class ProjectCreateRequest(StrictModel):
    name: str = Field(min_length=1, max_length=120)


class ProjectRecoverRequest(StrictModel):
    revision_id: str = Field(pattern=r"^revision_[0-9a-f]{32}$")


class ApiContext:
    def __init__(self, state_dir: Path) -> None:
        from .execution import JobService

        self.state_dir = private_directory(state_dir)
        self.plans_dir = private_directory(self.state_dir / "plans")
        self.current_bundle_path = self.state_dir / "current-bundle.json"
        self.runtime_config_path = self.state_dir / "runtime-config.json"
        self._state_lock = threading.RLock()
        self.runtime_paths = self._load_runtime_configuration()
        runtime_environment = dict(os.environ)
        runtime_environment.update(
            {
                runtime_environment_key(runtime_id): path
                for runtime_id, path in self.runtime_paths.items()
            }
        )
        self.jobs = JobService(
            self.state_dir / "jobs", runtime_environment=runtime_environment
        )
        self.projects = ProjectRepository(self.state_dir)
        self.projects.import_legacy(
            plans_dir=self.plans_dir,
            current_bundle_path=self.current_bundle_path,
            jobs=self.jobs.list(),
        )

    def _load_runtime_configuration(self) -> dict[str, str]:
        if not self.runtime_config_path.exists():
            return {}
        if (
            self.runtime_config_path.is_symlink()
            or not self.runtime_config_path.is_file()
        ):
            raise PermissionError(
                f"Aptus runtime configuration must be a regular file: {self.runtime_config_path}"
            )
        value = json.loads(self.runtime_config_path.read_text(encoding="utf-8"))
        runtimes = value.get("runtimes") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != "aptus.runtime-config.v1"
            or not isinstance(runtimes, dict)
        ):
            raise ValueError("Aptus runtime configuration has an invalid contract.")
        result: dict[str, str] = {}
        for runtime_id, path in runtimes.items():
            runtime_environment_key(str(runtime_id))
            if not isinstance(path, str) or not path:
                raise ValueError(
                    "Aptus runtime configuration contains an invalid path."
                )
            result[str(runtime_id)] = path
        return result

    def configure_runtime(
        self, runtime_id: str, interpreter_path: Path
    ) -> dict[str, Any]:
        probe = validate_runtime_configuration(runtime_id, interpreter_path)
        key = runtime_environment_key(runtime_id)
        with self._state_lock:
            self.runtime_paths[runtime_id] = probe.path
            self.jobs.runtime_environment[key] = probe.path
            atomic_write_json(
                self.runtime_config_path,
                {
                    "schema_version": "aptus.runtime-config.v1",
                    "runtimes": dict(sorted(self.runtime_paths.items())),
                },
                mode=0o600,
            )
        return {
            "status": "ok",
            "runtime_id": runtime_id,
            "interpreter_path": probe.path,
            "interpreter": probe.to_dict(),
            "persisted": True,
        }

    def save_plan(self, plan: TrainingPlan) -> None:
        path = self.plans_dir / f"{plan.plan_id}.json"
        with self._state_lock:
            atomic_write_json(path, to_primitive(plan), mode=0o600)

    def load_plan(self, plan_id: str) -> TrainingPlan | None:
        if not (
            plan_id.startswith("plan_")
            and len(plan_id) == 25
            and all(c in "0123456789abcdef" for c in plan_id[5:])
        ):
            return None
        path = self.plans_dir / f"{plan_id}.json"
        with self._state_lock:
            if not path.is_file():
                return None
            if path.is_symlink():
                raise PermissionError(f"Aptus saved plans cannot be symlinks: {path}")
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, RecursionError, ValueError):
                raise ValueError("Saved plan is unreadable or invalid JSON.") from None
        if not isinstance(value, Mapping):
            raise ValueError("Saved plan must be a JSON object.")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise UnsupportedPlanSchemaError(value.get("schema_version"))
        if value.get("plan_id") != plan_id:
            raise ValueError("Saved plan ID does not match its requested filename.")
        require_current_model_policy(value)
        validation_errors = validate_plan_payload(value, verify_dataset=False)
        if validation_errors:
            raise ValueError(
                "Saved plan failed the current executable contract: "
                + " ".join(validation_errors)
            )
        return training_plan_from_primitive(value)

    def save_bundle(
        self, bundle_dir: Path, archive_path: Path, *, plan_id: str
    ) -> None:
        with self._state_lock:
            atomic_write_json(
                self.current_bundle_path,
                {
                    "schema_version": "aptus.current-bundle.v1",
                    "plan_id": plan_id,
                    "bundle_dir": str(bundle_dir.resolve()),
                    "archive_path": str(archive_path.resolve()),
                },
                mode=0o600,
            )

    def load_bundle_reference(self) -> dict[str, str] | None:
        with self._state_lock:
            if (
                self.current_bundle_path.is_symlink()
                or not self.current_bundle_path.is_file()
            ):
                return None
            try:
                value = read_json_object(
                    self.current_bundle_path, "Aptus current-bundle reference"
                )
            except (OSError, ValueError):
                return None
        schema_version = value.get("schema_version")
        if schema_version not in {None, "aptus.current-bundle.v1"}:
            return None
        if not isinstance(value.get("bundle_dir"), str):
            return None
        return {
            "bundle_dir": value["bundle_dir"],
            "archive_path": str(value.get("archive_path", "")),
            "plan_id": str(value.get("plan_id", "")),
        }


def _build_plan(request: PlanRequest) -> TrainingPlan:
    dataset = profile_dataset(
        Path(request.dataset_path),
        sample_limit=request.sample_limit,
        sequence_length=request.target.sequence_length,
    )
    model = build_model_spec(**request.model.model_dump(exclude_none=True))
    reserve_gib = request.hardware.reserve_gib
    uses_unified_memory = (
        request.hardware.backend == Backend.MPS
        or request.target.training_runtime in {"mlx-lm", "pytorch-mps"}
        or (request.hardware.discovery == "local-scan" and sys.platform == "darwin")
    )
    if uses_unified_memory:
        reserve_gib = max(reserve_gib, 8.0)
    if request.hardware.discovery == "local-scan":
        hardware = probe_local_hardware(reserve_gib=reserve_gib)
    else:
        hardware_values = request.hardware.model_dump(exclude={"discovery"})
        hardware_values["reserve_gib"] = reserve_gib
        hardware = build_hardware_spec(**hardware_values)
    target = TrainingTarget(**request.target.model_dump())
    inspection_receipt = (
        model_inspection_receipt_from_primitive(
            request.inspection_receipt.model_dump(mode="json")
        )
        if request.inspection_receipt is not None
        else None
    )
    return plan_training(
        model=model,
        dataset=dataset,
        hardware=hardware,
        target=target,
        inspection_receipt=inspection_receipt,
    )


def create_app(
    *,
    state_dir: Path | None = None,
    static_dir: Path | None = None,
    allowed_hosts: tuple[str, ...] | None = None,
    session_token: str | None = None,
    execution_enabled: bool = True,
) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.encoders import jsonable_encoder
        from fastapi.exceptions import HTTPException as FastApiHTTPException
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
        from starlette.middleware.trustedhost import TrustedHostMiddleware
    except ImportError as error:  # pragma: no cover - exercised in deployment extras
        raise RuntimeError(
            "Install Aptus with the 'server' extra to use the API."
        ) from error

    from .generation import bundle_files, create_bundle_archive, generate_bundle
    from .inspection import inspect_huggingface_model
    from .execution import (
        ActiveJobError,
        JobPrerequisiteError,
        decorate_validation_authorization,
    )
    from .validation import validate_bundle

    desktop_security_headers = {
        "Content-Security-Policy": (
            "default-src 'self'; base-uri 'none'; connect-src 'self'; "
            "font-src 'self'; form-action 'self'; frame-ancestors 'none'; "
            "frame-src 'none'; img-src 'self' data:; object-src 'none'; "
            "script-src 'self'; style-src 'self' 'unsafe-inline'"
        ),
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
    }

    if session_token is not None and len(session_token) < 32:
        raise ValueError("Desktop session tokens must contain at least 32 characters.")

    context = ApiContext((state_dir or Path(".aptus-state")).resolve())
    app = FastAPI(
        title="Aptus API",
        version=__version__,
        responses={
            status_code: {"model": ErrorResponse}
            for status_code in (400, 403, 404, 409, 422, 502, 504)
        },
    )
    trusted_hosts = set(allowed_hosts or ())
    trusted_hosts.update({"127.0.0.1", "localhost", "[::1]", "testserver"})
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=sorted(trusted_hosts),
    )

    @app.middleware("http")
    async def enforce_session_boundary(request: Any, call_next: Any) -> Any:
        path = request.url.path
        protected = (
            path == "/api"
            or (path.startswith("/api/") and path != "/api/v1/health")
            or path in {"/docs", "/redoc", "/openapi.json"}
        )

        if session_token is not None:
            query_token = request.query_params.get("aptus_session_token", "")
            if request.method == "GET" and not protected and query_token:
                if not secrets.compare_digest(query_token, session_token):
                    return JSONResponse(
                        status_code=403,
                        content={"error": "desktop_session_required"},
                        headers=desktop_security_headers,
                    )
                remaining_query = urlencode(
                    [
                        (key, value)
                        for key, value in request.query_params.multi_items()
                        if key != "aptus_session_token"
                    ]
                )
                location = path or "/"
                if remaining_query:
                    location = f"{location}?{remaining_query}"
                response = RedirectResponse(url=location, status_code=303)
                response.set_cookie(
                    "aptus_desktop_session",
                    session_token,
                    httponly=True,
                    samesite="strict",
                    path="/",
                )
                for name, value in desktop_security_headers.items():
                    response.headers[name] = value
                return response

            if protected:
                cookie_token = request.cookies.get("aptus_desktop_session", "")
                authorization = request.headers.get("authorization", "")
                authorization_scheme, separator, authorization_value = (
                    authorization.partition(" ")
                )
                bearer_token = (
                    authorization_value
                    if separator and authorization_scheme.lower() == "bearer"
                    else ""
                )
                authorized = secrets.compare_digest(cookie_token, session_token)
                if not authorized:
                    authorized = secrets.compare_digest(bearer_token, session_token)
                if not authorized:
                    return JSONResponse(
                        status_code=403,
                        content={"error": "desktop_session_required"},
                        headers=desktop_security_headers,
                    )

        response = await call_next(request)
        for name, value in desktop_security_headers.items():
            response.headers[name] = value
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Any, error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(
                {"error": "request_validation", "details": error.errors()}
            ),
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found(_request: Any, error: FileNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404, content={"error": "path_not_found", "details": str(error)}
        )

    @app.exception_handler(FileExistsError)
    async def file_conflict(_request: Any, error: FileExistsError) -> JSONResponse:
        return JSONResponse(
            status_code=409, content={"error": "path_conflict", "details": str(error)}
        )

    @app.exception_handler(PermissionError)
    async def path_forbidden(_request: Any, error: PermissionError) -> JSONResponse:
        return JSONResponse(
            status_code=403, content={"error": "path_forbidden", "details": str(error)}
        )

    @app.exception_handler(OSError)
    async def filesystem_error(_request: Any, error: OSError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": "filesystem_error", "details": str(error)},
        )

    @app.exception_handler(ValueError)
    async def value_error(_request: Any, error: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400, content={"error": "invalid_request", "details": str(error)}
        )

    @app.exception_handler(NoFeasiblePlanError)
    async def no_feasible_plan(
        _request: Any, error: NoFeasiblePlanError
    ) -> JSONResponse:
        correction = build_no_path_correction(
            error.candidates,
            ranking_objective=error.ranking_objective,
        )
        payload = NoFeasiblePlanResponse.model_validate(
            {
                "error": "no_feasible_plan",
                "message": str(error),
                "model": to_primitive(error.model),
                "candidates": [to_primitive(item) for item in error.candidates],
                "model_policy_decision": to_primitive(error.model_policy_decision),
                "model_policy_decision_source": (
                    error.model_policy_decision_source.value
                ),
                "inspection_receipt": to_primitive(error.inspection_receipt),
                "correction": correction.to_primitive(),
            }
        )
        return JSONResponse(
            status_code=422,
            content=payload.model_dump(mode="json"),
        )

    @app.exception_handler(LocalInferenceError)
    async def local_inference_error(
        _request: Any, error: LocalInferenceError
    ) -> JSONResponse:
        if error.code in {
            "unsupported_service",
            "invalid_endpoint",
            "non_loopback_endpoint",
            "invalid_timeout",
            "invalid_request",
            "request_too_large",
        }:
            status_code = 400
        elif error.code == "timeout":
            status_code = 504
        else:
            status_code = 502
        return JSONResponse(status_code=status_code, content=error.to_dict())

    @app.exception_handler(FastApiHTTPException)
    async def http_error(_request: Any, error: FastApiHTTPException) -> JSONResponse:
        detail = error.detail
        if isinstance(detail, dict):
            content = detail
        else:
            content = {"error": "http_error", "details": detail}
        return JSONResponse(status_code=error.status_code, content=content)

    def require_current_project_revision(
        *,
        project_id: str,
        expected_revision_id: str,
        plan_id: str | None = None,
        bundle_dir: Path | None = None,
    ) -> dict[str, Any]:
        try:
            project = context.projects.get(project_id)
        except (KeyError, OSError, ValueError) as error:
            raise HTTPException(
                status_code=404,
                detail={"error": "project_not_found", "project_id": project_id},
            ) from error
        actual_revision_id = project.get("latest_revision_id")
        if actual_revision_id != expected_revision_id:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "project_revision_conflict",
                    "project_id": project_id,
                    "expected_project_revision_id": expected_revision_id,
                    "actual_project_revision_id": actual_revision_id,
                },
            )
        try:
            revision = context.projects.revision(project_id, expected_revision_id)
        except (KeyError, OSError, ValueError) as error:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "project_revision_not_found",
                    "project_id": project_id,
                    "revision_id": expected_revision_id,
                },
            ) from error
        if plan_id is not None and revision.get("plan_id") != plan_id:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "project_plan_mismatch",
                    "project_id": project_id,
                    "expected_project_revision_id": expected_revision_id,
                    "plan_id": plan_id,
                },
            )
        if bundle_dir is not None:
            recorded_bundle = revision.get("bundle")
            recorded_path = (
                recorded_bundle.get("bundle_dir")
                if isinstance(recorded_bundle, dict)
                else None
            )
            if not isinstance(recorded_path, str) or (
                Path(recorded_path).resolve() != bundle_dir.resolve()
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "project_bundle_mismatch",
                        "project_id": project_id,
                        "expected_project_revision_id": expected_revision_id,
                        "bundle_dir": str(bundle_dir.resolve()),
                    },
                )
            try:
                revision = context.projects.validate_revision_artifacts(
                    project_id, expected_revision_id
                )
            except (KeyError, OSError, ValueError) as error:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "project_bundle_binding_mismatch",
                        "project_id": project_id,
                        "expected_project_revision_id": expected_revision_id,
                        "bundle_dir": str(bundle_dir.resolve()),
                        "message": str(error),
                    },
                ) from error
        return revision

    def create_current_project_revision(
        *,
        project_id: str,
        expected_revision_id: str,
        reason: str,
        **changes: Any,
    ) -> dict[str, Any]:
        try:
            return context.projects.create_revision(
                project_id,
                reason=reason,
                base_revision_id=expected_revision_id,
                expected_latest_revision_id=expected_revision_id,
                **changes,
            )
        except ValueError as error:
            try:
                actual_revision_id = context.projects.get(project_id).get(
                    "latest_revision_id"
                )
            except (KeyError, OSError, ValueError):
                raise error
            if actual_revision_id != expected_revision_id:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "project_revision_conflict",
                        "project_id": project_id,
                        "expected_project_revision_id": expected_revision_id,
                        "actual_project_revision_id": actual_revision_id,
                    },
                ) from error
            raise

    @app.get("/api/v1/health", response_model=HealthResponse)
    @app.get("/health", include_in_schema=False, response_model=HealthResponse)
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "version": __version__,
            "api_contract_version": API_CONTRACT_VERSION,
        }

    @app.get("/api/v1/bootstrap", response_model=BootstrapResponse)
    def bootstrap() -> dict[str, Any]:
        preferred_backend = (
            Backend.MPS.value if sys.platform == "darwin" else Backend.CUDA.value
        )
        preferred_runtime = (
            "mlx-lm"
            if preferred_backend == Backend.MPS.value
            else "transformers-peft-cuda"
        )
        method_catalog = [to_primitive(item) for item in method_descriptors()]
        projects = context.projects.list()
        current_project = context.projects.current()
        capabilities = {
            "backends": [Backend.CUDA.value, Backend.MPS.value],
            "known_backends": [item.value for item in Backend],
            "training_runtimes": ["transformers-peft-cuda", "mlx-lm"],
            "known_training_runtimes": [
                "transformers-peft-cuda",
                "mlx-lm",
                "pytorch-mps",
            ],
            "inference_services": ["lm-studio", "omlx"],
            "methods": list(selectable_method_ids()),
            "method_catalog": method_catalog,
            "objectives": [item.value for item in Objective],
            "supported_execution_backend": preferred_backend,
            "supported_execution_backends": [Backend.CUDA.value, Backend.MPS.value],
            "local_execution_enabled": execution_enabled,
            "model_families": sorted(TARGET_MODULES),
            "validation_levels": [
                "contract",
                "static",
                "dependency",
                "model-data",
                "measured-preflight",
                "pilot",
            ],
        }
        result = {
            "api_contract_version": API_CONTRACT_VERSION,
            "version": __version__,
            **capabilities,
            "stack_versions": STACK_VERSIONS,
            "evidence": [to_primitive(item) for item in EVIDENCE_REGISTRY.values()],
            "calibrated": False,
            "service": {"name": "aptus", "version": __version__, "scope": "local-host"},
            "capabilities": capabilities,
            "defaults": {
                "sample_limit": 512,
                "backend": preferred_backend,
                "training_runtime": preferred_runtime,
                "reserve_gib": (8.0 if preferred_backend == Backend.MPS.value else 2.0),
                "task": "sft",
                "packing": False,
            },
            "projects": projects,
            "project": current_project,
            "project_history": (
                context.projects.history(current_project["project_id"])
                if current_project is not None
                else []
            ),
        }
        current_revision = (
            current_project.get("latest_revision")
            if isinstance(current_project, dict)
            else None
        )
        current_plan_snapshot = (
            current_revision.get("plan_snapshot")
            if isinstance(current_revision, dict)
            else None
        )
        if (
            isinstance(current_plan_snapshot, dict)
            and current_plan_snapshot.get("schema_version") != SCHEMA_VERSION
        ):
            result["replan_required"] = _replan_required_payload(
                current_plan_snapshot,
                source="project-revision",
                project_id=(
                    current_revision.get("project_id")
                    if isinstance(current_revision.get("project_id"), str)
                    else None
                ),
                project_revision_id=(
                    current_revision.get("revision_id")
                    if isinstance(current_revision.get("revision_id"), str)
                    else None
                ),
            )
        elif isinstance(current_plan_snapshot, dict):
            try:
                require_current_model_policy(current_plan_snapshot)
            except StaleModelPolicyError as error:
                result["replan_required"] = _replan_required_payload(
                    current_plan_snapshot,
                    source="project-revision",
                    project_id=(
                        current_revision.get("project_id")
                        if isinstance(current_revision.get("project_id"), str)
                        else None
                    ),
                    project_revision_id=(
                        current_revision.get("revision_id")
                        if isinstance(current_revision.get("revision_id"), str)
                        else None
                    ),
                    message=str(error),
                )
            except ValueError:
                pass
        current_revision_job_ids = (
            {
                str(job_id)
                for job_id in current_revision.get("job_ids", [])
                if isinstance(job_id, str)
            }
            if isinstance(current_revision, dict)
            else set()
        )
        current_revision_bundle = (
            current_revision.get("bundle")
            if isinstance(current_revision, dict)
            else None
        )
        current_revision_bundle_dir = (
            current_revision_bundle.get("bundle_dir")
            if isinstance(current_revision_bundle, dict)
            else None
        )

        def job_matches_current_revision(item: dict[str, Any]) -> bool:
            if current_project is None:
                return True
            job_bundle_dir = item.get("bundle_dir")
            if (
                item.get("id") not in current_revision_job_ids
                or not isinstance(current_revision_bundle_dir, str)
                or not isinstance(job_bundle_dir, str)
            ):
                return False
            return (
                Path(job_bundle_dir).resolve()
                == Path(current_revision_bundle_dir).resolve()
            )

        jobs = context.jobs.list()
        active_job = next(
            (
                item
                for item in jobs
                if item.get("state") in {"queued", "running", "cancelling"}
                and job_matches_current_revision(item)
            ),
            None,
        )
        if active_job is not None and current_project is None:
            result["job"] = active_job

        saved_reference = context.load_bundle_reference()
        references: list[dict[str, Any]] = []
        if active_job is not None and isinstance(active_job.get("bundle_dir"), str):
            active_archive = ""
            active_archive_sha256 = None
            active_archive_size_bytes = None
            if isinstance(
                current_revision_bundle, dict
            ) and current_revision_bundle.get("bundle_dir") == active_job.get(
                "bundle_dir"
            ):
                active_archive = str(current_revision_bundle.get("archive_path") or "")
                active_archive_sha256 = current_revision_bundle.get("archive_sha256")
                active_archive_size_bytes = current_revision_bundle.get(
                    "archive_size_bytes"
                )
            if (
                not active_archive
                and saved_reference is not None
                and Path(saved_reference["bundle_dir"]).resolve()
                == Path(active_job["bundle_dir"]).resolve()
            ):
                active_archive = saved_reference["archive_path"]
            references.append(
                {
                    "bundle_dir": active_job["bundle_dir"],
                    "archive_path": active_archive,
                    "archive_sha256": active_archive_sha256,
                    "archive_size_bytes": active_archive_size_bytes,
                    "current_revision_bound": current_project is not None,
                }
            )
        current_project_bundle = (
            current_revision.get("bundle")
            if isinstance(current_revision, dict)
            else None
        )
        if isinstance(current_project_bundle, dict) and isinstance(
            current_project_bundle.get("bundle_dir"), str
        ):
            project_reference = {
                "bundle_dir": current_project_bundle["bundle_dir"],
                "archive_path": str(current_project_bundle.get("archive_path") or ""),
                "archive_sha256": current_project_bundle.get("archive_sha256"),
                "archive_size_bytes": current_project_bundle.get("archive_size_bytes"),
                "current_revision_bound": True,
            }
            if all(
                item["bundle_dir"] != project_reference["bundle_dir"]
                for item in references
            ):
                references.append(project_reference)
        if (
            current_project is None
            and saved_reference is not None
            and all(
                item["bundle_dir"] != saved_reference["bundle_dir"]
                for item in references
            )
        ):
            references.append(saved_reference)

        restored_bundle_dir: Path | None = None
        for reference in references:
            bundle_dir = Path(reference["bundle_dir"]).resolve()
            plan_path = bundle_dir / "plan.json"
            if not bundle_dir.is_dir() or not plan_path.is_file():
                continue
            try:
                plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            except (OSError, RecursionError, ValueError):
                continue
            if (
                not isinstance(plan_payload, dict)
                or plan_payload.get("schema_version") != SCHEMA_VERSION
            ):
                if isinstance(plan_payload, dict):
                    result.setdefault(
                        "replan_required",
                        _replan_required_payload(
                            plan_payload,
                            source="compiled-bundle",
                            project_id=(
                                current_revision.get("project_id")
                                if isinstance(current_revision, dict)
                                and isinstance(current_revision.get("project_id"), str)
                                else None
                            ),
                            project_revision_id=(
                                current_revision.get("revision_id")
                                if isinstance(current_revision, dict)
                                and isinstance(current_revision.get("revision_id"), str)
                                else None
                            ),
                        ),
                    )
                continue
            try:
                require_current_model_policy(plan_payload)
            except StaleModelPolicyError as error:
                result.setdefault(
                    "replan_required",
                    _replan_required_payload(
                        plan_payload,
                        source="compiled-bundle",
                        project_id=(
                            current_revision.get("project_id")
                            if isinstance(current_revision, dict)
                            and isinstance(current_revision.get("project_id"), str)
                            else None
                        ),
                        project_revision_id=(
                            current_revision.get("revision_id")
                            if isinstance(current_revision, dict)
                            and isinstance(current_revision.get("revision_id"), str)
                            else None
                        ),
                        message=str(error),
                    ),
                )
                continue
            except ValueError:
                continue
            if validate_plan_payload(
                plan_payload, root=bundle_dir, verify_dataset=False
            ) or validate_bundle_manifest(bundle_dir):
                continue
            try:
                manifest_payload = json.loads(
                    (bundle_dir / "bundle-manifest.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(manifest_payload, dict):
                continue
            if reference.get("current_revision_bound") is True:
                expected_plan_id = (
                    current_revision.get("plan_id")
                    if isinstance(current_revision, dict)
                    else None
                )
                expected_candidate_id = (
                    current_revision.get("selected_candidate_id")
                    if isinstance(current_revision, dict)
                    else None
                )
                recommended = plan_payload.get("recommended")
                loaded_candidate_id = (
                    recommended.get("candidate_id")
                    if isinstance(recommended, dict)
                    else None
                )
                expected_fingerprint = (
                    current_revision_bundle.get("artifact_fingerprint")
                    if isinstance(current_revision_bundle, dict)
                    else None
                )
                if (
                    not isinstance(expected_plan_id, str)
                    or plan_payload.get("plan_id") != expected_plan_id
                    or manifest_payload.get("plan_id") != expected_plan_id
                    or not isinstance(expected_candidate_id, str)
                    or loaded_candidate_id != expected_candidate_id
                    or manifest_payload.get("candidate_id") != expected_candidate_id
                    or not isinstance(expected_fingerprint, str)
                    or sha256_file(bundle_dir / "bundle-manifest.json")
                    != expected_fingerprint
                ):
                    continue
            dataset_value = plan_payload.get("dataset", {})
            dataset_path = dataset_value.get("source_path")
            dataset_digest = dataset_value.get("source_sha256")
            manifest_files = manifest_payload.get("files", [])
            dataset_entry = next(
                (
                    item
                    for item in manifest_files
                    if isinstance(item, dict) and item.get("path") == dataset_path
                ),
                None,
            )
            if (
                not isinstance(dataset_entry, dict)
                or dataset_entry.get("sha256") != dataset_digest
            ):
                continue
            report_path = bundle_dir / "validation-report.json"
            try:
                report_payload = (
                    json.loads(report_path.read_text(encoding="utf-8"))
                    if report_path.is_file()
                    else None
                )
            except (OSError, json.JSONDecodeError):
                report_payload = None
            if not isinstance(report_payload, dict):
                report_payload = None
            elif report_payload.get("state") in {
                "pilot-pass",
                "execution-approved",
                "measured-run-pass",
            }:
                if active_job is not None:
                    cached_capacity = active_job.get("prelaunch_capacity_check")
                    authorization_is_current = bool(
                        active_job.get("action") == "train"
                        and isinstance(cached_capacity, dict)
                        and active_job.get("bundle_dir") == str(bundle_dir)
                    )
                    authorization_status = (
                        "current" if authorization_is_current else "blocked"
                    )
                    authorization_error = (
                        None
                        if authorization_is_current
                        else "Pilot authorization is not re-probed while any Aptus GPU job is active."
                    )
                    authorization_capacity = (
                        cached_capacity
                        if isinstance(cached_capacity, dict)
                        and active_job.get("bundle_dir") == str(bundle_dir)
                        else None
                    )
                else:
                    authorization_status = "deferred"
                    authorization_error = "Deep pilot binding, checkpoint, environment, and current capacity authorization is performed atomically when full training is submitted. Bootstrap does not rehash large pilot artifacts."
                    authorization_capacity = None
                report_payload = decorate_validation_authorization(
                    report_payload,
                    status=authorization_status,
                    error=authorization_error,
                    capacity=authorization_capacity,
                )
            archive_path = Path(reference["archive_path"])
            archive_matches = bool(
                reference["archive_path"]
                and archive_path.is_file()
                and not archive_path.is_symlink()
                and isinstance(reference.get("archive_sha256"), str)
                and isinstance(reference.get("archive_size_bytes"), int)
                and not isinstance(reference.get("archive_size_bytes"), bool)
                and archive_path.stat().st_size == reference["archive_size_bytes"]
                and sha256_file(archive_path) == reference["archive_sha256"]
            )
            result["plan"] = {
                **plan_payload,
                **(
                    {
                        "project_id": current_revision["project_id"],
                        "project_revision_id": current_revision["revision_id"],
                    }
                    if reference.get("current_revision_bound") is True
                    and isinstance(current_revision, dict)
                    else {}
                ),
            }
            result["bundle"] = {
                "bundle_dir": str(bundle_dir),
                "archive_path": (str(archive_path) if archive_matches else None),
                "files": bundle_files(bundle_dir),
                "runtime_contract": (
                    plan_payload.get("recommended", {}).get("runtime_contract")
                    if isinstance(plan_payload.get("recommended"), dict)
                    else None
                ),
                "report": report_payload,
                **(
                    {
                        "project_id": current_revision["project_id"],
                        "project_revision_id": current_revision["revision_id"],
                    }
                    if reference.get("current_revision_bound") is True
                    and isinstance(current_revision, dict)
                    else {}
                ),
            }
            restored_bundle_dir = bundle_dir
            break

        if (
            active_job is not None
            and current_project is not None
            and restored_bundle_dir is not None
            and isinstance(active_job.get("bundle_dir"), str)
            and Path(active_job["bundle_dir"]).resolve() == restored_bundle_dir
        ):
            result["job"] = active_job
        if active_job is None and restored_bundle_dir is not None:
            matching_job = next(
                (
                    item
                    for item in jobs
                    if item.get("bundle_dir") == str(restored_bundle_dir)
                    and job_matches_current_revision(item)
                ),
                None,
            )
            if matching_job is not None:
                result["job"] = matching_job
        if "plan" not in result and isinstance(current_revision, dict):
            plan_snapshot = current_revision.get("plan_snapshot")
            if isinstance(plan_snapshot, dict):
                if plan_snapshot.get("schema_version") != SCHEMA_VERSION:
                    result.setdefault(
                        "replan_required",
                        _replan_required_payload(
                            plan_snapshot,
                            source="project-revision",
                            project_id=current_revision["project_id"],
                            project_revision_id=current_revision["revision_id"],
                        ),
                    )
                elif not validate_plan_payload(plan_snapshot, verify_dataset=False):
                    result["plan"] = {
                        **plan_snapshot,
                        "project_id": current_revision["project_id"],
                        "project_revision_id": current_revision["revision_id"],
                    }
        return result

    @app.get("/api/v1/hardware", response_model=HardwareProbeResponse)
    def inspect_hardware() -> dict[str, Any]:
        try:
            with context.jobs.validation_guard():
                hardware = probe_local_hardware()
        except ActiveJobError as error:
            raise HTTPException(
                status_code=409,
                detail={"error": "active_job_conflict", "message": str(error)},
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            return {
                "status": "unavailable",
                "scope": "server-local",
                "error": str(error),
                "manual_facts_supported": True,
            }
        return {
            "status": "ok",
            "scope": "server-local",
            "hardware": to_primitive(hardware),
        }

    @app.get("/api/v1/runtimes", response_model=RuntimeInventoryResponse)
    def inspect_runtimes() -> dict[str, Any]:
        inventory = runtime_inventory(environment=context.jobs.runtime_environment)
        return {
            **inventory,
            "interpreters": inventory.get("interpreters", []),
            "configuration": inventory.get(
                "configuration",
                {
                    runtime_id: runtime_environment_key(runtime_id)
                    for runtime_id in (
                        "mlx-lm",
                        "pytorch-mps",
                        "transformers-peft-cuda",
                    )
                },
            ),
            "selected": dict(sorted(context.runtime_paths.items())),
        }

    @app.post("/api/v1/runtimes/configure", response_model=RuntimeConfiguredResponse)
    def configure_runtime(request: RuntimeConfigureRequest) -> dict[str, Any]:
        try:
            return context.configure_runtime(
                request.runtime_id,
                Path(request.interpreter_path).expanduser(),
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "runtime_configuration_invalid",
                    "details": str(error),
                },
            ) from error

    @app.get("/api/v1/platform", response_model=PlatformResponse)
    def inspect_platform() -> dict[str, Any]:
        try:
            profile = probe_apple_platform()
        except ValueError as error:
            return {
                "status": "unsupported",
                "platform": None,
                "error": str(error),
            }
        return {"status": "ok", "platform": profile.to_dict()}

    @app.get("/api/v1/inference/services", response_model=InferenceServicesResponse)
    def inspect_inference_services() -> dict[str, Any]:
        return {
            "status": "ok",
            "scope": "explicit-default-loopback-origins",
            "training_capability": False,
            "services": discover_local_inference_services(),
        }

    def inference_client(request: InferenceServiceRequest) -> Any:
        client_type = LMStudioClient if request.service == "lm-studio" else OMLXClient
        return client_type(
            endpoint=request.endpoint,
            timeout=request.timeout_seconds,
        )

    @app.post("/api/v1/inference/models", response_model=InferenceModelsResponse)
    def list_inference_models(request: InferenceServiceRequest) -> dict[str, Any]:
        return inference_client(request).list_models()

    @app.post("/api/v1/inference/generate", response_model=InferenceGenerateResponse)
    def generate_with_inference_service(
        request: InferenceGenerateRequest,
    ) -> dict[str, Any]:
        return inference_client(request).generate(
            model=request.model,
            messages=[item.model_dump() for item in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

    @app.post("/api/v1/models/inspect", response_model=ModelInspectionResponse)
    def inspect_model(request: ModelInspectRequest) -> dict[str, Any]:
        return inspect_huggingface_model(
            request.model_id,
            request.revision,
            timeout=request.timeout_seconds,
        )

    @app.post("/api/v1/profile", response_model=ProfileResponse)
    def profile(request: ProfileRequest) -> dict[str, Any]:
        return to_primitive(
            profile_dataset(
                Path(request.dataset_path),
                sample_limit=request.sample_limit,
                sequence_length=request.sequence_length,
            )
        )

    @app.post(
        "/api/v1/evaluations/contracts",
        response_model=EvaluationContractResponse,
    )
    def create_evaluation_contract(
        request: EvaluationContractRequest,
    ) -> dict[str, Any]:
        from .evaluation import build_evaluation_contract

        return build_evaluation_contract(
            dataset_path=Path(request.dataset_path),
            claim=request.claim,
            threshold=request.threshold,
            metric=request.metric,
            gold_field=request.gold_field,
            id_field=request.id_field,
            casefold=request.casefold,
            plan_id=request.plan_id,
            candidate_id=request.candidate_id,
            job_id=request.job_id,
            export_digest=request.export_digest,
            export_kind=request.export_kind,
        ).to_primitive()

    @app.post("/api/v1/evaluations", response_model=EvaluationResultResponse)
    def score_evaluation(request: EvaluationScoreRequest) -> dict[str, Any]:
        from .evaluation import (
            evaluate_predictions,
            evaluation_contract_from_primitive,
        )

        contract = evaluation_contract_from_primitive(
            request.contract.model_dump(mode="json")
        )
        return evaluate_predictions(
            contract,
            Path(request.gold_path),
            Path(request.predictions_path),
            expected_export_digest=request.export_digest,
        ).to_primitive()

    @app.post(
        "/api/v1/plan",
        response_model=TrainingPlanResponse,
        responses={
            422: {"model": NoFeasiblePlanResponse | ErrorResponse},
        },
    )
    def plan(request: PlanRequest) -> dict[str, Any]:
        try:
            if request.hardware.discovery == "local-scan":
                with context.jobs.validation_guard():
                    result = _build_plan(request)
            else:
                result = _build_plan(request)
        except ActiveJobError as error:
            raise HTTPException(
                status_code=409,
                detail={"error": "active_job_conflict", "message": str(error)},
            ) from error
        context.save_plan(result)
        plan_payload = to_primitive(result)
        project_id, revision = context.projects.record_plan(
            project_id=request.project_id,
            project_name=(
                request.project_name or f"{request.model.model_id} fine-tuning"
            ),
            facts=request.model_dump(
                mode="json", exclude={"project_id", "project_name"}
            ),
            plan=plan_payload,
        )
        return attach_correction(
            {
                **plan_payload,
                "project_id": project_id,
                "project_revision_id": revision["revision_id"],
            },
            build_plan_correction(result),
        )

    @app.get("/api/v1/plans/{plan_id}", response_model=TrainingPlanResponse)
    def get_plan(plan_id: str) -> dict[str, Any]:
        try:
            result = context.load_plan(plan_id)
        except StaleModelPolicyError as error:
            raise HTTPException(
                status_code=409,
                detail=_stale_policy_error_payload(
                    plan_id=plan_id,
                    message=str(error),
                ),
            ) from error
        except UnsupportedPlanSchemaError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "replan_required",
                    "plan_id": plan_id,
                    "found_schema": error.found_schema,
                    "required_schema": error.required_schema,
                    "message": str(error),
                },
            ) from error
        if result is None:
            raise HTTPException(
                status_code=404, detail={"error": "plan_not_found", "plan_id": plan_id}
            )
        return attach_correction(to_primitive(result), build_plan_correction(result))

    @app.post("/api/v1/plans/select", response_model=TrainingPlanResponse)
    def select_plan_candidate(request: SelectCandidateRequest) -> dict[str, Any]:
        source_revision = require_current_project_revision(
            project_id=request.project_id,
            expected_revision_id=request.expected_project_revision_id,
            plan_id=request.plan_id,
        )
        try:
            source_plan = context.load_plan(request.plan_id)
        except (StaleModelPolicyError, UnsupportedPlanSchemaError) as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "replan_required",
                    "plan_id": request.plan_id,
                    "message": str(error),
                },
            ) from error
        if source_plan is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "plan_not_found", "plan_id": request.plan_id},
            )
        if source_revision.get("plan_snapshot") != to_primitive(source_plan):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "project_plan_snapshot_mismatch",
                    "plan_id": request.plan_id,
                    "project_id": request.project_id,
                },
            )
        try:
            selected_plan = select_candidate(source_plan, request.candidate_id)
        except ValueError as error:
            raise HTTPException(
                status_code=409,
                detail={"error": "candidate_selection_rejected", "message": str(error)},
            ) from error
        facts = source_revision.get("facts")
        if not isinstance(facts, Mapping):
            raise HTTPException(
                status_code=409,
                detail={"error": "project_revision_facts_missing"},
            )
        selected_payload = to_primitive(selected_plan)
        context.save_plan(selected_plan)
        revision = create_current_project_revision(
            project_id=request.project_id,
            expected_revision_id=request.expected_project_revision_id,
            reason="candidate-selected",
            facts=facts,
            plan_id=selected_plan.plan_id,
            plan_snapshot=selected_payload,
            selected_candidate_id=selected_plan.recommended.candidate_id,
            bundle={},
            validation={},
            job_ids=[],
        )
        return attach_correction(
            {
                **selected_payload,
                "project_id": request.project_id,
                "project_revision_id": revision["revision_id"],
            },
            build_plan_correction(selected_plan),
        )

    @app.post("/api/v1/compile", response_model=CompileResponse)
    def compile_artifacts(request: CompileRequest) -> dict[str, Any]:
        source_revision = require_current_project_revision(
            project_id=request.project_id,
            expected_revision_id=request.expected_project_revision_id,
            plan_id=request.plan_id,
        )
        try:
            plan_value = context.load_plan(request.plan_id)
        except StaleModelPolicyError as error:
            raise HTTPException(
                status_code=409,
                detail=_stale_policy_error_payload(
                    plan_id=request.plan_id,
                    project_id=request.project_id,
                    project_revision_id=request.expected_project_revision_id,
                    message=str(error),
                ),
            ) from error
        except UnsupportedPlanSchemaError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "replan_required",
                    "plan_id": request.plan_id,
                    "found_schema": error.found_schema,
                    "required_schema": error.required_schema,
                    "message": str(error),
                },
            ) from error
        if plan_value is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "plan_not_found", "plan_id": request.plan_id},
            )
        plan_payload = to_primitive(plan_value)
        if source_revision.get("plan_snapshot") != plan_payload:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "project_plan_snapshot_mismatch",
                    "project_id": request.project_id,
                    "expected_project_revision_id": request.expected_project_revision_id,
                    "plan_id": request.plan_id,
                },
            )
        bundle_dir = Path(request.output_dir).resolve()
        archive_target = bundle_dir.with_suffix(".zip")
        if archive_target.exists():
            raise FileExistsError(f"Archive output already exists: {archive_target}")
        bundle_preexisted = bundle_dir.is_dir() and not bundle_dir.is_symlink()
        bundle_preexisting_mode = (
            stat.S_IMODE(bundle_dir.stat().st_mode) if bundle_preexisted else None
        )
        report = generate_bundle(plan_value, bundle_dir)
        archive = create_bundle_archive(bundle_dir)
        response = {
            "bundle_dir": str(bundle_dir),
            "archive_path": str(archive),
            "files": bundle_files(bundle_dir),
            "runtime_contract": to_primitive(
                getattr(plan_value.recommended, "runtime_contract", None)
            ),
            "report": to_primitive(report),
        }
        bundle_metadata = bundle_dir.stat()
        archive_metadata = archive.stat()
        artifact_fingerprint = response["report"].get("artifact_fingerprint")
        archive_fingerprint = sha256_file(archive)
        archive_size_bytes = archive_metadata.st_size
        try:
            revision = create_current_project_revision(
                project_id=request.project_id,
                expected_revision_id=request.expected_project_revision_id,
                reason="bundle-compiled",
                plan_id=request.plan_id,
                bundle={
                    "bundle_dir": str(bundle_dir),
                    "archive_path": str(archive),
                    "files": response["files"],
                    "artifact_fingerprint": artifact_fingerprint,
                    "archive_sha256": archive_fingerprint,
                    "archive_size_bytes": archive_size_bytes,
                },
                validation={
                    "state": response["report"].get("state"),
                    "report": response["report"],
                    "report_path": str(bundle_dir / "validation-report.json"),
                },
            )
        except HTTPException as error:
            detail = error.detail
            if (
                error.status_code == 409
                and isinstance(detail, dict)
                and detail.get("error") == "project_revision_conflict"
            ):
                archive_is_generated = False
                if archive.is_file() and not archive.is_symlink():
                    current_archive = archive.stat()
                    archive_is_generated = bool(
                        current_archive.st_dev == archive_metadata.st_dev
                        and current_archive.st_ino == archive_metadata.st_ino
                        and current_archive.st_size == archive_size_bytes
                        and sha256_file(archive) == archive_fingerprint
                    )
                if archive_is_generated:
                    archive.unlink()
                bundle_is_generated = False
                if bundle_dir.is_dir() and not bundle_dir.is_symlink():
                    current_bundle = bundle_dir.stat()
                    manifest_path = bundle_dir / "bundle-manifest.json"
                    bundle_is_generated = bool(
                        current_bundle.st_dev == bundle_metadata.st_dev
                        and current_bundle.st_ino == bundle_metadata.st_ino
                        and isinstance(artifact_fingerprint, str)
                        and manifest_path.is_file()
                        and not manifest_path.is_symlink()
                        and sha256_file(manifest_path) == artifact_fingerprint
                    )
                if bundle_is_generated:
                    shutil.rmtree(bundle_dir)
                    if bundle_preexisted:
                        bundle_dir.mkdir(parents=True, exist_ok=True)
                        if bundle_preexisting_mode is not None:
                            bundle_dir.chmod(bundle_preexisting_mode)
            raise
        response["project_id"] = revision["project_id"]
        response["project_revision_id"] = revision["revision_id"]
        context.save_bundle(bundle_dir, archive, plan_id=request.plan_id)
        return response

    @app.post(
        "/api/v1/validate",
        response_model=ValidationResponse,
        response_model_exclude_unset=True,
    )
    def validate(request: ValidateRequest) -> dict[str, Any]:
        if request.run and request.level in {
            "dependency",
            "model-data",
            "measured-preflight",
            "pilot",
        }:
            if not execution_enabled:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "desktop_execution_disabled",
                        "message": (
                            "This Aptus service was started with local execution "
                            "disabled. Enable execution for a compatible local runtime "
                            "or transfer the bundle to its target host."
                        ),
                    },
                )
            action = {
                "dependency": "dependency",
                "model-data": "model-data",
                "measured-preflight": "preflight",
                "pilot": "pilot",
            }[request.level]
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "runtime_validation_requires_job",
                    "message": "Runtime validation is cancellable job work; submit it through /api/v1/jobs.",
                    "suggested_action": action,
                },
            )
        bundle_dir = Path(request.bundle_dir).resolve()
        require_current_project_revision(
            project_id=request.project_id,
            expected_revision_id=request.expected_project_revision_id,
            bundle_dir=bundle_dir,
        )
        try:
            with context.jobs.validation_guard():
                require_current_project_revision(
                    project_id=request.project_id,
                    expected_revision_id=request.expected_project_revision_id,
                    bundle_dir=bundle_dir,
                )
                report = to_primitive(
                    validate_bundle(bundle_dir, level=request.level, run=request.run)
                )
                require_current_project_revision(
                    project_id=request.project_id,
                    expected_revision_id=request.expected_project_revision_id,
                    bundle_dir=bundle_dir,
                )
                if report.get("state") in {
                    "pilot-pass",
                    "execution-approved",
                    "measured-run-pass",
                }:
                    report = decorate_validation_authorization(
                        report,
                        status="deferred",
                        error=(
                            "Deep pilot binding, checkpoint, environment, and current capacity authorization is performed atomically when full training is submitted. Synchronous validation does not rehash large pilot artifacts."
                        ),
                        capacity=None,
                    )
        except ActiveJobError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "active_job_conflict",
                    "message": str(error),
                },
            ) from error
        revision = create_current_project_revision(
            project_id=request.project_id,
            expected_revision_id=request.expected_project_revision_id,
            reason="bundle-validated",
            validation={
                "state": report.get("state"),
                "report": dict(report),
                "report_path": str(bundle_dir / "validation-report.json"),
            },
        )
        report = {
            **report,
            "project_id": revision["project_id"],
            "project_revision_id": revision["revision_id"],
        }
        return report

    @app.post("/api/v1/jobs", response_model=JobResponse)
    def create_job(request: JobRequest) -> dict[str, Any]:
        if not execution_enabled:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "desktop_execution_disabled",
                    "message": (
                        "This Aptus service was started with local execution disabled. "
                        "Enable execution for a compatible local runtime or transfer "
                        "the bundle to its target host."
                    ),
                },
            )
        bundle_dir = Path(request.bundle_dir).resolve()
        require_current_project_revision(
            project_id=request.project_id,
            expected_revision_id=request.expected_project_revision_id,
            bundle_dir=bundle_dir,
        )

        def persist_job_revision(job_record: Mapping[str, Any]) -> dict[str, str]:
            require_current_project_revision(
                project_id=request.project_id,
                expected_revision_id=request.expected_project_revision_id,
                bundle_dir=bundle_dir,
            )
            revision = create_current_project_revision(
                project_id=request.project_id,
                expected_revision_id=request.expected_project_revision_id,
                reason="job-submitted",
                job_ids=[str(job_record["id"])],
            )
            return {
                "project_id": str(revision["project_id"]),
                "project_revision_id": str(revision["revision_id"]),
            }

        try:
            return context.jobs.submit(
                bundle_dir,
                action=request.action,
                confirm_full_train=request.confirm_full_train,
                admission_check=lambda: require_current_project_revision(
                    project_id=request.project_id,
                    expected_revision_id=request.expected_project_revision_id,
                    bundle_dir=bundle_dir,
                ),
                expected_artifact_fingerprint=str(
                    require_current_project_revision(
                        project_id=request.project_id,
                        expected_revision_id=request.expected_project_revision_id,
                        bundle_dir=bundle_dir,
                    )["bundle"]["artifact_fingerprint"]
                ),
                before_start=persist_job_revision,
            )
        except StaleModelPolicyError as error:
            plan_id: str | None = None
            try:
                plan_value = json.loads(
                    (bundle_dir / "plan.json").read_text(encoding="utf-8")
                )
                if isinstance(plan_value, dict) and isinstance(
                    plan_value.get("plan_id"), str
                ):
                    plan_id = plan_value["plan_id"]
            except (OSError, RecursionError, ValueError):
                pass
            raise HTTPException(
                status_code=409,
                detail=_stale_policy_error_payload(
                    message=str(error),
                    plan_id=plan_id,
                    project_id=request.project_id,
                    project_revision_id=request.expected_project_revision_id,
                ),
            ) from error
        except JobPrerequisiteError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": error.code,
                    "message": str(error),
                    "action": error.action,
                    "required_state": error.required_state,
                    "current_state": error.current_state,
                    "reason": error.reason,
                },
            ) from error
        except RuntimeError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "runtime_unavailable",
                    "message": str(error),
                },
            ) from error
        except ActiveJobError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "active_job_conflict",
                    "message": str(error),
                },
            ) from error

    @app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return context.jobs.get(job_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail={"error": "job_not_found", "job_id": job_id}
            ) from None

    @app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobResponse)
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return context.jobs.cancel(job_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail={"error": "job_not_found", "job_id": job_id}
            ) from None

    @app.get("/api/v1/jobs", response_model=list[JobResponse])
    def list_jobs() -> list[dict[str, Any]]:
        return context.jobs.list()

    @app.post("/api/v1/projects", status_code=201, response_model=ProjectResponse)
    def create_project(request: ProjectCreateRequest) -> dict[str, Any]:
        project = context.projects.create(request.name)
        return context.projects.get(project["project_id"])

    @app.get("/api/v1/projects", response_model=list[ProjectSummaryResponse])
    def list_projects() -> list[dict[str, Any]]:
        return context.projects.list()

    @app.get("/api/v1/projects/{project_id}", response_model=ProjectResponse)
    def get_project(project_id: str) -> dict[str, Any]:
        try:
            return context.projects.get(project_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail={"error": "project_not_found", "project_id": project_id},
            ) from None

    @app.get(
        "/api/v1/projects/{project_id}/revisions",
        response_model=list[ProjectRevisionSummary],
    )
    def list_project_revisions(project_id: str) -> list[dict[str, Any]]:
        try:
            return context.projects.history(project_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail={"error": "project_not_found", "project_id": project_id},
            ) from None

    @app.get(
        "/api/v1/projects/{project_id}/revisions/{revision_id}",
        response_model=ProjectRevisionResponse,
    )
    def get_project_revision(project_id: str, revision_id: str) -> dict[str, Any]:
        try:
            return context.projects.revision(project_id, revision_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "project_revision_not_found",
                    "project_id": project_id,
                    "revision_id": revision_id,
                },
            ) from None

    @app.post(
        "/api/v1/projects/{project_id}/recover",
        response_model=ProjectRecoveryResponse,
    )
    def recover_project_revision(
        project_id: str, request: ProjectRecoverRequest
    ) -> dict[str, Any]:
        try:
            source_revision = context.projects.revision(project_id, request.revision_id)
            plan_snapshot = source_revision.get("plan_snapshot")
            if (
                isinstance(plan_snapshot, Mapping)
                and plan_snapshot.get("schema_version") == SCHEMA_VERSION
            ):
                require_current_model_policy(plan_snapshot)
                validation_errors = validate_plan_payload(
                    plan_snapshot, verify_dataset=False
                )
                if validation_errors:
                    raise ValueError(
                        "Saved project plan failed the current executable contract: "
                        + " ".join(validation_errors)
                    )
            revision = context.projects.recover(project_id, request.revision_id)
        except StaleModelPolicyError as error:
            raise HTTPException(
                status_code=409,
                detail=_stale_policy_error_payload(
                    project_id=project_id,
                    project_revision_id=request.revision_id,
                    message=str(error),
                ),
            ) from error
        except UnsupportedPlanSchemaError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "replan_required",
                    "project_id": project_id,
                    "project_revision_id": request.revision_id,
                    "found_schema": error.found_schema,
                    "required_schema": error.required_schema,
                    "message": str(error),
                },
            ) from error
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "project_revision_not_found",
                    "project_id": project_id,
                    "revision_id": request.revision_id,
                },
            ) from None
        return {
            "status": "recovered",
            "project_id": project_id,
            "revision": revision,
            "training_authorization_current": False,
        }

    resolved_static = _resolve_static_dir(static_dir)
    if resolved_static is not None:

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> Any:
            if full_path.startswith("api/"):
                return JSONResponse(status_code=404, content={"error": "not_found"})
            requested = (resolved_static / full_path).resolve()
            if resolved_static in requested.parents and requested.is_file():
                return FileResponse(requested)
            return FileResponse(resolved_static / "index.html")

    app.state.aptus = context
    return app


def _resolve_static_dir(explicit: Path | None) -> Path | None:
    if explicit is not None:
        resolved = explicit.resolve()
        if not (resolved / "index.html").is_file():
            raise ValueError(f"Workbench directory must contain index.html: {resolved}")
        return resolved

    candidates = (
        Path(__file__).resolve().parent / "_web",
        Path(__file__).resolve().parents[2] / "web" / "dist",
        Path(sys.prefix) / "share" / "aptus" / "web",
    )
    return next(
        (
            candidate.resolve()
            for candidate in candidates
            if (candidate / "index.html").is_file()
        ),
        None,
    )
