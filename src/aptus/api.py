from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .catalog import STACK_VERSIONS, TARGET_MODULES
from .domain import (
    Backend,
    Method,
    Objective,
    TrainingPlan,
    TrainingTarget,
    to_primitive,
    training_plan_from_primitive,
)
from .evidence import EVIDENCE_REGISTRY
from .plan_contract import validate_bundle_manifest, validate_plan_payload
from .planning import NoFeasiblePlanError, plan_training
from .profiling import (
    build_hardware_spec,
    build_model_spec,
    probe_local_hardware,
    profile_dataset,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
    task: str = "sft"
    evaluation_fraction: float = Field(default=0.1, ge=0, lt=1)
    packing: bool = False
    checkpoint_steps: int = Field(default=100, gt=0)


class ProfileRequest(StrictModel):
    dataset_path: str
    sample_limit: int | None = Field(default=512, gt=0)
    sequence_length: int | None = Field(default=None, gt=0)


class PlanRequest(StrictModel):
    model: ModelFactsRequest
    hardware: HardwareFactsRequest
    target: TargetRequest
    dataset_path: str
    sample_limit: int | None = Field(default=512, gt=0)


class CompileRequest(StrictModel):
    plan_id: str
    output_dir: str


class ValidateRequest(StrictModel):
    bundle_dir: str
    level: Literal[
        "contract", "static", "dependency", "model-data", "measured-preflight", "pilot"
    ] = "static"
    run: bool = False


class JobRequest(StrictModel):
    bundle_dir: str
    action: Literal["dependency", "model-data", "preflight", "pilot", "train"] = (
        "preflight"
    )
    confirm_full_train: bool = False


class ModelInspectRequest(StrictModel):
    model_id: str
    revision: str
    timeout_seconds: float = Field(default=10.0, gt=0, le=30)


class ApiContext:
    def __init__(self, state_dir: Path) -> None:
        from .execution import JobService

        self.state_dir = state_dir.resolve()
        self.plans_dir = self.state_dir / "plans"
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.current_bundle_path = self.state_dir / "current-bundle.json"
        self._state_lock = threading.RLock()
        self.jobs = JobService(state_dir / "jobs")

    def save_plan(self, plan: TrainingPlan) -> None:
        from .execution import _atomic_write_json

        path = self.plans_dir / f"{plan.plan_id}.json"
        with self._state_lock:
            _atomic_write_json(path, to_primitive(plan))

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
            value = json.loads(path.read_text(encoding="utf-8"))
        return training_plan_from_primitive(value)

    def save_bundle(self, bundle_dir: Path, archive_path: Path) -> None:
        from .execution import _atomic_write_json

        with self._state_lock:
            _atomic_write_json(
                self.current_bundle_path,
                {
                    "bundle_dir": str(bundle_dir.resolve()),
                    "archive_path": str(archive_path.resolve()),
                },
            )

    def load_bundle_reference(self) -> dict[str, str] | None:
        with self._state_lock:
            if not self.current_bundle_path.is_file():
                return None
            try:
                value = json.loads(self.current_bundle_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        if not isinstance(value, dict) or not isinstance(value.get("bundle_dir"), str):
            return None
        return {
            "bundle_dir": value["bundle_dir"],
            "archive_path": str(value.get("archive_path", "")),
        }


def _build_plan(request: PlanRequest) -> TrainingPlan:
    dataset = profile_dataset(
        Path(request.dataset_path),
        sample_limit=request.sample_limit,
        sequence_length=request.target.sequence_length,
    )
    model = build_model_spec(**request.model.model_dump())
    if request.hardware.discovery == "local-scan":
        hardware = probe_local_hardware(reserve_gib=request.hardware.reserve_gib)
    else:
        hardware = build_hardware_spec(
            **request.hardware.model_dump(exclude={"discovery"})
        )
    target = TrainingTarget(**request.target.model_dump())
    return plan_training(model=model, dataset=dataset, hardware=hardware, target=target)


def create_app(
    *,
    state_dir: Path | None = None,
    static_dir: Path | None = None,
    allowed_hosts: tuple[str, ...] | None = None,
) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.exceptions import HTTPException as FastApiHTTPException
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import FileResponse, JSONResponse
        from starlette.middleware.trustedhost import TrustedHostMiddleware
    except ImportError as error:  # pragma: no cover - exercised in deployment extras
        raise RuntimeError(
            "Install Aptus with the 'server' extra to use the API."
        ) from error

    from .generation import bundle_files, create_bundle_archive, generate_bundle
    from .inspection import inspect_huggingface_model
    from .execution import ActiveJobError, JobPrerequisiteError
    from .validation import validate_bundle

    context = ApiContext((state_dir or Path(".aptus-state")).resolve())
    app = FastAPI(title="Aptus API", version="0.2.0")
    trusted_hosts = set(allowed_hosts or ())
    trusted_hosts.update({"127.0.0.1", "localhost", "[::1]", "testserver"})
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=sorted(trusted_hosts),
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Any, error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "request_validation", "details": error.errors()},
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
        return JSONResponse(
            status_code=422,
            content={
                "error": "no_feasible_plan",
                "message": str(error),
                "candidates": [to_primitive(item) for item in error.candidates],
            },
        )

    @app.exception_handler(FastApiHTTPException)
    async def http_error(_request: Any, error: FastApiHTTPException) -> JSONResponse:
        detail = error.detail
        if isinstance(detail, dict):
            content = detail
        else:
            content = {"error": "http_error", "details": detail}
        return JSONResponse(status_code=error.status_code, content=content)

    @app.get("/api/v1/health")
    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.2.0"}

    @app.get("/api/v1/bootstrap")
    def bootstrap() -> dict[str, Any]:
        capabilities = {
            "backends": [Backend.CUDA.value],
            "known_backends": [item.value for item in Backend],
            "methods": [item.value for item in Method],
            "objectives": [item.value for item in Objective],
            "supported_execution_backend": Backend.CUDA.value,
            "supported_execution_backends": [Backend.CUDA.value],
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
            "version": "0.2.0",
            **capabilities,
            "stack_versions": STACK_VERSIONS,
            "evidence": [to_primitive(item) for item in EVIDENCE_REGISTRY.values()],
            "calibrated": False,
            "service": {"name": "aptus", "version": "0.2.0", "scope": "local-host"},
            "capabilities": capabilities,
            "defaults": {
                "sample_limit": 512,
                "reserve_gib": 2.0,
                "task": "sft",
                "packing": False,
            },
        }
        jobs = context.jobs.list()
        active_job = next(
            (
                item
                for item in jobs
                if item.get("state") in {"queued", "running", "cancelling"}
            ),
            None,
        )
        if active_job is not None:
            result["job"] = active_job

        saved_reference = context.load_bundle_reference()
        references: list[dict[str, str]] = []
        if active_job is not None and isinstance(active_job.get("bundle_dir"), str):
            active_archive = ""
            if (
                saved_reference is not None
                and Path(saved_reference["bundle_dir"]).resolve()
                == Path(active_job["bundle_dir"]).resolve()
            ):
                active_archive = saved_reference["archive_path"]
            references.append(
                {
                    "bundle_dir": active_job["bundle_dir"],
                    "archive_path": active_archive,
                }
            )
        if saved_reference is not None and all(
            item["bundle_dir"] != saved_reference["bundle_dir"] for item in references
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
            except (OSError, json.JSONDecodeError):
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
                    authorization = {
                        "current": bool(
                            active_job.get("action") == "train"
                            and isinstance(cached_capacity, dict)
                            and active_job.get("bundle_dir") == str(bundle_dir)
                        ),
                        "error": (
                            None
                            if active_job.get("action") == "train"
                            and isinstance(cached_capacity, dict)
                            and active_job.get("bundle_dir") == str(bundle_dir)
                            else "Pilot authorization is not re-probed while any Aptus GPU job is active."
                        ),
                        "capacity": (
                            cached_capacity
                            if isinstance(cached_capacity, dict)
                            and active_job.get("bundle_dir") == str(bundle_dir)
                            else None
                        ),
                    }
                else:
                    authorization = {
                        "current": False,
                        "error": (
                            "Deep pilot binding, checkpoint, environment, and current capacity authorization is performed atomically when full training is submitted. Bootstrap does not rehash large pilot artifacts."
                        ),
                        "capacity": None,
                    }
                report_payload = {
                    **report_payload,
                    "authorization_current": authorization["current"],
                    "authorization_error": authorization["error"],
                    "prelaunch_capacity_check": authorization["capacity"],
                }
            archive_path = Path(reference["archive_path"])
            result["plan"] = plan_payload
            result["bundle"] = {
                "bundle_dir": str(bundle_dir),
                "archive_path": (
                    str(archive_path)
                    if reference["archive_path"] and archive_path.is_file()
                    else None
                ),
                "files": bundle_files(bundle_dir),
                "report": report_payload,
            }
            restored_bundle_dir = bundle_dir
            break

        if active_job is None and restored_bundle_dir is not None:
            matching_job = next(
                (
                    item
                    for item in jobs
                    if item.get("bundle_dir") == str(restored_bundle_dir)
                ),
                None,
            )
            if matching_job is not None:
                result["job"] = matching_job
        return result

    @app.get("/api/v1/hardware")
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

    @app.post("/api/v1/models/inspect")
    def inspect_model(request: ModelInspectRequest) -> dict[str, Any]:
        return inspect_huggingface_model(
            request.model_id,
            request.revision,
            timeout=request.timeout_seconds,
        )

    @app.post("/api/v1/profile")
    def profile(request: ProfileRequest) -> dict[str, Any]:
        return to_primitive(
            profile_dataset(
                Path(request.dataset_path),
                sample_limit=request.sample_limit,
                sequence_length=request.sequence_length,
            )
        )

    @app.post("/api/v1/plan")
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
        return to_primitive(result)

    @app.get("/api/v1/plans/{plan_id}")
    def get_plan(plan_id: str) -> dict[str, Any]:
        result = context.load_plan(plan_id)
        if result is None:
            raise HTTPException(
                status_code=404, detail={"error": "plan_not_found", "plan_id": plan_id}
            )
        return to_primitive(result)

    @app.post("/api/v1/compile")
    def compile_artifacts(request: CompileRequest) -> dict[str, Any]:
        plan_value = context.load_plan(request.plan_id)
        if plan_value is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "plan_not_found", "plan_id": request.plan_id},
            )
        bundle_dir = Path(request.output_dir).resolve()
        archive_target = bundle_dir.with_suffix(".zip")
        if archive_target.exists():
            raise FileExistsError(f"Archive output already exists: {archive_target}")
        report = generate_bundle(plan_value, bundle_dir)
        archive = create_bundle_archive(bundle_dir)
        context.save_bundle(bundle_dir, archive)
        return {
            "bundle_dir": str(bundle_dir),
            "archive_path": str(archive),
            "files": bundle_files(bundle_dir),
            "report": to_primitive(report),
        }

    @app.post("/api/v1/validate")
    def validate(request: ValidateRequest) -> dict[str, Any]:
        if request.run and request.level in {
            "dependency",
            "model-data",
            "measured-preflight",
            "pilot",
        }:
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
        try:
            with context.jobs.validation_guard():
                report = to_primitive(
                    validate_bundle(bundle_dir, level=request.level, run=request.run)
                )
                if report.get("state") in {
                    "pilot-pass",
                    "execution-approved",
                    "measured-run-pass",
                }:
                    report.update(
                        authorization_current=False,
                        authorization_error=(
                            "Deep pilot binding, checkpoint, environment, and current capacity authorization is performed atomically when full training is submitted. Synchronous validation does not rehash large pilot artifacts."
                        ),
                        prelaunch_capacity_check=None,
                    )
        except ActiveJobError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "active_job_conflict",
                    "message": str(error),
                },
            ) from error
        return report

    @app.post("/api/v1/jobs")
    def create_job(request: JobRequest) -> dict[str, Any]:
        try:
            return context.jobs.submit(
                Path(request.bundle_dir),
                action=request.action,
                confirm_full_train=request.confirm_full_train,
            )
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
        except ActiveJobError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "active_job_conflict",
                    "message": str(error),
                },
            ) from error

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return context.jobs.get(job_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail={"error": "job_not_found", "job_id": job_id}
            ) from None

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return context.jobs.cancel(job_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail={"error": "job_not_found", "job_id": job_id}
            ) from None

    @app.get("/api/v1/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return context.jobs.list()

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
