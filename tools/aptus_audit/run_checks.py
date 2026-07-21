from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import venv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .checks import run_check
from .inventory import write_jsonl


TYPESCRIPT_VERSION = "7.0.2"


def build_check_specs(
    *,
    repository_root: Path,
    legacy_copy: Path,
    node_workspace: Path,
    python_executable: Path | None = None,
) -> list[dict[str, Any]]:
    python_executable = python_executable or Path(sys.executable)
    generator_path = legacy_copy / "src/python/script_generator_v2.py"
    generator_probe = (
        "import importlib.util; "
        f"path = {str(generator_path)!r}; "
        "spec = importlib.util.spec_from_file_location('legacy_generator_v2', path); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "module.ScriptGenerator()"
    )
    generator_salvage_probe = (
        "import importlib.util, sys, types; "
        "fake_jinja = types.ModuleType('jinja2'); "
        "fake_jinja.Template = type('Template', (), {}); "
        "sys.modules['jinja2'] = fake_jinja; "
        f"path = {str(generator_path)!r}; "
        "spec = importlib.util.spec_from_file_location('legacy_generator_v2', path); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "module.ScriptGenerator()"
    )
    resource_scanner_path = legacy_copy / "src/python/resource_scanner.py"
    resource_scanner_probe = (
        "import importlib.util; "
        f"path = {str(resource_scanner_path)!r}; "
        "spec = importlib.util.spec_from_file_location('legacy_resource_scanner', path); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "data = module.ResourceInfo().to_dict(); "
        "required = {'system', 'cpu_count', 'total_memory_gb', 'gpu_info', 'usable_gpu_memory_gb'}; "
        "assert required.issubset(data); "
        "print(f\"resource scan ok: system={data['system']}, gpu_count={len(data['gpu_info'])}\")"
    )
    resource_scanner_salvage_probe = (
        "import importlib.util, sys, types; "
        "fake_psutil = types.ModuleType('psutil'); "
        "fake_psutil.cpu_count = lambda logical=True: 8 if logical else 4; "
        "fake_psutil.virtual_memory = lambda: types.SimpleNamespace(total=16 * 1024**3); "
        "sys.modules['psutil'] = fake_psutil; "
        f"path = {str(resource_scanner_path)!r}; "
        "spec = importlib.util.spec_from_file_location('legacy_resource_scanner', path); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "data = module.ResourceInfo().to_dict(); "
        "required = {'system', 'cpu_count', 'total_memory_gb', 'gpu_info', 'usable_gpu_memory_gb'}; "
        "assert required.issubset(data); "
        "print(f\"resource salvage probe ok: system={data['system']}, gpu_count={len(data['gpu_info'])}\")"
    )

    return [
        {
            "check_id": "node-server-parse",
            "command": ["node", "--check", str(legacy_copy / "server.js")],
            "cwd": legacy_copy,
            "timeout_seconds": 20,
            "registry_network": False,
        },
        {
            "check_id": "typescript-project-check",
            "command": [
                "npx",
                "--yes",
                "--package",
                f"typescript@{TYPESCRIPT_VERSION}",
                "tsc",
                "--project",
                str(node_workspace / "tsconfig.audit.json"),
                "--pretty",
                "false",
            ],
            "cwd": node_workspace,
            "timeout_seconds": 120,
            "registry_network": True,
        },
        {
            "check_id": "node-lock-resolution",
            "command": [
                "npm",
                "install",
                "--package-lock-only",
                "--ignore-scripts",
                "--no-audit",
                "--no-fund",
            ],
            "cwd": node_workspace,
            "timeout_seconds": 120,
            "registry_network": True,
        },
        {
            "check_id": "python-requirements-resolution",
            "command": [
                str(python_executable),
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--ignore-installed",
                "--only-binary=:all:",
                "--disable-pip-version-check",
                "-r",
                str(legacy_copy / "requirements_v2.txt"),
            ],
            "cwd": legacy_copy,
            "timeout_seconds": 180,
            "registry_network": True,
        },
        {
            "check_id": "python-test-collection",
            "command": [
                str(python_executable),
                "-I",
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                str(legacy_copy / "tests"),
            ],
            "cwd": legacy_copy,
            "timeout_seconds": 60,
            "registry_network": False,
            "requires": ["python-requirements-resolution"],
            "execution_policy": "blocked_without_installed_environment",
        },
        {
            "check_id": "python-resource-scanner-smoke",
            "command": [
                str(python_executable),
                "-I",
                "-c",
                resource_scanner_probe,
            ],
            "cwd": legacy_copy,
            "timeout_seconds": 30,
            "registry_network": False,
        },
        {
            "check_id": "python-resource-scanner-salvage-probe",
            "command": [
                str(python_executable),
                "-I",
                "-c",
                resource_scanner_salvage_probe,
            ],
            "cwd": legacy_copy,
            "timeout_seconds": 30,
            "registry_network": False,
        },
        {
            "check_id": "python-v2-generator-construction",
            "command": [str(python_executable), "-I", "-c", generator_probe],
            "cwd": legacy_copy,
            "timeout_seconds": 30,
            "registry_network": False,
        },
        {
            "check_id": "python-v2-generator-salvage-probe",
            "command": [
                str(python_executable),
                "-I",
                "-c",
                generator_salvage_probe,
            ],
            "cwd": legacy_copy,
            "timeout_seconds": 30,
            "registry_network": False,
        },
    ]


def _write_typescript_config(legacy_copy: Path, node_workspace: Path) -> None:
    files = [
        str(path)
        for path in sorted(legacy_copy.rglob("*.ts"), key=lambda item: item.as_posix())
    ]
    config = {
        "compilerOptions": {
            "allowJs": False,
            "module": "NodeNext",
            "moduleResolution": "NodeNext",
            "noEmit": True,
            "skipLibCheck": True,
            "strict": False,
            "target": "ES2022",
        },
        "files": files,
    }
    (node_workspace / "tsconfig.audit.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )


def _blocked_record(
    spec: dict[str, Any],
    *,
    reason: str,
    dependency_statuses: dict[str, str],
) -> dict[str, Any]:
    empty_digest = hashlib.sha256(b"").hexdigest()
    return {
        "check_id": spec["check_id"],
        "safety_class": "blocked",
        "command": spec["command"],
        "cwd": str(spec["cwd"].resolve()),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "duration_ms": 0,
        "timeout_seconds": spec["timeout_seconds"],
        "timed_out": False,
        "exit_code": None,
        "status": "blocked",
        "block_reason": reason,
        "dependency_statuses": dependency_statuses,
        "inherited_environment_keys": [],
        "stdout_sha256": empty_digest,
        "stderr_sha256": empty_digest,
        "stdout_preview": "",
        "stderr_preview": "",
    }


def execute_legacy_checks(
    repository_root: Path,
    output_path: Path,
    *,
    allow_host_subprocesses: bool = False,
) -> list[dict[str, Any]]:
    if not allow_host_subprocesses:
        raise RuntimeError(
            "The audit runner does not enforce OS-level isolation. "
            "Run it only inside an externally sandboxed environment and pass "
            "allow_host_subprocesses=True after reviewing the command plan."
        )

    repository_root = repository_root.resolve(strict=True)
    legacy_source = repository_root / "HyperTune"
    if not legacy_source.is_dir():
        raise FileNotFoundError(f"Legacy source is unavailable: {legacy_source}")
    output_path = output_path.resolve()
    if output_path == legacy_source or output_path.is_relative_to(legacy_source):
        raise ValueError("Audit output must be outside the legacy source tree.")
    sandbox_root = Path(tempfile.mkdtemp(prefix="aptus-legacy-audit-"))
    try:
        legacy_copy = sandbox_root / "legacy"
        node_workspace = sandbox_root / "node"
        home = sandbox_root / "home"
        virtual_environment = sandbox_root / "python-venv"
        node_workspace.mkdir()
        home.mkdir()
        shutil.copytree(legacy_source, legacy_copy)
        shutil.copy2(legacy_source / "package.json", node_workspace)
        _write_typescript_config(legacy_copy, node_workspace)
        venv.EnvBuilder(
            with_pip=True,
            clear=True,
            system_site_packages=False,
        ).create(virtual_environment)
        python_executable = virtual_environment / "bin/python"

        environment = {
            "HOME": str(home),
            "NPM_CONFIG_CACHE": str(sandbox_root / "npm-cache"),
            "NPM_CONFIG_IGNORE_SCRIPTS": "true",
            "PIP_CACHE_DIR": str(sandbox_root / "pip-cache"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        results = []
        results_by_id: dict[str, dict[str, Any]] = {}
        for spec in build_check_specs(
            repository_root=repository_root,
            legacy_copy=legacy_copy,
            node_workspace=node_workspace,
            python_executable=python_executable,
        ):
            dependencies = spec.get("requires", [])
            dependency_statuses = {
                dependency: results_by_id.get(dependency, {}).get(
                    "status", "not_run"
                )
                for dependency in dependencies
            }
            if spec.get("execution_policy") == "blocked_without_installed_environment":
                record = _blocked_record(
                    spec,
                    reason=(
                        "Dependency resolution is dry-run only; no project "
                        "environment was installed, so importing the test suite "
                        "would measure the host rather than the legacy project."
                    ),
                    dependency_statuses=dependency_statuses,
                )
            elif any(status != "passed" for status in dependency_statuses.values()):
                record = _blocked_record(
                    spec,
                    reason="A prerequisite check did not pass.",
                    dependency_statuses=dependency_statuses,
                )
            else:
                record = run_check(
                    check_id=spec["check_id"],
                    command=spec["command"],
                    cwd=spec["cwd"],
                    timeout_seconds=spec["timeout_seconds"],
                    environment=environment,
                    inherit_proxy=spec["registry_network"],
                )
            record["sandbox"] = {
                "kind": "disposable-copy-with-host-subprocesses",
                "os_isolation_enforced_by_runner": False,
                "host_subprocesses_acknowledged": True,
                "credential_isolation": (
                    "Common credential variables are excluded. Registry checks "
                    "may receive host proxy variables, which can contain "
                    "credentials; legacy-import checks do not receive them."
                ),
                "lifecycle_scripts_disabled": True,
                "network_policy": (
                    "Not enforced by this runner. Registry proxy variables are "
                    "forwarded only when registry_network is true; external "
                    "sandbox policy must provide actual egress control."
                ),
                "registry_network_requested": spec["registry_network"],
                "source_tree": str(legacy_copy),
            }
            results.append(record)
            results_by_id[spec["check_id"]] = record

        write_jsonl(output_path, results)
        return results
    finally:
        shutil.rmtree(sandbox_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded legacy checks. This command does not provide "
            "OS-level sandboxing."
        )
    )
    parser.add_argument(
        "--allow-host-subprocesses",
        action="store_true",
        help=(
            "Acknowledge that subprocess filesystem/network isolation must be "
            "provided by the surrounding environment."
        ),
    )
    arguments = parser.parse_args()
    if not arguments.allow_host_subprocesses:
        parser.error(
            "--allow-host-subprocesses is required; run only inside an "
            "externally sandboxed environment."
        )

    repository_root = Path(__file__).resolve().parents[2]
    output_path = (
        repository_root
        / "docs/audits/aptus-legacy/sandbox-results.jsonl"
    )
    results = execute_legacy_checks(
        repository_root,
        output_path,
        allow_host_subprocesses=True,
    )
    passed = sum(result["status"] == "passed" for result in results)
    failed = sum(result["status"] == "failed" for result in results)
    blocked = sum(result["status"] == "blocked" for result in results)
    timed_out = sum(result["status"] == "timed_out" for result in results)
    print(
        f"Completed {len(results)} checks: {passed} passed, {failed} failed, "
        f"{blocked} blocked, {timed_out} timed out."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
