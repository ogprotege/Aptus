from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPENAPI_PATH = REPOSITORY_ROOT / "docs" / "reference" / "openapi.v1.json"
SWIFT_CLIENT_PATH = (
    REPOSITORY_ROOT / "desktop" / "macos" / "Sources" / "DesktopBackendClient.swift"
)
RUNTIME_SCHEMA_VERSION = "aptus.runtime-inventory.v1"
ENDPOINTS = {
    "runtimeConfigurationPath": ("post", "/api/v1/runtimes/configure"),
    "runtimeInventoryPath": ("get", "/api/v1/runtimes"),
    "platformPath": ("get", "/api/v1/platform"),
}
RUNTIME_REQUIRED_FIELDS = {
    "schema_version",
    "interpreters",
    "available",
    "compatible",
    "configuration",
    "selected",
}
SWIFT_RUNTIME_FIELDS = {
    "schema_version",
    "interpreters",
    "compatible",
    "selected",
}


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _resolve_schema(document: dict[str, Any], value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("OpenAPI response schema is not an object.")
    reference = value.get("$ref")
    if reference is None:
        return value
    if not isinstance(reference, str) or not reference.startswith("#/"):
        raise ValueError(f"Unsupported OpenAPI schema reference: {reference!r}")
    current: Any = document
    for component in reference[2:].split("/"):
        if not isinstance(current, dict) or component not in current:
            raise ValueError(f"Unresolvable OpenAPI schema reference: {reference}")
        current = current[component]
    if not isinstance(current, dict):
        raise ValueError(f"OpenAPI schema reference is not an object: {reference}")
    return current


def verify_client_contracts(
    *,
    openapi_path: Path = OPENAPI_PATH,
    swift_client_path: Path = SWIFT_CLIENT_PATH,
) -> list[str]:
    errors: list[str] = []
    try:
        openapi = _json_object(openapi_path)
        swift_source = swift_client_path.read_text(encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [str(error)]

    constants = dict(
        re.findall(
            r'static\s+let\s+(\w+Path)\s*=\s*"([^"]+)"',
            swift_source,
        )
    )
    paths = openapi.get("paths")
    if not isinstance(paths, dict):
        return ["OpenAPI document has no paths object."]

    for constant, (method, expected_path) in ENDPOINTS.items():
        actual_path = constants.get(constant)
        if actual_path != expected_path:
            errors.append(
                f"Swift {constant} is {actual_path!r}; expected {expected_path!r}."
            )
            continue
        operation = paths.get(actual_path)
        if not isinstance(operation, dict) or method not in operation:
            errors.append(
                f"OpenAPI does not define {method.upper()} {actual_path} used by Swift."
            )

    try:
        runtime_response = paths[ENDPOINTS["runtimeInventoryPath"][1]]["get"][
            "responses"
        ]["200"]["content"]["application/json"]["schema"]
        runtime_schema = _resolve_schema(openapi, runtime_response)
    except (KeyError, TypeError, ValueError) as error:
        errors.append(
            f"Could not resolve the runtime inventory response schema: {error}"
        )
        return errors

    required = runtime_schema.get("required")
    required_fields = set(required) if isinstance(required, list) else set()
    missing_required = RUNTIME_REQUIRED_FIELDS - required_fields
    if missing_required:
        errors.append(
            "RuntimeInventoryResponse must require: "
            + ", ".join(sorted(missing_required))
        )
    properties = runtime_schema.get("properties")
    schema_version = (
        properties.get("schema_version") if isinstance(properties, dict) else None
    )
    if not isinstance(schema_version, dict) or schema_version.get("const") != (
        RUNTIME_SCHEMA_VERSION
    ):
        errors.append(
            "RuntimeInventoryResponse schema_version must be the exact "
            f"{RUNTIME_SCHEMA_VERSION!r} constant."
        )

    if RUNTIME_SCHEMA_VERSION not in swift_source:
        errors.append(f"Swift does not fail closed on {RUNTIME_SCHEMA_VERSION!r}.")
    for field in sorted(SWIFT_RUNTIME_FIELDS):
        if f'["{field}"]' not in swift_source:
            errors.append(f"Swift does not consume required runtime field {field!r}.")
        if field not in required_fields:
            errors.append(
                f"Swift consumes runtime field {field!r}, but OpenAPI does not require it."
            )
    return errors


def main() -> int:
    errors = verify_client_contracts()
    if not errors:
        print("Swift desktop endpoints and runtime inventory match OpenAPI.")
        return 0
    for error in errors:
        print(f"client-contract error: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
