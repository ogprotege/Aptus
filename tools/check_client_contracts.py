from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPENAPI_PATH = REPOSITORY_ROOT / "docs" / "reference" / "openapi.v1.json"
SWIFT_CLIENT_PATH = (
    REPOSITORY_ROOT / "desktop" / "macos" / "Sources" / "DesktopBackendClient.swift"
)
SWIFT_CONTROLLER_PATH = (
    REPOSITORY_ROOT / "desktop" / "macos" / "Sources" / "BackendController.swift"
)
SWIFT_MODELS_PATH = (
    REPOSITORY_ROOT / "desktop" / "macos" / "Sources" / "BackendModels.swift"
)


@dataclass(frozen=True)
class ResponseContract:
    schema_name: str
    constant_name: str
    method: str
    path: str
    decoder_source: str
    required_fields: frozenset[str]
    field_markers: dict[str, tuple[str, ...]]
    exact_values: dict[str, str]
    enum_values: dict[str, frozenset[str]]


RESPONSE_CONTRACTS = (
    ResponseContract(
        schema_name="HealthResponse",
        constant_name="healthPath",
        method="get",
        path="/api/v1/health",
        decoder_source="models",
        required_fields=frozenset({"status", "version", "api_contract_version"}),
        field_markers={
            "status": ("case status",),
            "version": ("case version",),
            "api_contract_version": ('"api_contract_version"',),
        },
        exact_values={"status": "ok", "api_contract_version": "aptus.api.v1"},
        enum_values={},
    ),
    ResponseContract(
        schema_name="RuntimeConfiguredResponse",
        constant_name="runtimeConfigurationPath",
        method="post",
        path="/api/v1/runtimes/configure",
        decoder_source="client",
        required_fields=frozenset(
            {"status", "runtime_id", "interpreter_path", "interpreter", "persisted"}
        ),
        field_markers={
            field: (f'["{field}"]',)
            for field in (
                "status",
                "runtime_id",
                "interpreter_path",
                "interpreter",
                "persisted",
            )
        },
        exact_values={"status": "ok"},
        enum_values={},
    ),
    ResponseContract(
        schema_name="RuntimeInventoryResponse",
        constant_name="runtimeInventoryPath",
        method="get",
        path="/api/v1/runtimes",
        decoder_source="client",
        required_fields=frozenset(
            {
                "schema_version",
                "interpreters",
                "available",
                "compatible",
                "configuration",
                "selected",
            }
        ),
        field_markers={
            "schema_version": ('["schema_version"]',),
            "interpreters": ('["interpreters"]',),
            "available": ('named: "available"',),
            "compatible": ('named: "compatible"',),
            "configuration": ('["configuration"]',),
            "selected": ('["selected"]',),
        },
        exact_values={"schema_version": "aptus.runtime-inventory.v1"},
        enum_values={},
    ),
    ResponseContract(
        schema_name="PlatformResponse",
        constant_name="platformPath",
        method="get",
        path="/api/v1/platform",
        decoder_source="client",
        required_fields=frozenset({"status", "platform"}),
        field_markers={
            "status": ('["status"]',),
            "platform": ('["platform"]',),
        },
        exact_values={},
        enum_values={"status": frozenset({"ok", "unsupported"})},
    ),
)

SWIFT_INTEGRATION_MARKERS = {
    "controller": (
        "DesktopBackendEndpointPolicy.healthPath",
        "JSONDecoder().decode(BackendHealthResponse.self",
        "health.validate(expectedVersion: expectedVersion)",
    ),
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


def _response_schema(
    openapi: dict[str, Any],
    paths: dict[str, Any],
    contract: ResponseContract,
) -> dict[str, Any]:
    operation = paths.get(contract.path)
    if not isinstance(operation, dict) or contract.method not in operation:
        raise ValueError(
            f"OpenAPI does not define {contract.method.upper()} {contract.path}."
        )
    response = operation[contract.method]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    expected_reference = f"#/components/schemas/{contract.schema_name}"
    if not isinstance(response, dict) or response.get("$ref") != expected_reference:
        raise ValueError(
            f"{contract.method.upper()} {contract.path} must return "
            f"{contract.schema_name}."
        )
    return _resolve_schema(openapi, response)


def verify_client_contracts(
    *,
    openapi_path: Path = OPENAPI_PATH,
    swift_client_path: Path = SWIFT_CLIENT_PATH,
    swift_controller_path: Path = SWIFT_CONTROLLER_PATH,
    swift_models_path: Path = SWIFT_MODELS_PATH,
) -> list[str]:
    errors: list[str] = []
    try:
        openapi = _json_object(openapi_path)
        swift_sources = {
            "client": swift_client_path.read_text(encoding="utf-8"),
            "controller": swift_controller_path.read_text(encoding="utf-8"),
            "models": swift_models_path.read_text(encoding="utf-8"),
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [str(error)]

    constants = dict(
        re.findall(
            r'static\s+let\s+(\w+Path)\s*=\s*"([^"]+)"',
            "\n".join(swift_sources.values()),
        )
    )
    paths = openapi.get("paths")
    if not isinstance(paths, dict):
        return ["OpenAPI document has no paths object."]

    for source_name, markers in SWIFT_INTEGRATION_MARKERS.items():
        source = swift_sources[source_name]
        for marker in markers:
            if marker not in source:
                errors.append(
                    f"Swift {source_name} is missing response integration marker "
                    f"{marker!r}."
                )

    for contract in RESPONSE_CONTRACTS:
        decoder_source = swift_sources[contract.decoder_source]
        actual_path = constants.get(contract.constant_name)
        if actual_path != contract.path:
            errors.append(
                f"Swift {contract.constant_name} is {actual_path!r}; "
                f"expected {contract.path!r}."
            )
        try:
            schema = _response_schema(openapi, paths, contract)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
            continue

        required = schema.get("required")
        required_fields = (
            frozenset(required)
            if isinstance(required, list)
            and all(isinstance(field, str) for field in required)
            else frozenset()
        )
        missing_required = contract.required_fields - required_fields
        if missing_required:
            errors.append(
                f"{contract.schema_name} must require: "
                + ", ".join(sorted(missing_required))
            )
        unconsumed_required = required_fields - contract.field_markers.keys()
        if unconsumed_required:
            errors.append(
                f"{contract.schema_name} has required fields with no Swift decoder "
                "coverage: " + ", ".join(sorted(unconsumed_required))
            )

        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        for field, markers in contract.field_markers.items():
            if field not in required_fields:
                errors.append(
                    f"Swift consumes {contract.schema_name} field {field!r}, "
                    "but OpenAPI does not require it."
                )
            if not any(marker in decoder_source for marker in markers):
                errors.append(
                    f"Swift does not consume required {contract.schema_name} "
                    f"field {field!r}."
                )

        for field, expected in contract.exact_values.items():
            property_schema = properties.get(field)
            actual = (
                property_schema.get("const")
                if isinstance(property_schema, dict)
                else None
            )
            if actual != expected:
                errors.append(
                    f"{contract.schema_name} {field} must be the exact "
                    f"{expected!r} constant."
                )
            if f'"{expected}"' not in decoder_source:
                errors.append(
                    f"Swift does not fail closed on {contract.schema_name} "
                    f"{field} value {expected!r}."
                )

        for field, expected in contract.enum_values.items():
            property_schema = properties.get(field)
            raw_values = (
                property_schema.get("enum")
                if isinstance(property_schema, dict)
                else None
            )
            actual = (
                frozenset(raw_values)
                if isinstance(raw_values, list)
                and all(isinstance(value, str) for value in raw_values)
                else frozenset()
            )
            if actual != expected:
                errors.append(
                    f"{contract.schema_name} {field} values must be exactly: "
                    + ", ".join(sorted(expected))
                )
            for value in sorted(expected):
                if f'"{value}"' not in decoder_source:
                    errors.append(
                        f"Swift does not handle {contract.schema_name} {field} "
                        f"value {value!r}."
                    )
    return errors


def main() -> int:
    errors = verify_client_contracts()
    if not errors:
        print("Swift desktop response contracts match OpenAPI.")
        return 0
    for error in errors:
        print(f"client-contract error: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
