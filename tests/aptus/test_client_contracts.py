from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.check_client_contracts import (
    OPENAPI_PATH,
    SWIFT_CLIENT_PATH,
    SWIFT_CONTROLLER_PATH,
    SWIFT_MODELS_PATH,
    verify_client_contracts,
)


class ClientContractTests(unittest.TestCase):
    def test_swift_desktop_contract_matches_checked_openapi(self) -> None:
        self.assertEqual(verify_client_contracts(), [])

    def test_new_required_response_field_needs_swift_decoder_coverage(self) -> None:
        openapi = json.loads(OPENAPI_PATH.read_text(encoding="utf-8"))
        schema = openapi["components"]["schemas"]["HealthResponse"]
        schema["properties"]["nonce"] = {"type": "string"}
        schema["required"].append("nonce")

        with tempfile.TemporaryDirectory() as directory:
            openapi_path = Path(directory) / "openapi.json"
            openapi_path.write_text(json.dumps(openapi), encoding="utf-8")
            errors = verify_client_contracts(openapi_path=openapi_path)

        self.assertIn(
            "HealthResponse has required fields with no Swift decoder coverage: nonce",
            errors,
        )

    def test_closed_openapi_value_drift_is_rejected(self) -> None:
        openapi = json.loads(OPENAPI_PATH.read_text(encoding="utf-8"))
        openapi["components"]["schemas"]["HealthResponse"]["properties"]["status"][
            "const"
        ] = "healthy"

        with tempfile.TemporaryDirectory() as directory:
            openapi_path = Path(directory) / "openapi.json"
            openapi_path.write_text(json.dumps(openapi), encoding="utf-8")
            errors = verify_client_contracts(openapi_path=openapi_path)

        self.assertIn(
            "HealthResponse status must be the exact 'ok' constant.",
            errors,
        )

    def test_missing_swift_required_field_decoder_is_rejected(self) -> None:
        models = SWIFT_MODELS_PATH.read_text(encoding="utf-8").replace(
            'case apiContractVersion = "api_contract_version"',
            'case apiContractVersion = "contract_version"',
            1,
        )

        with tempfile.TemporaryDirectory() as directory:
            models_path = Path(directory) / "BackendModels.swift"
            models_path.write_text(models, encoding="utf-8")
            errors = verify_client_contracts(swift_models_path=models_path)

        self.assertIn(
            "Swift does not consume required HealthResponse field "
            "'api_contract_version'.",
            errors,
        )

    def test_swift_endpoint_drift_is_rejected(self) -> None:
        client = SWIFT_CLIENT_PATH.read_text(encoding="utf-8").replace(
            'healthPath = "/api/v1/health"',
            'healthPath = "/health"',
            1,
        )

        with tempfile.TemporaryDirectory() as directory:
            client_path = Path(directory) / "DesktopBackendClient.swift"
            client_path.write_text(client, encoding="utf-8")
            errors = verify_client_contracts(swift_client_path=client_path)

        self.assertIn(
            "Swift healthPath is '/health'; expected '/api/v1/health'.",
            errors,
        )

    def test_health_decoder_must_remain_integrated_with_readiness(self) -> None:
        controller = SWIFT_CONTROLLER_PATH.read_text(encoding="utf-8").replace(
            "JSONDecoder().decode(BackendHealthResponse.self",
            "JSONDecoder().decode(BackendReadiness.self",
            1,
        )

        with tempfile.TemporaryDirectory() as directory:
            controller_path = Path(directory) / "BackendController.swift"
            controller_path.write_text(controller, encoding="utf-8")
            errors = verify_client_contracts(swift_controller_path=controller_path)

        self.assertIn(
            "Swift controller is missing response integration marker "
            "'JSONDecoder().decode(BackendHealthResponse.self'.",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
