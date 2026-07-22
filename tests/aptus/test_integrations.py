import json
import unittest
from urllib.error import URLError

from aptus.integrations import (
    DEFAULT_LOCAL_ENDPOINTS,
    LMStudioClient,
    LocalInferenceError,
    OMLXClient,
    discover_local_inference_services,
    validate_local_endpoint,
)


class FakeResponse:
    def __init__(self, value, *, url=None, headers=None, status=200):
        self.payload = json.dumps(value).encode("utf-8")
        self.url = url
        self.headers = headers or {}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self):
        return self.url

    def read(self, limit):
        return self.payload[:limit]


class SequenceTransport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append((request, timeout))
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        if value.url is None:
            value.url = request.full_url
        return value


class LocalInferenceIntegrationTests(unittest.TestCase):
    def test_defaults_are_two_known_loopback_origins(self) -> None:
        self.assertEqual(
            validate_local_endpoint("lm-studio"),
            DEFAULT_LOCAL_ENDPOINTS["lm-studio"],
        )
        self.assertEqual(
            validate_local_endpoint("omlx"), DEFAULT_LOCAL_ENDPOINTS["omlx"]
        )
        self.assertEqual(
            validate_local_endpoint("omlx", "http://[::1]:9000/"),
            "http://[::1]:9000",
        )

    def test_endpoint_validation_rejects_remote_and_ambiguous_urls(self) -> None:
        invalid = (
            "https://127.0.0.1:8000",
            "http://192.168.1.10:8000",
            "http://example.com:8000",
            "http://127.0.0.1",
            "http://user:secret@127.0.0.1:8000",
            "http://127.0.0.1:8000/v1",
            "http://127.0.0.1:8000?next=http://example.com",
        )
        for endpoint in invalid:
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(LocalInferenceError):
                    validate_local_endpoint("omlx", endpoint)

    def test_lm_studio_health_uses_model_list_as_bounded_reachability(self) -> None:
        transport = SequenceTransport([FakeResponse({"data": [{"id": "local"}]})])
        result = LMStudioClient(transport=transport, timeout=1.5).health()
        request, timeout = transport.requests[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:1234/v1/models")
        self.assertEqual(request.method, "GET")
        self.assertEqual(timeout, 1.5)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["service"], "lm-studio")

    def test_omlx_health_uses_documented_health_route(self) -> None:
        transport = SequenceTransport([FakeResponse({"status": "healthy"})])
        result = OMLXClient(transport=transport).health()
        request, _timeout = transport.requests[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/health")
        self.assertEqual(result["payload"]["status"], "healthy")

    def test_model_listing_requires_openai_compatible_ids(self) -> None:
        transport = SequenceTransport(
            [FakeResponse({"object": "list", "data": [{"id": "mlx/model"}]})]
        )
        result = OMLXClient(transport=transport).list_models()
        self.assertEqual(result["models"], [{"id": "mlx/model"}])
        with self.assertRaisesRegex(LocalInferenceError, "string id"):
            OMLXClient(
                transport=SequenceTransport([FakeResponse({"data": [{}]})])
            ).list_models()

    def test_generation_is_non_streaming_and_returns_normalized_text(self) -> None:
        transport = SequenceTransport(
            [
                FakeResponse(
                    {
                        "id": "response-1",
                        "model": "local-model",
                        "choices": [
                            {"message": {"role": "assistant", "content": "answer"}}
                        ],
                        "usage": {"completion_tokens": 1},
                    }
                )
            ]
        )
        client = LMStudioClient(transport=transport, api_key="local-secret")
        result = client.generate(
            model="local-model",
            messages=[{"role": "user", "content": "question"}],
            max_tokens=64,
            temperature=0.2,
        )
        request, _timeout = transport.requests[0]
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://127.0.0.1:1234/v1/chat/completions")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer local-secret")
        self.assertFalse(sent["stream"])
        self.assertEqual(sent["max_tokens"], 64)
        self.assertEqual(result["content"], "answer")
        self.assertEqual(result["response_id"], "response-1")

    def test_invalid_generation_is_rejected_before_transport(self) -> None:
        transport = SequenceTransport([])
        with self.assertRaisesRegex(LocalInferenceError, "At least one message"):
            OMLXClient(transport=transport).generate(model="model", messages=[])
        self.assertEqual(transport.requests, [])

    def test_network_failures_are_structured_and_do_not_include_api_keys(self) -> None:
        client = OMLXClient(
            transport=SequenceTransport([URLError("connection refused")]),
            api_key="do-not-report",
        )
        with self.assertRaises(LocalInferenceError) as raised:
            client.list_models()
        value = raised.exception.to_dict()
        self.assertEqual(value["error"]["code"], "unavailable")
        self.assertEqual(value["error"]["service"], "omlx")
        self.assertNotIn("do-not-report", str(value))

    def test_transport_redirect_to_remote_origin_is_blocked(self) -> None:
        response = FakeResponse({"data": []}, url="http://example.com/models")
        with self.assertRaisesRegex(LocalInferenceError, "leave its configured"):
            LMStudioClient(transport=SequenceTransport([response])).list_models()

    def test_declared_oversized_response_is_rejected_without_reading_it(self) -> None:
        response = FakeResponse(
            {"data": []}, headers={"Content-Length": str(9 * 1024 * 1024)}
        )
        with self.assertRaisesRegex(LocalInferenceError, "exceeds"):
            OMLXClient(transport=SequenceTransport([response])).list_models()

    def test_discovery_checks_only_both_configured_origins(self) -> None:
        transport = SequenceTransport(
            [
                FakeResponse({"data": []}),
                FakeResponse({"status": "healthy"}),
            ]
        )
        result = discover_local_inference_services(transport=transport)
        self.assertEqual(set(result), {"lm-studio", "omlx"})
        self.assertEqual(
            [request.full_url for request, _timeout in transport.requests],
            [
                "http://127.0.0.1:1234/v1/models",
                "http://127.0.0.1:8000/health",
            ],
        )


if __name__ == "__main__":
    unittest.main()
