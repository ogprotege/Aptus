from __future__ import annotations

import ipaddress
import json
import re
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)


Transport = Callable[..., Any]
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_REQUEST_BYTES = 2 * 1024 * 1024
DEFAULT_LOCAL_ENDPOINTS = {
    "lm-studio": "http://127.0.0.1:1234",
    "omlx": "http://127.0.0.1:8000",
}
_SERVICE_LABELS = {"lm-studio": "LM Studio", "omlx": "oMLX"}
_MODEL_PATH = "/v1/models"
_GENERATION_PATH = "/v1/chat/completions"


class LocalInferenceError(RuntimeError):
    """A safe, structured failure from an inference-only local service call."""

    def __init__(
        self,
        *,
        code: str,
        service: str,
        operation: str,
        message: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.service = service
        self.operation = operation
        self.message = message
        self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "error",
            "error": {
                "code": self.code,
                "service": self.service,
                "operation": self.operation,
                "message": self.message,
                "status_code": self.status_code,
            },
        }


def _integration_error(
    service: str,
    operation: str,
    code: str,
    message: str,
    *,
    status_code: int | None = None,
) -> LocalInferenceError:
    return LocalInferenceError(
        code=code,
        service=service,
        operation=operation,
        message=message,
        status_code=status_code,
    )


def validate_local_endpoint(service: str, endpoint: str | None = None) -> str:
    """Accept one explicit loopback origin. Never accept LAN or remote hosts."""

    if service not in DEFAULT_LOCAL_ENDPOINTS:
        raise _integration_error(
            service,
            "configuration",
            "unsupported_service",
            "Supported local inference services are lm-studio and omlx.",
        )
    candidate = endpoint or DEFAULT_LOCAL_ENDPOINTS[service]
    if not isinstance(candidate, str) or not candidate.strip():
        raise _integration_error(
            service,
            "configuration",
            "invalid_endpoint",
            "A local endpoint is required.",
        )
    candidate = candidate.strip()
    if re.search(r"[\x00-\x20]", candidate):
        raise _integration_error(
            service,
            "configuration",
            "invalid_endpoint",
            "The local endpoint contains whitespace or control characters.",
        )
    parsed = urlsplit(candidate)
    if parsed.scheme != "http":
        raise _integration_error(
            service,
            "configuration",
            "invalid_endpoint",
            "Local inference endpoints must use http.",
        )
    if parsed.username is not None or parsed.password is not None:
        raise _integration_error(
            service,
            "configuration",
            "invalid_endpoint",
            "Credentials are not allowed in a local endpoint URL.",
        )
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise _integration_error(
            service,
            "configuration",
            "invalid_endpoint",
            "Supply only the local service origin, without a path, query, or fragment.",
        )
    hostname = (parsed.hostname or "").lower()
    if hostname != "localhost":
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError as error:
            raise _integration_error(
                service,
                "configuration",
                "non_loopback_endpoint",
                "The inference endpoint must use a loopback address.",
            ) from error
        if not address.is_loopback or "%" in hostname:
            raise _integration_error(
                service,
                "configuration",
                "non_loopback_endpoint",
                "The inference endpoint must use a loopback address.",
            )
    try:
        port = parsed.port
    except ValueError as error:
        raise _integration_error(
            service,
            "configuration",
            "invalid_endpoint",
            "The local inference endpoint has an invalid port.",
        ) from error
    if port is None or not 1 <= port <= 65535:
        raise _integration_error(
            service,
            "configuration",
            "invalid_endpoint",
            "The local inference endpoint requires an explicit valid port.",
        )
    normalized_host = f"[{hostname}]" if ":" in hostname else hostname
    return f"http://{normalized_host}:{port}"


def _loopback_origin(url: str) -> tuple[str, str, int] | None:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.username or parsed.password:
        return None
    hostname = (parsed.hostname or "").lower()
    if hostname != "localhost":
        try:
            if not ipaddress.ip_address(hostname).is_loopback:
                return None
        except ValueError:
            return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port is None:
        return None
    return parsed.scheme, hostname, port


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


_LOCAL_OPENER = build_opener(ProxyHandler({}), _RejectRedirects())


def _default_transport(request: Request, *, timeout: float) -> Any:
    return _LOCAL_OPENER.open(request, timeout=timeout)


def _http_error_message(error: HTTPError) -> str:
    message = f"Local service returned HTTP {error.code}."
    try:
        body = error.read(MAX_RESPONSE_BYTES + 1)
        if len(body) <= MAX_RESPONSE_BYTES:
            value = json.loads(body.decode("utf-8"))
            if isinstance(value, dict):
                candidate: Any = value.get("message")
                if isinstance(value.get("error"), dict):
                    candidate = value["error"].get("message", candidate)
                if isinstance(candidate, str) and candidate.strip():
                    message = f"{message} {candidate.strip()[:500]}"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        pass
    return message


class LocalInferenceClient:
    """Bounded inference and evaluation calls to one explicit local service."""

    def __init__(
        self,
        service: str,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout: float = 5.0,
        transport: Transport | None = None,
    ) -> None:
        if not 0 < timeout <= 30:
            raise _integration_error(
                service,
                "configuration",
                "invalid_timeout",
                "timeout must be in (0, 30].",
            )
        self.service = service
        self.endpoint = validate_local_endpoint(service, endpoint)
        self.timeout = timeout
        self._api_key = api_key.strip() if isinstance(api_key, str) else None
        self._transport = transport or _default_transport

    def _request_json(
        self,
        operation: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.endpoint}{path}"
        headers = {
            "Accept": "application/json",
            "User-Agent": "aptus/0.2.0",
        }
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            if len(body) > MAX_REQUEST_BYTES:
                raise _integration_error(
                    self.service,
                    operation,
                    "request_too_large",
                    "The local inference request exceeds the Aptus bound.",
                )
            headers["Content-Type"] = "application/json"
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(
            url,
            data=body,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )
        try:
            response = self._transport(request, timeout=self.timeout)
            with response:
                final_url = response.geturl() if hasattr(response, "geturl") else url
                if _loopback_origin(final_url) != _loopback_origin(self.endpoint):
                    raise _integration_error(
                        self.service,
                        operation,
                        "redirect_blocked",
                        "The local service attempted to leave its configured loopback origin.",
                    )
                status_code = int(getattr(response, "status", 200))
                if status_code >= 400:
                    raise _integration_error(
                        self.service,
                        operation,
                        "http_error",
                        f"Local service returned HTTP {status_code}.",
                        status_code=status_code,
                    )
                content_length = response.headers.get("Content-Length")
                if (
                    content_length is not None
                    and int(content_length) > MAX_RESPONSE_BYTES
                ):
                    raise _integration_error(
                        self.service,
                        operation,
                        "response_too_large",
                        "The local service response exceeds the Aptus bound.",
                    )
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except LocalInferenceError:
            raise
        except HTTPError as error:
            raise _integration_error(
                self.service,
                operation,
                "http_error",
                _http_error_message(error),
                status_code=error.code,
            ) from error
        except (TimeoutError, URLError, OSError) as error:
            reason = getattr(error, "reason", error)
            code = "timeout" if isinstance(reason, TimeoutError) else "unavailable"
            raise _integration_error(
                self.service,
                operation,
                code,
                f"The local service is unavailable: {reason}",
            ) from error
        except (TypeError, ValueError) as error:
            raise _integration_error(
                self.service,
                operation,
                "invalid_response",
                f"The local service returned invalid response metadata: {error}",
            ) from error
        if len(raw) > MAX_RESPONSE_BYTES:
            raise _integration_error(
                self.service,
                operation,
                "response_too_large",
                "The local service response exceeds the Aptus bound.",
            )
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _integration_error(
                self.service,
                operation,
                "invalid_response",
                "The local service returned invalid JSON.",
            ) from error
        if not isinstance(value, dict):
            raise _integration_error(
                self.service,
                operation,
                "invalid_response",
                "The local service returned a non-object JSON document.",
            )
        return value

    def health(self) -> dict[str, Any]:
        path = "/health" if self.service == "omlx" else _MODEL_PATH
        payload = self._request_json("health", path)
        if self.service == "lm-studio" and not isinstance(payload.get("data"), list):
            raise _integration_error(
                self.service,
                "health",
                "invalid_response",
                "LM Studio health validation requires an OpenAI-compatible model list.",
            )
        return {
            "status": "ok",
            "service": self.service,
            "service_name": _SERVICE_LABELS[self.service],
            "endpoint": self.endpoint,
            "payload": payload,
        }

    def list_models(self) -> dict[str, Any]:
        payload = self._request_json("list-models", _MODEL_PATH)
        raw_models = payload.get("data")
        if not isinstance(raw_models, list):
            raise _integration_error(
                self.service,
                "list-models",
                "invalid_response",
                "The local service model response has no data list.",
            )
        models: list[dict[str, Any]] = []
        for item in raw_models:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise _integration_error(
                    self.service,
                    "list-models",
                    "invalid_response",
                    "Every local model entry requires a string id.",
                )
            models.append(dict(item))
        return {
            "status": "ok",
            "service": self.service,
            "endpoint": self.endpoint,
            "models": models,
        }

    def generate(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        if (
            not isinstance(model, str)
            or not model.strip()
            or len(model) > 256
            or re.search(r"[\x00-\x1f]", model)
        ):
            raise _integration_error(
                self.service,
                "generate",
                "invalid_request",
                "model must be a non-empty local model identifier.",
            )
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 32768:
            raise _integration_error(
                self.service,
                "generate",
                "invalid_request",
                "max_tokens must be an integer in [1, 32768].",
            )
        if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise _integration_error(
                self.service,
                "generate",
                "invalid_request",
                "temperature must be in [0, 2].",
            )
        normalized_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise _integration_error(
                    self.service,
                    "generate",
                    "invalid_request",
                    "Every message must be an object.",
                )
            role, content = message.get("role"), message.get("content")
            if role not in {"system", "user", "assistant", "tool"}:
                raise _integration_error(
                    self.service,
                    "generate",
                    "invalid_request",
                    "Every message requires a supported role.",
                )
            if not isinstance(content, str) or not content:
                raise _integration_error(
                    self.service,
                    "generate",
                    "invalid_request",
                    "Every message requires non-empty string content.",
                )
            normalized_messages.append({"role": role, "content": content})
        if not normalized_messages:
            raise _integration_error(
                self.service,
                "generate",
                "invalid_request",
                "At least one message is required.",
            )
        payload = self._request_json(
            "generate",
            _GENERATION_PATH,
            payload={
                "model": model.strip(),
                "messages": normalized_messages,
                "max_tokens": max_tokens,
                "temperature": float(temperature),
                "stream": False,
            },
        )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise _integration_error(
                self.service,
                "generate",
                "invalid_response",
                "The local generation response has no choices.",
            )
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise _integration_error(
                self.service,
                "generate",
                "invalid_response",
                "The local generation response has no text content.",
            )
        return {
            "status": "ok",
            "service": self.service,
            "endpoint": self.endpoint,
            "model": payload.get("model", model.strip()),
            "content": content,
            "usage": payload.get("usage"),
            "response_id": payload.get("id"),
            "payload": payload,
        }


class LMStudioClient(LocalInferenceClient):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("lm-studio", **kwargs)


class OMLXClient(LocalInferenceClient):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("omlx", **kwargs)


def discover_local_inference_services(
    *,
    lm_studio_endpoint: str | None = None,
    omlx_endpoint: str | None = None,
    timeout: float = 2.0,
    transport: Transport | None = None,
) -> dict[str, dict[str, Any]]:
    """Probe two known loopback origins only. This never scans local ports."""

    clients = (
        LMStudioClient(
            endpoint=lm_studio_endpoint, timeout=timeout, transport=transport
        ),
        OMLXClient(endpoint=omlx_endpoint, timeout=timeout, transport=transport),
    )
    results: dict[str, dict[str, Any]] = {}
    for client in clients:
        try:
            results[client.service] = client.health()
        except LocalInferenceError as error:
            results[client.service] = error.to_dict()
            results[client.service]["endpoint"] = client.endpoint
    return results
