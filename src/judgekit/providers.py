"""Chat-completion providers behind one protocol, with retries, timeouts, and usage accounting."""

import os
import random
import time
from collections.abc import Callable
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from judgekit.errors import ProviderError

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


class ProviderRequest(BaseModel):
    """A model-agnostic chat-completion request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    prompt: str
    temperature: float = Field(default=0.0, ge=0)
    max_tokens: int = Field(ge=1)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stop: tuple[str, ...] = ()


class ProviderResponse(BaseModel):
    """A model-agnostic chat-completion response with required usage accounting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class Provider(Protocol):
    """Structural interface for a chat-completion backend."""

    provider_id: str

    def complete(self, request: ProviderRequest) -> ProviderResponse: ...


def _compute_backoff(attempt_n: int, retry_after: str | None) -> float:
    """Backoff before retry attempt n (n=1 for the first retry), honoring Retry-After."""
    if retry_after is not None:
        try:
            parsed = float(retry_after)
        except ValueError:
            parsed = -1.0
        if parsed >= 0:
            return min(30.0, parsed)
    return min(30.0, 0.5 * 2.0 ** (attempt_n - 1)) + random.uniform(0, 0.25)


class _RetryingClient:
    """Shared attempt/retry/backoff machinery used by every provider."""

    def __init__(
        self,
        *,
        timeout_s: float,
        max_retries: int,
        transport: httpx.BaseTransport | None,
        sleep: Callable[[float], None],
    ) -> None:
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._transport = transport
        self._sleep = sleep

    def post_json(
        self, provider_id: str, url: str, headers: dict[str, str], body: dict[str, object]
    ) -> dict[str, object]:
        """POST body as JSON, retrying transient failures, and return the parsed JSON body."""
        attempts = self._max_retries + 1
        last_detail = "no attempt was made"
        for attempt in range(1, attempts + 1):
            try:
                with httpx.Client(transport=self._transport, timeout=self._timeout_s) as client:
                    response = client.post(url, headers=headers, json=body)
            except httpx.TransportError as exc:
                last_detail = f"{type(exc).__name__} (transport error)"
                if attempt == attempts:
                    break
                self._sleep(_compute_backoff(attempt, None))
                continue

            if response.status_code == 200:
                try:
                    parsed = response.json()
                except ValueError as exc:
                    excerpt = response.text[:200]
                    raise ProviderError(
                        f"{provider_id}: 200 response body was not valid JSON "
                        f"(excerpt: {excerpt!r})"
                    ) from exc
                if not isinstance(parsed, dict):
                    excerpt = response.text[:200]
                    raise ProviderError(
                        f"{provider_id}: response body was not a JSON object (excerpt: {excerpt!r})"
                    )
                return parsed

            excerpt = response.text[:200]
            retryable = response.status_code == 429 or 500 <= response.status_code < 600
            if not retryable:
                raise ProviderError(
                    f"{provider_id}: request failed with status {response.status_code} "
                    f"(excerpt: {excerpt!r})"
                )

            last_detail = f"status {response.status_code} (excerpt: {excerpt!r})"
            if attempt == attempts:
                break
            self._sleep(_compute_backoff(attempt, response.headers.get("retry-after")))

        raise ProviderError(
            f"{provider_id}: request failed after {attempts} attempt(s) ({last_detail})"
        )


def _require_api_key(provider_id: str, api_key_env: str) -> str:
    """Read the API key from the environment, or raise ProviderError naming the var."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ProviderError(
            f"{provider_id}: missing or empty API key in environment variable {api_key_env!r}"
        )
    return api_key


def _require_int(payload: object, field_name: str, provider_id: str) -> int:
    """Return payload as a plain int, or raise ProviderError (never fabricate a zero)."""
    if isinstance(payload, bool) or not isinstance(payload, int):
        raise ProviderError(f"{provider_id}: {field_name} is missing or not an integer")
    return payload


class AnthropicProvider:
    """Anthropic Messages API chat-completion provider."""

    provider_id = "anthropic"

    def __init__(
        self,
        *,
        api_key_env: str = "ANTHROPIC_API_KEY",
        timeout_s: float = 60.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key_env = api_key_env
        self._client = _RetryingClient(
            timeout_s=timeout_s, max_retries=max_retries, transport=transport, sleep=sleep
        )

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        api_key = _require_api_key(self.provider_id, self._api_key_env)

        body: dict[str, object] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop:
            body["stop_sequences"] = list(request.stop)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        payload = self._client.post_json(self.provider_id, _ANTHROPIC_API_URL, headers, body)
        return self._parse(payload)

    def _parse(self, payload: dict[str, object]) -> ProviderResponse:
        if payload.get("stop_reason") == "max_tokens":
            raise ProviderError(
                f"{self.provider_id}: response was truncated by max_tokens; "
                "raise max_tokens and issue a fresh request"
            )

        content = payload.get("content")
        if not isinstance(content, list):
            raise ProviderError(f"{self.provider_id}: response has no content array")
        text_parts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if not text_parts:
            raise ProviderError(f"{self.provider_id}: response has no text content blocks")
        text = "".join(text_parts)

        usage = payload.get("usage")
        if not isinstance(usage, dict):
            raise ProviderError(f"{self.provider_id}: response is missing a usage block")
        input_tokens = _require_int(
            usage.get("input_tokens"), "usage.input_tokens", self.provider_id
        )
        output_tokens = _require_int(
            usage.get("output_tokens"), "usage.output_tokens", self.provider_id
        )

        return ProviderResponse(text=text, input_tokens=input_tokens, output_tokens=output_tokens)


class OpenAICompatibleProvider:
    """Chat-completion provider for the OpenAI-compatible dialect (vLLM, Ollama, gateways)."""

    provider_id = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 60.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._client = _RetryingClient(
            timeout_s=timeout_s, max_retries=max_retries, transport=transport, sleep=sleep
        )

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        api_key = _require_api_key(self.provider_id, self._api_key_env)

        body: dict[str, object] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop:
            body["stop"] = list(request.stop)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }

        url = f"{self._base_url}/chat/completions"
        payload = self._client.post_json(self.provider_id, url, headers, body)
        return self._parse(payload)

    def _parse(self, payload: dict[str, object]) -> ProviderResponse:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError(f"{self.provider_id}: response has no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ProviderError(f"{self.provider_id}: response choice is malformed")

        if choice.get("finish_reason") == "length":
            raise ProviderError(
                f"{self.provider_id}: response was truncated (finish_reason=length); "
                "raise max_tokens and issue a fresh request"
            )

        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ProviderError(f"{self.provider_id}: response message content is missing or null")

        usage = payload.get("usage")
        if not isinstance(usage, dict):
            raise ProviderError(f"{self.provider_id}: response is missing a usage block")
        input_tokens = _require_int(
            usage.get("prompt_tokens"), "usage.prompt_tokens", self.provider_id
        )
        output_tokens = _require_int(
            usage.get("completion_tokens"), "usage.completion_tokens", self.provider_id
        )

        return ProviderResponse(
            text=content, input_tokens=input_tokens, output_tokens=output_tokens
        )


def get_provider(
    provider: str,
    *,
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout_s: float = 60.0,
    max_retries: int = 3,
) -> Provider:
    """Construct a registered provider by id, applying id-specific defaults."""
    if provider == "anthropic":
        if base_url is not None:
            raise ProviderError("base_url is only supported for the openai-compatible provider")
        return AnthropicProvider(
            api_key_env=api_key_env if api_key_env is not None else "ANTHROPIC_API_KEY",
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
    if provider == "openai-compatible":
        return OpenAICompatibleProvider(
            base_url=base_url if base_url is not None else "https://api.openai.com/v1",
            api_key_env=api_key_env if api_key_env is not None else "OPENAI_API_KEY",
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
    raise ProviderError(
        f"unknown provider id: {provider!r}; supported providers: 'anthropic', 'openai-compatible'"
    )
