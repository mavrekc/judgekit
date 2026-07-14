import json

import httpx
import pytest

from judgekit.errors import ProviderError
from judgekit.providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    ProviderRequest,
    get_provider,
)


def _request(**overrides: object) -> ProviderRequest:
    fields: dict[str, object] = {"model": "test-model", "prompt": "hi", "max_tokens": 64}
    fields.update(overrides)
    return ProviderRequest(**fields)  # type: ignore[arg-type]


def _anthropic_payload(
    *,
    texts: list[str] | None = None,
    stop_reason: str = "end_turn",
    input_tokens: object = 10,
    output_tokens: object = 5,
    include_usage: bool = True,
) -> dict[str, object]:
    if texts is None:
        texts = ["hello"]
    payload: dict[str, object] = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": [{"type": "text", "text": t, "citations": []} for t in texts],
        "stop_reason": stop_reason,
        "stop_sequence": None,
    }
    if include_usage:
        usage: dict[str, object] = {}
        if input_tokens is not None:
            usage["input_tokens"] = input_tokens
        if output_tokens is not None:
            usage["output_tokens"] = output_tokens
        payload["usage"] = usage
    return payload


def _openai_payload(
    *,
    content: object = "hello",
    finish_reason: str = "stop",
    prompt_tokens: object = 10,
    completion_tokens: object = 5,
    include_usage: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    if include_usage:
        usage: dict[str, object] = {}
        if prompt_tokens is not None:
            usage["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            usage["completion_tokens"] = completion_tokens
        payload["usage"] = usage
    return payload


def test_anthropic_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://api.anthropic.com/v1/messages")
        assert request.method == "POST"
        assert request.headers["x-api-key"] == "dummy-key"
        assert "anthropic-version" in request.headers
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["max_tokens"] == 64
        assert body["temperature"] == 0.0
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert "top_p" not in body
        assert "stop_sequences" not in body
        return httpx.Response(200, json=_anthropic_payload())

    sleeps: list[float] = []
    provider = AnthropicProvider(transport=httpx.MockTransport(handler), sleep=sleeps.append)
    response = provider.complete(_request())
    assert response.text == "hello"
    assert response.input_tokens == 10
    assert response.output_tokens == 5


def test_anthropic_top_p_and_stop_included_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["top_p"] == 0.9
        assert body["stop_sequences"] == ["STOP", "END"]
        return httpx.Response(200, json=_anthropic_payload())

    provider = AnthropicProvider(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    provider.complete(_request(top_p=0.9, stop=("STOP", "END")))


def test_anthropic_multi_block_content_concatenated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_anthropic_payload(texts=["Hello, ", "world!"]))

    provider = AnthropicProvider(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    response = provider.complete(_request())
    assert response.text == "Hello, world!"


def test_openai_compatible_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://localhost:8000/v1/chat/completions")
        assert request.headers["authorization"] == "Bearer dummy-key"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["max_tokens"] == 64
        assert body["temperature"] == 0.0
        return httpx.Response(200, json=_openai_payload())

    provider = OpenAICompatibleProvider(
        base_url="http://localhost:8000/v1/",
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    response = provider.complete(_request())
    assert response.text == "hello"
    assert response.input_tokens == 10
    assert response.output_tokens == 5


def test_429_with_retry_after_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "2"}, text="rate limited")
        return httpx.Response(200, json=_anthropic_payload())

    sleeps: list[float] = []
    provider = AnthropicProvider(transport=httpx.MockTransport(handler), sleep=sleeps.append)
    response = provider.complete(_request())
    assert calls["n"] == 2
    assert sleeps == [2.0]
    assert response.text == "hello"


def test_500_then_success_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="internal error")
        return httpx.Response(200, json=_anthropic_payload())

    provider = AnthropicProvider(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    response = provider.complete(_request())
    assert calls["n"] == 2
    assert response.text == "hello"


def test_timeout_then_success_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json=_anthropic_payload())

    provider = AnthropicProvider(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    response = provider.complete(_request())
    assert calls["n"] == 2
    assert response.text == "hello"


def test_exhaustion_raises_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="still broken")

    provider = AnthropicProvider(
        transport=httpx.MockTransport(handler), sleep=lambda _: None, max_retries=2
    )
    with pytest.raises(ProviderError):
        provider.complete(_request())
    assert calls["n"] == 3


def test_401_fails_fast_without_retry_and_never_leaks_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"type": "authentication_error"}})

    provider = AnthropicProvider(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    with pytest.raises(ProviderError) as exc_info:
        provider.complete(_request())
    assert calls["n"] == 1
    assert "dummy-key" not in str(exc_info.value)


def test_anthropic_missing_usage_block_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_anthropic_payload(include_usage=False))

    provider = AnthropicProvider(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    with pytest.raises(ProviderError):
        provider.complete(_request())


def test_openai_compatible_missing_usage_block_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_openai_payload(include_usage=False))

    provider = OpenAICompatibleProvider(
        transport=httpx.MockTransport(handler), sleep=lambda _: None
    )
    with pytest.raises(ProviderError):
        provider.complete(_request())


def test_anthropic_max_tokens_stop_reason_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_anthropic_payload(stop_reason="max_tokens"))

    provider = AnthropicProvider(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    with pytest.raises(ProviderError, match="max_tokens"):
        provider.complete(_request())


def test_openai_compatible_length_finish_reason_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_openai_payload(finish_reason="length"))

    provider = OpenAICompatibleProvider(
        transport=httpx.MockTransport(handler), sleep=lambda _: None
    )
    with pytest.raises(ProviderError, match="length"):
        provider.complete(_request())


def test_anthropic_no_text_block_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = _anthropic_payload()
        payload["content"] = [{"type": "thinking", "thinking": "hmm"}]
        return httpx.Response(200, json=payload)

    provider = AnthropicProvider(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    with pytest.raises(ProviderError):
        provider.complete(_request())


def test_openai_compatible_null_content_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_openai_payload(content=None))

    provider = OpenAICompatibleProvider(
        transport=httpx.MockTransport(handler), sleep=lambda _: None
    )
    with pytest.raises(ProviderError):
        provider.complete(_request())


def test_missing_env_var_raises_naming_var_and_sends_no_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_anthropic_payload())

    provider = AnthropicProvider(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        provider.complete(_request())
    assert calls["n"] == 0


def test_non_json_200_body_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json at all")

    provider = AnthropicProvider(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    with pytest.raises(ProviderError):
        provider.complete(_request())


def test_get_provider_anthropic_defaults() -> None:
    provider = get_provider("anthropic")
    assert isinstance(provider, AnthropicProvider)
    assert provider.provider_id == "anthropic"
    assert provider._api_key_env == "ANTHROPIC_API_KEY"


def test_get_provider_openai_compatible_defaults() -> None:
    provider = get_provider("openai-compatible")
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.provider_id == "openai-compatible"
    assert provider._api_key_env == "OPENAI_API_KEY"
    assert provider._base_url == "https://api.openai.com/v1"


def test_get_provider_unknown_id_raises() -> None:
    with pytest.raises(ProviderError, match="unknown provider id"):
        get_provider("not-a-provider")


def test_get_provider_anthropic_with_base_url_raises() -> None:
    with pytest.raises(ProviderError, match="base_url"):
        get_provider("anthropic", base_url="https://example.com")
