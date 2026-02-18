from __future__ import annotations

from types import SimpleNamespace

import pytest

from branham_model_api.generation.api_keys import OpenRouterKeyManager
from branham_model_api.generation.litellm_client import (
    LiteLLMClient,
    LiteLLMClientConfig,
    LiteLLMRateLimitError,
    LiteLLMServiceUnavailableError,
)


class FakeRateLimitError(Exception):
    status_code = 429


def _set_two_keys(monkeypatch: pytest.MonkeyPatch) -> OpenRouterKeyManager:
    monkeypatch.setenv("OPENROUTER_API_KEY_A", "key-a")
    monkeypatch.setenv("OPENROUTER_API_KEY_B", "key-b")
    monkeypatch.delenv("OPENROUTER_API_KEY_C", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_D", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_E", raising=False)
    return OpenRouterKeyManager(seed=1)


def test_client_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _set_two_keys(monkeypatch)
    client = LiteLLMClient(
        config=LiteLLMClientConfig(model="openrouter/deepseek/deepseek-chat"),
        key_manager=manager,
    )

    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    monkeypatch.setattr("branham_model_api.generation.litellm_client.litellm.completion", fake_completion)

    response = client.completion(messages=[{"role": "user", "content": "hi"}])
    assert response.choices[0].message.content == "ok"
    assert len(calls) == 1
    assert calls[0]["model"] == "openrouter/deepseek/deepseek-chat"


def test_client_retries_once_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _set_two_keys(monkeypatch)
    client = LiteLLMClient(
        config=LiteLLMClientConfig(model="openrouter/deepseek/deepseek-chat"),
        key_manager=manager,
    )

    state = {"count": 0}

    def fake_completion(**kwargs):
        state["count"] += 1
        if state["count"] == 1:
            raise FakeRateLimitError("rate limit")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="retry-ok"))]
        )

    monkeypatch.setattr("branham_model_api.generation.litellm_client.litellm.completion", fake_completion)

    response = client.completion(messages=[{"role": "user", "content": "hi"}])
    assert response.choices[0].message.content == "retry-ok"
    assert state["count"] == 2


def test_client_raises_rate_limit_after_second_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _set_two_keys(monkeypatch)
    client = LiteLLMClient(
        config=LiteLLMClientConfig(model="openrouter/deepseek/deepseek-chat"),
        key_manager=manager,
    )

    def fake_completion(**kwargs):
        raise FakeRateLimitError("too many requests")

    monkeypatch.setattr("branham_model_api.generation.litellm_client.litellm.completion", fake_completion)

    with pytest.raises(LiteLLMRateLimitError):
        client.completion(messages=[{"role": "user", "content": "hi"}])


def test_client_raises_service_unavailable_on_non_rate_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _set_two_keys(monkeypatch)
    client = LiteLLMClient(
        config=LiteLLMClientConfig(model="openrouter/deepseek/deepseek-chat"),
        key_manager=manager,
    )

    def fake_completion(**kwargs):
        raise ValueError("upstream broke")

    monkeypatch.setattr("branham_model_api.generation.litellm_client.litellm.completion", fake_completion)

    with pytest.raises(LiteLLMServiceUnavailableError):
        client.completion(messages=[{"role": "user", "content": "hi"}])


def test_stream_completion_success(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _set_two_keys(monkeypatch)
    client = LiteLLMClient(
        config=LiteLLMClientConfig(model="openrouter/deepseek/deepseek-chat"),
        key_manager=manager,
    )

    fake_iter = iter(["chunk-1"])

    def fake_completion(**kwargs):
        assert kwargs["stream"] is True
        return fake_iter

    monkeypatch.setattr(
        "branham_model_api.generation.litellm_client.litellm.completion",
        fake_completion,
    )
    out = client.stream_completion(messages=[{"role": "user", "content": "hi"}])
    assert out is fake_iter
