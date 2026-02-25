from __future__ import annotations

from types import SimpleNamespace

import pytest

from branham_model_api.generation.api_keys import MixedLLMKeyManager
from branham_model_api.generation.litellm_client import (
    LiteLLMMixedClient,
    LiteLLMRouteConfig,
    LiteLLMRateLimitError,
)


class FakeRateLimitError(Exception):
    status_code = 429


def _set_mixed_keys(monkeypatch: pytest.MonkeyPatch) -> MixedLLMKeyManager:
    # 4 deepseek keys + 1 openrouter key
    monkeypatch.setenv("DEEPSEEK_API_KEY_A", "ds-a")
    monkeypatch.setenv("DEEPSEEK_API_KEY_B", "ds-b")
    monkeypatch.setenv("DEEPSEEK_API_KEY_C", "ds-c")
    monkeypatch.setenv("DEEPSEEK_API_KEY_D", "ds-d")
    monkeypatch.delenv("DEEPSEEK_API_KEY_E", raising=False)

    monkeypatch.setenv("OPENROUTER_API_KEY_A", "or-a")
    monkeypatch.delenv("OPENROUTER_API_KEY_B", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_C", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_D", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_E", raising=False)

    return MixedLLMKeyManager(
        route_key_prefixes={
            "deepseek": "DEEPSEEK_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        },
        seed=1,
    )


def test_mixed_client_uses_route_model_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _set_mixed_keys(monkeypatch)
    client = LiteLLMMixedClient(
        routes={
            "deepseek": LiteLLMRouteConfig(model="deepseek/deepseek-chat", base_url="https://api.deepseek.com"),
            "openrouter": LiteLLMRouteConfig(model="openrouter/deepseek/deepseek-chat", base_url="https://openrouter.ai/api/v1"),
        },
        key_manager=mgr,
    )

    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    monkeypatch.setattr(
        "branham_model_api.generation.litellm_client.litellm.completion",
        fake_completion,
    )

    _ = client.completion(messages=[{"role": "user", "content": "hi"}])
    assert len(calls) == 1
    assert calls[0]["model"] in {
        "deepseek/deepseek-chat",
        "openrouter/deepseek/deepseek-chat",
    }
    # base_url must match the selected route model.
    if calls[0]["model"].startswith("openrouter/"):
        assert calls[0]["base_url"] == "https://openrouter.ai/api/v1"
    else:
        assert calls[0]["base_url"] == "https://api.deepseek.com"


def test_mixed_client_retries_on_rate_limit_with_different_key(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _set_mixed_keys(monkeypatch)
    client = LiteLLMMixedClient(
        routes={
            "deepseek": LiteLLMRouteConfig(model="deepseek/deepseek-chat", base_url="https://api.deepseek.com"),
            "openrouter": LiteLLMRouteConfig(model="openrouter/deepseek/deepseek-chat", base_url="https://openrouter.ai/api/v1"),
        },
        key_manager=mgr,
    )

    state = {"count": 0, "api_keys": []}

    def fake_completion(**kwargs):
        state["count"] += 1
        state["api_keys"].append(kwargs.get("api_key"))
        if state["count"] == 1:
            raise FakeRateLimitError("rate limit")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="retry-ok"))]
        )

    monkeypatch.setattr(
        "branham_model_api.generation.litellm_client.litellm.completion",
        fake_completion,
    )

    response = client.completion(messages=[{"role": "user", "content": "hi"}])
    assert response.choices[0].message.content == "retry-ok"
    assert state["count"] == 2
    assert len(state["api_keys"]) == 2
    assert state["api_keys"][0] != state["api_keys"][1]


def test_mixed_client_raises_rate_limit_after_second_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _set_mixed_keys(monkeypatch)
    client = LiteLLMMixedClient(
        routes={
            "deepseek": LiteLLMRouteConfig(model="deepseek/deepseek-chat", base_url="https://api.deepseek.com"),
            "openrouter": LiteLLMRouteConfig(model="openrouter/deepseek/deepseek-chat", base_url="https://openrouter.ai/api/v1"),
        },
        key_manager=mgr,
    )

    def fake_completion(**kwargs):
        raise FakeRateLimitError("too many requests")

    monkeypatch.setattr(
        "branham_model_api.generation.litellm_client.litellm.completion",
        fake_completion,
    )

    with pytest.raises(LiteLLMRateLimitError):
        client.completion(messages=[{"role": "user", "content": "hi"}])

