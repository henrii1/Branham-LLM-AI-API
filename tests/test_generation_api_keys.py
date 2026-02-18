from __future__ import annotations

import pytest

from branham_model_api.generation.api_keys import OpenRouterKeyManager


def test_key_manager_loads_available_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY_A", "key-a")
    monkeypatch.setenv("OPENROUTER_API_KEY_B", "key-b")
    monkeypatch.delenv("OPENROUTER_API_KEY_C", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_D", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_E", raising=False)

    manager = OpenRouterKeyManager(seed=42)
    assert manager.has_keys is True
    assert manager.key_count == 2
    assert set(manager.available_key_names) == {
        "OPENROUTER_API_KEY_A",
        "OPENROUTER_API_KEY_B",
    }


def test_pick_random_key_honors_exclude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY_A", "key-a")
    monkeypatch.setenv("OPENROUTER_API_KEY_B", "key-b")
    manager = OpenRouterKeyManager(seed=7)

    first = manager.pick_random_key()
    second = manager.pick_random_key(exclude_env_name=first.env_name)
    assert second.env_name != first.env_name


def test_pick_random_key_raises_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY_A", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_B", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_C", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_D", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_E", raising=False)

    manager = OpenRouterKeyManager()
    with pytest.raises(RuntimeError):
        manager.pick_random_key()
