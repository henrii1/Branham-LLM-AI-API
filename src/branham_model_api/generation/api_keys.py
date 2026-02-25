"""
API key lookup and random selection for LLM providers.

Supports any provider by configuring a key_prefix.
Keys are read from env vars: {key_prefix}_A .. {key_prefix}_E.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

_KEY_SUFFIXES = ("_A", "_B", "_C", "_D", "_E")


@dataclass(frozen=True)
class ApiKeyRecord:
    """Resolved API key from environment."""

    env_name: str
    value: str


class LLMKeyManager:
    """
    Manages LLM provider API keys from environment variables.

    Selection policy is uniform random across available keys.
    Works with any provider by configuring the key_prefix
    (e.g. "OPENROUTER_API_KEY" or "DEEPSEEK_API_KEY").
    """

    def __init__(
        self,
        key_prefix: str = "OPENROUTER_API_KEY",
        seed: int | None = None,
    ) -> None:
        self._key_prefix = key_prefix
        self._key_names = tuple(f"{key_prefix}{s}" for s in _KEY_SUFFIXES)
        self._rng = random.Random(seed)
        self._keys = self._load_keys()

    def _load_keys(self) -> tuple[ApiKeyRecord, ...]:
        keys: list[ApiKeyRecord] = []
        for env_name in self._key_names:
            value = os.getenv(env_name, "").strip()
            if value:
                keys.append(ApiKeyRecord(env_name=env_name, value=value))
        return tuple(keys)

    @property
    def available_key_names(self) -> list[str]:
        """Return names (not values) of loaded keys."""
        return [k.env_name for k in self._keys]

    @property
    def has_keys(self) -> bool:
        return len(self._keys) > 0

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def pick_random_key(self, exclude_env_name: str | None = None) -> ApiKeyRecord:
        if not self._keys:
            raise RuntimeError(
                f"No API keys found for prefix '{self._key_prefix}'. "
                f"Set one or more of: {', '.join(self._key_names)}"
            )

        candidates = [k for k in self._keys if k.env_name != exclude_env_name]
        if not candidates:
            candidates = list(self._keys)
        return self._rng.choice(candidates)


# Backwards-compatible alias
OpenRouterKeyManager = LLMKeyManager


@dataclass(frozen=True)
class MixedApiKeyRecord:
    """Resolved API key with associated provider route."""

    env_name: str
    value: str
    route_id: str  # e.g., "deepseek" or "openrouter"


class MixedLLMKeyManager:
    """
    Loads keys from multiple key prefixes and returns keys uniformly at random.

    Ratio behavior:
    - If you set 4 DeepSeek keys (DEEPSEEK_API_KEY_A..D) and 1 OpenRouter key
      (OPENROUTER_API_KEY_A), uniform selection yields an expected ~4:1 split.
    """

    def __init__(
        self,
        *,
        route_key_prefixes: dict[str, str],
        seed: int | None = None,
    ) -> None:
        self._route_key_prefixes = dict(route_key_prefixes)
        self._rng = random.Random(seed)
        self._keys = self._load_keys()

    def _load_keys(self) -> tuple[MixedApiKeyRecord, ...]:
        keys: list[MixedApiKeyRecord] = []
        for route_id, prefix in self._route_key_prefixes.items():
            for suf in _KEY_SUFFIXES:
                env_name = f"{prefix}{suf}"
                value = os.getenv(env_name, "").strip()
                if value:
                    keys.append(
                        MixedApiKeyRecord(
                            env_name=env_name,
                            value=value,
                            route_id=route_id,
                        )
                    )
        return tuple(keys)

    @property
    def has_keys(self) -> bool:
        return len(self._keys) > 0

    @property
    def key_count(self) -> int:
        return len(self._keys)

    @property
    def available_key_names(self) -> list[str]:
        return [k.env_name for k in self._keys]

    def pick_random_key(self, exclude_env_name: str | None = None) -> MixedApiKeyRecord:
        if not self._keys:
            prefixes = ", ".join(sorted(set(self._route_key_prefixes.values())))
            raise RuntimeError(
                "No API keys found for mixed mode. "
                f"Set one or more keys for prefixes: {prefixes}"
            )
        candidates = [k for k in self._keys if k.env_name != exclude_env_name]
        if not candidates:
            candidates = list(self._keys)
        return self._rng.choice(candidates)
