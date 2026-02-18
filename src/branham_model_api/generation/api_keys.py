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
