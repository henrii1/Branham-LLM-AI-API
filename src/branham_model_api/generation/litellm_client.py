"""
LiteLLM client wrapper for multi-provider LLM generation.

Supports OpenRouter, DeepSeek direct, and any LiteLLM-compatible provider.
Provider is selected via config/default.yaml → llm.provider.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import litellm

from branham_model_api.generation.api_keys import LLMKeyManager

logger = logging.getLogger(__name__)


class LiteLLMRateLimitError(Exception):
    """Raised when the upstream provider is rate-limited."""


class LiteLLMServiceUnavailableError(Exception):
    """Raised for non-rate-limit upstream failures."""


@dataclass
class LiteLLMClientConfig:
    """Runtime configuration for LiteLLM requests."""

    model: str
    base_url: str | None = None
    timeout: float = 30.0
    temperature: float = 0.2


def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Detect rate-limit class errors in a provider-agnostic way.

    We retry only for rate-limit conditions per project policy.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    message = str(exc).lower()
    return (
        "rate limit" in message
        or "rate_limit" in message
        or "too many requests" in message
        or "429" in message
    )


class LiteLLMClient:
    """Small wrapper that handles key selection and one rate-limit retry."""

    def __init__(
        self,
        config: LiteLLMClientConfig,
        key_manager: LLMKeyManager,
    ) -> None:
        self.config = config
        self.key_manager = key_manager
        if not self.key_manager.has_keys:
            raise RuntimeError(
                f"No API keys found for key manager "
                f"(available prefixes checked: {self.key_manager.available_key_names or 'none'}). "
                f"Check your .env file."
            )

    def _completion_with_key(
        self,
        *,
        api_key: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        max_tokens: int | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "api_key": api_key,
            "timeout": self.config.timeout,
            "temperature": self.config.temperature,
            "stream": stream,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return litellm.completion(**kwargs)

    def completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        max_tokens: int | None = None,
    ) -> Any:
        """
        Call LLM with one retry on rate-limit using a different key.
        """
        first_key = self.key_manager.pick_random_key()
        try:
            logger.debug("LiteLLM call with key: %s", first_key.env_name)
            return self._completion_with_key(
                api_key=first_key.value,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise LiteLLMServiceUnavailableError(
                    "LLM service unavailable. Please try again later."
                ) from exc

            logger.warning(
                "Rate limit on key %s. Retrying once with another key.",
                first_key.env_name,
            )
            second_key = self.key_manager.pick_random_key(exclude_env_name=first_key.env_name)
            try:
                return self._completion_with_key(
                    api_key=second_key.value,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    stream=stream,
                    max_tokens=max_tokens,
                )
            except Exception as second_exc:
                if _is_rate_limit_error(second_exc):
                    raise LiteLLMRateLimitError(
                        "LLM service is rate-limited right now. Please try again shortly."
                    ) from second_exc
                raise LiteLLMServiceUnavailableError(
                    "LLM service unavailable. Please try again later."
                ) from second_exc

    def _stream_with_key(
        self,
        *,
        api_key: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "api_key": api_key,
            "timeout": self.config.timeout,
            "temperature": self.config.temperature,
            "stream": True,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return litellm.completion(**kwargs)

    def stream_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """
        Call LLM in streaming mode with one retry on startup rate-limit.

        Supports optional *tools* so tool-call detection can happen on the
        same streaming pass used for the final answer (avoids a redundant
        non-streaming call).
        """
        first_key = self.key_manager.pick_random_key()
        try:
            logger.debug("LiteLLM stream call with key: %s", first_key.env_name)
            return self._stream_with_key(
                api_key=first_key.value,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise LiteLLMServiceUnavailableError(
                    "LLM service unavailable. Please try again later."
                ) from exc
            second_key = self.key_manager.pick_random_key(exclude_env_name=first_key.env_name)
            try:
                return self._stream_with_key(
                    api_key=second_key.value,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=max_tokens,
                )
            except Exception as second_exc:
                if _is_rate_limit_error(second_exc):
                    raise LiteLLMRateLimitError(
                        "LLM service is rate-limited right now. Please try again shortly."
                    ) from second_exc
                raise LiteLLMServiceUnavailableError(
                    "LLM service unavailable. Please try again later."
                ) from second_exc
