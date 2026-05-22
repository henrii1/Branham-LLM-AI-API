"""
LiteLLM client wrapper for multi-provider LLM generation.

Supports OpenRouter, DeepSeek direct, and any LiteLLM-compatible provider.
Provider is selected via config/default.yaml → llm.provider.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import litellm

from branham_model_api.generation.api_keys import LLMKeyManager, MixedApiKeyRecord, MixedLLMKeyManager

logger = logging.getLogger(__name__)

if os.getenv("LITELLM_DEBUG", "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
    # Very verbose; enable only when debugging upstream failures.
    litellm._turn_on_debug()


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
    reasoning: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None


def _looks_like_openrouter_route(*, model: str, base_url: str | None) -> bool:
    if str(model or "").strip().lower().startswith("openrouter/"):
        return True
    if base_url and "openrouter.ai" in str(base_url).lower():
        return True
    return False


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


def _safe_messages_stats(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Return non-sensitive stats about the prompt payload (no full content)."""
    total_chars = 0
    role_counts: dict[str, int] = {}
    for m in messages or []:
        role = str(m.get("role") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        content = m.get("content")
        if isinstance(content, str):
            total_chars += len(content)
    return {
        "message_count": len(messages or []),
        "total_content_chars": total_chars,
        "role_counts": role_counts,
    }


def _extract_exception_debug(exc: Exception) -> dict[str, Any]:
    """
    Best-effort extraction of upstream error attributes.

    Important: do NOT include secrets (api keys) or full prompt content.
    """
    debug: dict[str, Any] = {
        "exc_type": type(exc).__name__,
        "exc_str": str(exc),
        "status_code": getattr(exc, "status_code", None),
    }
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            debug["response_status_code"] = getattr(resp, "status_code", None)
        except Exception:
            pass
        try:
            headers = getattr(resp, "headers", None)
            if headers:
                for k in (
                    "x-request-id",
                    "request-id",
                    "x-ratelimit-remaining",
                    "retry-after",
                    "cf-ray",
                ):
                    if k in headers:
                        debug[f"response_header_{k}"] = headers.get(k)
        except Exception:
            pass
        try:
            text = getattr(resp, "text", None)
            if isinstance(text, str) and text:
                debug["response_text_head"] = text[:500]
        except Exception:
            pass
    return debug


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
        if (
            self.config.reasoning
            and _looks_like_openrouter_route(model=self.config.model, base_url=self.config.base_url)
        ):
            kwargs["reasoning"] = self.config.reasoning
        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body
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
                logger.error(
                    "Upstream LLM completion failed (non-rate-limit).",
                    extra={
                        "llm_model": self.config.model,
                        "llm_base_url": self.config.base_url,
                        "llm_stream": bool(stream),
                        "llm_tool_count": len(tools or []) if tools is not None else 0,
                        "llm_tool_choice": tool_choice,
                        "llm_key_env": first_key.env_name,
                        **_safe_messages_stats(messages),
                        **_extract_exception_debug(exc),
                    },
                    exc_info=True,
                )
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
                logger.error(
                    "Upstream LLM completion failed after retry (non-rate-limit).",
                    extra={
                        "llm_model": self.config.model,
                        "llm_base_url": self.config.base_url,
                        "llm_stream": bool(stream),
                        "llm_tool_count": len(tools or []) if tools is not None else 0,
                        "llm_tool_choice": tool_choice,
                        "llm_key_env_first": first_key.env_name,
                        "llm_key_env_second": second_key.env_name,
                        **_safe_messages_stats(messages),
                        **_extract_exception_debug(second_exc),
                    },
                    exc_info=True,
                )
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
        if (
            self.config.reasoning
            and _looks_like_openrouter_route(model=self.config.model, base_url=self.config.base_url)
        ):
            kwargs["reasoning"] = self.config.reasoning
        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body
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
                logger.error(
                    "Upstream LLM stream startup failed (non-rate-limit).",
                    extra={
                        "llm_model": self.config.model,
                        "llm_base_url": self.config.base_url,
                        "llm_stream": True,
                        "llm_tool_count": len(tools or []) if tools is not None else 0,
                        "llm_tool_choice": tool_choice,
                        "llm_key_env": first_key.env_name,
                        **_safe_messages_stats(messages),
                        **_extract_exception_debug(exc),
                    },
                    exc_info=True,
                )
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
                logger.error(
                    "Upstream LLM stream startup failed after retry (non-rate-limit).",
                    extra={
                        "llm_model": self.config.model,
                        "llm_base_url": self.config.base_url,
                        "llm_stream": True,
                        "llm_tool_count": len(tools or []) if tools is not None else 0,
                        "llm_tool_choice": tool_choice,
                        "llm_key_env_first": first_key.env_name,
                        "llm_key_env_second": second_key.env_name,
                        **_safe_messages_stats(messages),
                        **_extract_exception_debug(second_exc),
                    },
                    exc_info=True,
                )
                raise LiteLLMServiceUnavailableError(
                    "LLM service unavailable. Please try again later."
                ) from second_exc


@dataclass(frozen=True)
class LiteLLMRouteConfig:
    """Per-provider route config for mixed mode."""

    model: str
    base_url: str | None


class LiteLLMMixedClient:
    """
    Mixed-provider LiteLLM client.

    Selection policy:
    - Uniform random across all available keys loaded by MixedLLMKeyManager.
      (So 4 DeepSeek keys + 1 OpenRouter key => expected ~4:1 usage split.)
    - Retry once on rate-limit with a different key (possibly different provider).
    """

    def __init__(
        self,
        *,
        routes: dict[str, LiteLLMRouteConfig],
        timeout: float = 30.0,
        temperature: float = 0.2,
        key_manager: MixedLLMKeyManager,
    ) -> None:
        self.routes = dict(routes)
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.key_manager = key_manager
        if not self.key_manager.has_keys:
            raise RuntimeError("No API keys found for mixed provider mode.")

    def _completion_with_key(
        self,
        *,
        key: MixedApiKeyRecord,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        max_tokens: int | None = None,
    ) -> Any:
        route = self.routes.get(key.route_id)
        if route is None:
            raise RuntimeError(f"Unknown mixed route_id: {key.route_id}")
        kwargs: dict[str, Any] = {
            "model": route.model,
            "messages": messages,
            "api_key": key.value,
            "timeout": self.timeout,
            "temperature": self.temperature,
            "stream": stream,
        }
        if route.base_url:
            kwargs["base_url"] = route.base_url
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
        first_key = self.key_manager.pick_random_key()
        try:
            logger.debug("LiteLLM mixed call with key: %s", first_key.env_name)
            return self._completion_with_key(
                key=first_key,
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
            second_key = self.key_manager.pick_random_key(exclude_env_name=first_key.env_name)
            try:
                return self._completion_with_key(
                    key=second_key,
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

    def stream_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        first_key = self.key_manager.pick_random_key()
        try:
            logger.debug("LiteLLM mixed stream call with key: %s", first_key.env_name)
            return self._completion_with_key(
                key=first_key,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=True,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise LiteLLMServiceUnavailableError(
                    "LLM service unavailable. Please try again later."
                ) from exc
            second_key = self.key_manager.pick_random_key(exclude_env_name=first_key.env_name)
            try:
                return self._completion_with_key(
                    key=second_key,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    stream=True,
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
