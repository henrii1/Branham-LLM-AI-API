"""
Generation layer (LiteLLM + API key selection).
"""

from .api_keys import LLMKeyManager, OpenRouterKeyManager
from .litellm_client import (
    LiteLLMClient,
    LiteLLMClientConfig,
    LiteLLMRateLimitError,
    LiteLLMServiceUnavailableError,
)

__all__ = [
    "LiteLLMClient",
    "LiteLLMClientConfig",
    "LiteLLMRateLimitError",
    "LiteLLMServiceUnavailableError",
    "LLMKeyManager",
    "OpenRouterKeyManager",
]
