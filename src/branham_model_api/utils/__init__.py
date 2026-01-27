"""
Utility modules for the Branham Model API.
"""

from branham_model_api.utils.device import get_device, get_dtype
from branham_model_api.utils.git_branch import (
    get_git_branch,
    get_inference_backend,
    is_development_branch,
    is_production_branch,
    should_use_vllm,
)

__all__ = [
    # Device utilities
    "get_device",
    "get_dtype",
    # Git branch utilities
    "get_git_branch",
    "get_inference_backend",
    "is_development_branch",
    "is_production_branch",
    "should_use_vllm",
]
