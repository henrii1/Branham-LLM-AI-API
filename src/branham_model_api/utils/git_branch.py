"""
Git branch detection utilities.

Used for branch-based inference backend selection:
- develop branch → Direct HuggingFace
- main branch + CUDA → vLLM
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def get_git_branch() -> str:
    """
    Detect the current git branch.
    
    Returns:
        Branch name (e.g., "main", "develop") or "unknown" if detection fails.
    
    Detection order:
    1. GIT_BRANCH environment variable (for CI/CD overrides)
    2. git rev-parse --abbrev-ref HEAD
    3. Fallback to "unknown"
    """
    # 1. Check environment variable (CI/CD override)
    env_branch = os.environ.get("GIT_BRANCH")
    if env_branch:
        # Handle refs/heads/main format from some CI systems
        if env_branch.startswith("refs/heads/"):
            return env_branch.replace("refs/heads/", "")
        return env_branch
    
    # 2. Try git command
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            if branch and branch != "HEAD":  # Detached HEAD state
                return branch
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    
    # 3. Fallback
    return "unknown"


def is_production_branch() -> bool:
    """Check if current branch is a production branch (main/master)."""
    branch = get_git_branch()
    return branch in ("main", "master")


def is_development_branch() -> bool:
    """Check if current branch is a development branch."""
    branch = get_git_branch()
    return branch in ("develop", "development", "dev")


def should_use_vllm() -> bool:
    """
    Determine if vLLM backend should be used.
    
    Returns True if:
    - Branch is main/master AND
    - CUDA is available
    
    Otherwise returns False (use direct HuggingFace).
    """
    import torch
    
    if not is_production_branch():
        return False
    
    return torch.cuda.is_available()


def get_inference_backend(config_override: str | None = None) -> str:
    """
    Get the inference backend to use.
    
    Args:
        config_override: Optional override from config ("auto", "huggingface", "vllm").
                        If None or "auto", uses branch-based selection.
    
    Returns:
        "vllm" or "huggingface"
    
    Selection logic:
    1. If config_override is "huggingface" or "vllm", use that directly
    2. If config_override is "auto" or None, use branch-based selection:
       - develop branch → huggingface
       - main branch + CUDA → vllm
       - main branch + no CUDA → huggingface (fallback)
    """
    # Check for explicit override
    if config_override and config_override != "auto":
        if config_override in ("huggingface", "vllm"):
            return config_override
        # Invalid value, fall through to auto
    
    # Auto selection based on branch + device
    return "vllm" if should_use_vllm() else "huggingface"
