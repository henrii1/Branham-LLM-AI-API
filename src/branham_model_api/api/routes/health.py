"""
Health check endpoints.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, Any]:
    """
    Health check endpoint.
    
    Should verify:
    - Indices are loaded
    - Models are loaded
    - System is ready to serve requests
    """
    # Import lazily so health route can be imported without eagerly loading ML deps.
    from branham_model_api.api.routes.chat import (  # noqa: WPS433 (runtime import is intentional)
        _get_expected_chat_bearer_key,
        get_chat_runtime,
    )

    checks: dict[str, Any] = {
        "bearer_key_configured": bool(_get_expected_chat_bearer_key()),
    }

    try:
        runtime = get_chat_runtime()
        # If we can build ChatRuntime, we can load indices + embedder + LLM client.
        checks["retrieval_pipeline_loaded"] = bool(getattr(runtime, "pipeline", None))
        checks["tool_registry_loaded"] = bool(getattr(runtime, "tool_registry", None))

        llm_client = getattr(runtime, "llm_client", None)
        checks["llm_client_loaded"] = bool(llm_client)
        key_mgr = getattr(llm_client, "key_manager", None) if llm_client else None
        checks["llm_key_count"] = int(getattr(key_mgr, "key_count", 0)) if key_mgr else 0

        ready = (
            checks["bearer_key_configured"]
            and checks["retrieval_pipeline_loaded"]
            and checks["tool_registry_loaded"]
            and checks["llm_client_loaded"]
            and checks["llm_key_count"] > 0
        )
        return {"status": "healthy", "ready": ready, "checks": checks}
    except Exception as exc:
        # Do not raise: Cloud Run health probes should get a structured response.
        return {
            "status": "healthy",
            "ready": False,
            "checks": checks,
            "error": f"{type(exc).__name__}: {exc}",
        }

