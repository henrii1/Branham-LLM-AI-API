"""
Health check endpoints.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """
    Health check endpoint.
    
    Should verify:
    - Indices are loaded
    - Models are loaded
    - System is ready to serve requests
    """
    # TODO: Implement proper health checks
    return {"status": "healthy", "ready": "false"}

