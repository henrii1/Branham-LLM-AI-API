"""
Main FastAPI application entry point.
"""
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from branham_model_api.api.routes import chat, health

app = FastAPI(
    title="Branham Model API",
    description="Production-grade RAG pipeline + multilingual generator with strict grounding",
    version="0.1.0",
)

# CORS middleware (configure properly in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure from settings
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(health.router, prefix="/api", tags=["health"])


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Branham Model API", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

