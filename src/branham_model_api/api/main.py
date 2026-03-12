"""
Main FastAPI application entry point.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from branham_model_api.api.routes import chat, health

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Preload the chat runtime inside the serving process before accepting traffic.

    This is the only warm path that removes first-request latency from model/index
    initialization, because the loaded runtime lives in the same process as Uvicorn.
    """
    _ = app
    logger.info("Preloading chat runtime at startup")
    chat.get_chat_runtime()
    logger.info("Chat runtime preloaded")
    yield


app = FastAPI(
    title="Branham Model API",
    description="Production-grade RAG pipeline + multilingual generator with strict grounding",
    version="0.1.0",
    lifespan=lifespan,
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

