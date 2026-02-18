"""
Request schemas for API endpoints.
"""
from typing import Optional

from pydantic import AliasChoices, BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request schema for POST /chat endpoint.
    
    As per Section 10.1 of .cursorrules.
    """

    conversation_id: str = Field(
        ...,
        validation_alias=AliasChoices("conversation_id", "session_id"),
        serialization_alias="conversation_id",
        description="Unique conversation identifier (stable across turns)",
    )
    user_language: str | None = Field(
        default=None,
        description="Optional ISO/BCP-47 language code (e.g., 'en', 'es', 'fr')"
    )
    query: str = Field(..., description="User's question about the sermons")
    
    # Optional fields
    history_window: Optional[list[dict[str, str]]] = Field(
        default=None,
        description="Optional recent conversation turns for LLM continuity"
    )
    conversation_summary: Optional[str] = Field(
        default=None,
        description="Optional compact summary used primarily for retrieval context"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                    "query": "What did Brother Branham say about the third pull?",
                    "conversation_summary": "User is asking about the third pull teaching and prior answer discussed 1963 sermon context.",
                }
            ]
        }
    }

