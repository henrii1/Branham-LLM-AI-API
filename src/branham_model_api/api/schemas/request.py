"""
Request schemas for API endpoints.
"""
from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request schema for POST /chat endpoint.
    
    As per Section 10.1 of .cursorrules.
    """

    session_id: str = Field(..., description="Unique session identifier")
    user_language: str = Field(
        ..., 
        description="ISO/BCP-47 language code (e.g., 'en', 'es', 'fr')"
    )
    query: str = Field(..., description="User's question about the sermons")
    
    # Optional fields
    history_window: Optional[list[dict[str, str]]] = Field(
        default=None,
        description="Recent conversation turns for context"
    )
    conversation_summary: Optional[str] = Field(
        default=None,
        description="Summary of conversation (provided on first message or when it changes)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "550e8400-e29b-41d4-a716-446655440000",
                    "user_language": "en",
                    "query": "What did Brother Branham say about the third pull?",
                }
            ]
        }
    }

