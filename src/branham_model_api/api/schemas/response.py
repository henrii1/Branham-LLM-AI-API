"""
Response schemas for API endpoints.
"""
from typing import Literal

from pydantic import BaseModel, Field

class ExternalInfo(BaseModel):
    """External information metadata (present only when web search is used)."""

    disclaimer: str = Field(
        ...,
        description="Disclosure that content is from external/unverified sources",
    )
    sources: list[str] = Field(
        ...,
        description="External source URLs in markdown-friendly format",
    )


class ChatResponse(BaseModel):
    """
    Response schema for POST /chat endpoint.
    
    As per Section 10.1 of .cursorrules.
    """

    answer: str = Field(..., description="Generated answer with inline references")
    mode: Literal["answer", "refusal", "error"] = Field(
        ...,
        description="Final response mode"
    )
    external_info: ExternalInfo | None = Field(
        default=None,
        description="Present only when external web tool data was used",
    )
    conversation_summary: str | None = Field(
        default=None,
        description="Optional compact summary for frontend memory handoff",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Brother Branham explained the third pull in [47-0412M: ¶12–¶15]...",
                    "mode": "answer",
                    "conversation_summary": "Conversation has focused on third pull definition and context sermons from 1963.",
                }
            ]
        }
    }

