"""
Response schemas for API endpoints.
"""
from typing import Literal

from pydantic import BaseModel, Field


class Reference(BaseModel):
    """
    Reference to a sermon chunk.
    
    As per Section 10.1 and 1.2 (canonical reference system).
    """

    date_id: str = Field(..., description="Sermon date ID (dd-mm-yy--M/E)")
    paragraph_start: int = Field(..., description="Starting paragraph number")
    paragraph_end: int = Field(..., description="Ending paragraph number")
    chunk_ids: list[str] = Field(..., description="List of chunk IDs used")
    sermon_title: str | None = Field(default=None, description="Optional sermon title")


class ChatResponse(BaseModel):
    """
    Response schema for POST /chat endpoint.
    
    As per Section 10.1 of .cursorrules.
    """

    answer: str = Field(..., description="Generated answer with inline references")
    mode: Literal["quote", "synthesis", "refusal"] = Field(
        ..., 
        description="Response mode: quote (direct quote), synthesis (narrative), refusal (can't answer)"
    )
    references: list[Reference] = Field(
        ..., 
        description="List of sermon references used to ground the answer"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Brother Branham explained the third pull in [47-0412--M: ¶12–¶15]...",
                    "mode": "synthesis",
                    "references": [
                        {
                            "date_id": "47-0412--M",
                            "paragraph_start": 12,
                            "paragraph_end": 15,
                            "chunk_ids": ["47-0412--M_chunk_003", "47-0412--M_chunk_004"],
                            "sermon_title": "Faith Is The Substance",
                        }
                    ],
                }
            ]
        }
    }

