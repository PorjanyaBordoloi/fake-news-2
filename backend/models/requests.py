"""
Pydantic Request Models

Defines request schemas for the API endpoints.
"""

from typing import Optional
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    """Request body for /api/analyze-stream and /api/analyze-simple."""

    user_input: str  # URL or plain text
    top_n: Optional[int] = 3  # Max number of claims to process
