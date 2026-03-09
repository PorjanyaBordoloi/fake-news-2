"""
Pydantic Response Models

Defines response schemas for the API endpoints matching the
3-agent pipeline output contracts.
"""

from typing import List, Optional
from pydantic import BaseModel


# ── Extracted Claim ──────────────────────────────────────
class ExtractedClaim(BaseModel):
    claim_text: str
    checkworthiness_score: int  # 1-10
    reasoning: str


# ── Article Metadata ─────────────────────────────────────
class ArticleMetadata(BaseModel):
    pub_date: Optional[str] = None
    author: Optional[str] = None
    source_domain: Optional[str] = None


# ── Evidence Item ────────────────────────────────────────
class EvidenceItem(BaseModel):
    url: str
    content: str  # truncated snippet


# ── Fact Check Verdict ───────────────────────────────────
class FactCheckVerdict(BaseModel):
    claim_text: str
    verdict: str  # SUPPORTED | CONTRADICTED | MISLEADING | UNVERIFIED
    truth_score: int  # 0-100
    explanation: str  # 2-sentence explanation
    citations: List[str]  # deduplicated evidence URLs


# ── Agent Chain Entry ────────────────────────────────────
class AgentChainEntry(BaseModel):
    agent: str
    status: str
    thought: str
    timestamp: str


# ── Full Pipeline Response ───────────────────────────────
class PipelineResponse(BaseModel):
    success: bool
    agent: str
    thought: str
    data: dict
    timestamp: str


# ── Simple Response (Chrome Extension) ───────────────────
class SimpleAnalysisResponse(BaseModel):
    verdicts: List[FactCheckVerdict]
