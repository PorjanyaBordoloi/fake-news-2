"""
Scoring Utility

Simplified scoring — now each claim gets its own truth_score from Gemini,
so this module provides helpers for aggregating per-claim scores into an
overall article-level score when needed.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def aggregate_verdicts(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-claim verdicts into an overall article summary.

    Args:
        verdicts: List of verdict dicts from Agent 3 (fact_checker).
                  Each has: claim_text, verdict, truth_score, explanation, citations.

    Returns:
        {
            "overall_score": int (0-100, average of truth_scores),
            "overall_verdict": str (majority verdict label),
            "total_claims": int,
            "verdict_breakdown": {"SUPPORTED": n, "CONTRADICTED": n, ...},
        }
    """
    if not verdicts:
        return {
            "overall_score": 50,
            "overall_verdict": "UNVERIFIED",
            "total_claims": 0,
            "verdict_breakdown": {},
        }

    # Compute average truth score
    scores = [v.get("truth_score", 50) for v in verdicts]
    overall_score = round(sum(scores) / len(scores))

    # Count verdicts
    breakdown: dict[str, int] = {}
    for v in verdicts:
        label = v.get("verdict", "UNVERIFIED")
        breakdown[label] = breakdown.get(label, 0) + 1

    # Majority verdict
    overall_verdict = max(breakdown, key=breakdown.get)

    return {
        "overall_score": max(0, min(100, overall_score)),
        "overall_verdict": overall_verdict,
        "total_claims": len(verdicts),
        "verdict_breakdown": breakdown,
    }
