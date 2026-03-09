"""
Gemini LLM Service

Wrapper for the Google Gemini API used by Agent 3 (Fact Checker)
to generate structured FactCheckVerdict outputs per claim.
"""

import os
import logging

from google import genai

logger = logging.getLogger(__name__)


class GeminiService:
    """Google Gemini API wrapper for structured fact-check verdicts."""

    def __init__(self):
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
        )

    async def generate_verdict(self, claim: str, evidence_context: str) -> str:
        """
        Generate a structured fact-check verdict for a claim using Gemini.

        Args:
            claim: The claim text to fact-check.
            evidence_context: Compiled evidence snippets from retrieved sources.

        Returns:
            LLM response text (JSON-formatted FactCheckVerdict string).
        """
        prompt = f"""You are a fact-checking agent. Given the claim and evidence below,
produce a structured verdict.

CLAIM:
{claim}

EVIDENCE:
{evidence_context}

Respond in JSON format with:
- verdict: one of "SUPPORTED", "CONTRADICTED", "MISLEADING", "UNVERIFIED"
- truth_score: integer 0-100
- explanation: exactly 2 sentences explaining your verdict
- citations: array of evidence URLs that support your verdict (deduplicated)
"""

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        return response.text
