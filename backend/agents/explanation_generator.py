"""Explanation Generator — Final LangGraph node.

Consumes fact-checker verdicts, reasoning summaries, credibility scores,
and evidence gaps. Produces plain-English explanations readable by a
general audience at a 10th grade reading level.

One Gemini call per pipeline run (not per claim) — all verdicts are
explained in a single batched call to minimise quota usage.

Pipeline position: fact_checker → explanation_generator → END
"""

from __future__ import annotations

import importlib
import os
import traceback
from typing import TYPE_CHECKING, Any, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from agents.claim_extraction import AgentState

load_dotenv()

try:
    from core.config import pipeline_config
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from core.config import pipeline_config

_eg_cfg: dict = pipeline_config.get("explanation_generator", {})


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class VerdictExplanation(BaseModel):
    claim_text: str = Field(
        description="The exact claim text being explained"
    )
    plain_english: str = Field(
        description=(
            "3-4 sentences explaining what the verdict means in plain language. "
            "Must state specifically what the evidence showed, not just the verdict label."
        )
    )
    confidence_statement: str = Field(
        description=(
            "One sentence describing how confident the reader should be. "
            "Example: 'High confidence — confirmed by two major wire services.'"
        )
    )
    source_quality_note: str = Field(
        description=(
            "One sentence describing the quality of sources used. "
            "Must reference actual source tiers, not generic language."
        )
    )
    reader_advisory: Optional[str] = Field(
        default=None,
        description=(
            "One sentence warning for the reader if sources are weak, "
            "contradictions exist, or the verdict is MISLEADING. "
            "Return null if no advisory is needed."
        )
    )
    evidence_gaps_plain: Optional[str] = Field(
        default=None,
        description=(
            "One sentence describing what is still unknown or unconfirmed. "
            "Return null if no significant gaps exist."
        )
    )

    @field_validator("reader_advisory", "evidence_gaps_plain", mode="before")
    @classmethod
    def coerce_none_strings(cls, v: Any) -> Optional[str]:
        """
        Gemini sometimes returns 'None', 'null', 'N/A' as strings.
        Coerce these to actual None to prevent downstream type errors.
        """
        if isinstance(v, str) and v.lower().strip() in (
            "none", "null", "n/a", "", "not applicable", "no advisory needed",
            "no gaps", "no significant gaps", "no gaps exist",
        ):
            return None
        return v or None


class ExplanationOutput(BaseModel):
    verdicts_explained: list[VerdictExplanation] = Field(
        description="One explanation object per verdict, in the same order as input"
    )
    overall_credibility: str = Field(
        description=(
            "One of: CREDIBLE / MOSTLY CREDIBLE / MIXED / "
            "LOW CREDIBILITY / UNRELIABLE. "
            "Based on the aggregate of all verdicts and source quality."
        )
    )
    bottom_line: str = Field(
        description=(
            "One plain sentence summarizing the article's overall trustworthiness "
            "for the reader. Specific and honest — not generic."
        )
    )


# ---------------------------------------------------------------------------
# Prompt constant
# ---------------------------------------------------------------------------

EXPLANATION_SYSTEM_PROMPT = """
You are a fact-check communicator writing for a general audience.

You will receive a list of fact-check verdicts with technical reasoning.
Your job is to rewrite each verdict into plain, honest language that a
non-expert reader can immediately understand and act on.

WRITING RULES:
- Write at a 10th grade reading level
- No jargon: never use "sub-parts", "corroborating", "epistemically",
  "assert", "snippet", or "evidence context"
- Maximum 4 sentences for plain_english per verdict
- Be honest about uncertainty — do not oversell confidence
- Be specific — say what the evidence actually showed, not just
  "evidence supports this claim"
- Never say "the model" or "the analyst" — write as if you personally
  reviewed the sources

SOURCE QUALITY LANGUAGE — use exactly these phrases based on tier:
- Tier 1 (score 90-100): "confirmed by major wire services"
- Tier 2 (score 80-89):  "reported by established news outlets"
- Tier 3 (score 60-79):  "reported by specialist or regional sources"
- Tier 4 (score 30-59):  "sourced from outlets with limited editorial standards"
- Tier 5 (score 0-29):   "sourced primarily from social media — treat with caution"

VERDICT LANGUAGE GUIDE:
- SUPPORTED:     "This claim checks out." or "This appears to be accurate."
- CONTRADICTED:  "This claim is false based on available evidence."
- MISLEADING:    "This claim contains a misleading element." — then explain
                 specifically what part is true and what part is false or exaggerated
- UNVERIFIED:    "This claim could not be verified." — explain what is missing

READER ADVISORY RULES — always include reader_advisory when:
- Any citation has credibility score below 30 (social media)
- Verdict is MISLEADING
- Verdict is CONTRADICTED
- Sources contradict each other (noted in reasoning_summary)
- confidence_level from fact-checker is LOW

OVERALL CREDIBILITY RULES:
- CREDIBLE:        All verdicts SUPPORTED with HIGH confidence
- MOSTLY CREDIBLE: Most verdicts SUPPORTED, one MISLEADING or MEDIUM confidence
- MIXED:           Mix of SUPPORTED and UNVERIFIED/MISLEADING
- LOW CREDIBILITY: Multiple UNVERIFIED or one CONTRADICTED
- UNRELIABLE:      Multiple CONTRADICTED or fabricated claims detected

OUTPUT: Follow the structured schema exactly.
Produce one VerdictExplanation per verdict in the same order as input.
""".strip()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _read_env_var(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"').strip("'")
    return ""


def _build_explanation_llm() -> Any:
    """
    Gemini 2.5 Flash with structured output.
    Temperature 0.2 for natural prose — intentionally non-zero.
    """
    module = importlib.import_module("langchain_google_genai")
    ChatGoogleGenerativeAI = getattr(module, "ChatGoogleGenerativeAI")

    gemini_api_key = _read_env_var("GEMINI_API_KEY", "gemini_api_key")
    if not gemini_api_key:
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY in .env"
        )

    llm = ChatGoogleGenerativeAI(
        model=_eg_cfg.get("model", "gemini-2.5-flash"),
        google_api_key=gemini_api_key,
        temperature=_eg_cfg.get("temperature", 0.2),
    )
    return llm.with_structured_output(ExplanationOutput)


def _format_verdicts_for_prompt(
    verdicts: list[dict[str, Any]],
    credibility_map: dict[str, Any],
) -> str:
    """Serialize all verdicts into a structured text block for the prompt."""
    blocks: list[str] = []

    for idx, verdict in enumerate(verdicts, start=1):
        claim_text = verdict.get("claim_text", "")
        evidence_gaps = verdict.get("evidence_gaps", [])
        if isinstance(evidence_gaps, list):
            gaps_str = "; ".join(evidence_gaps) if evidence_gaps else "None"
        else:
            gaps_str = str(evidence_gaps) if evidence_gaps else "None"

        lines = [
            f"VERDICT {idx}:",
            f"Claim: {claim_text}",
            f"Verdict Label: {verdict.get('verdict', 'UNVERIFIED')}",
            f"Truth Score: {verdict.get('truth_score', 0)}/100",
            f"Confidence Level: {verdict.get('confidence_level', 'MEDIUM')}",
            f"Reasoning Summary: {verdict.get('reasoning_summary') or 'Not available'}",
            f"Evidence Gaps: {gaps_str}",
            f"Current Explanation: {verdict.get('explanation', '')}",
            "",
            "Source Quality for this claim:",
        ]

        cred_entries: list[dict[str, Any]] = credibility_map.get(claim_text, [])
        if cred_entries:
            for entry in cred_entries:
                domain = entry.get("domain", "unknown")
                score = entry.get("score", "?")
                label = entry.get("label", "Unknown")
                lines.append(f"  - {domain}: score {score}/100, {label}")
        else:
            lines.append("  - No credibility data available")

        blocks.append("\n".join(lines))

    return "\n---\n".join(blocks)


def _fallback_explanations(verdicts: list[dict[str, Any]]) -> ExplanationOutput:
    """Returns a minimal valid ExplanationOutput when the LLM call fails."""
    explained = [
        VerdictExplanation(
            claim_text=v.get("claim_text", "Unknown claim"),
            plain_english=(
                "This claim could not be explained at this time due to a processing error."
            ),
            confidence_statement="Confidence unknown — explanation generation failed.",
            source_quality_note="Source quality assessment unavailable.",
            reader_advisory=None,
            evidence_gaps_plain=None,
        )
        for v in verdicts
    ]
    return ExplanationOutput(
        verdicts_explained=explained,
        overall_credibility="MIXED",
        bottom_line=(
            "Explanation generation encountered an error. "
            "Please review the raw verdicts."
        ),
    )


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def explanation_generator_node(state: "AgentState") -> dict[str, Any]:
    """Generate plain-English explanations for all fact-check verdicts.

    Reads  ``state["verdicts"]`` and ``state["credibility_map"]``
    Writes ``state["explanations"]``

    Makes exactly ONE LLM call regardless of how many claims are present.
    On failure, returns a valid fallback so the pipeline always completes.
    """
    try:
        verdicts: list[dict[str, Any]] = state.get("verdicts", [])
        credibility_map: dict[str, Any] = state.get("credibility_map", {})

        # Guard: nothing to explain
        if not verdicts:
            updated_state = dict(state)
            updated_state["explanations"] = {}
            return updated_state

        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", EXPLANATION_SYSTEM_PROMPT),
            (
                "human",
                "Please explain the following fact-check verdicts:\n\n"
                "{verdicts_text}",
            ),
        ])

        # Format all verdicts into one text block
        verdicts_text = _format_verdicts_for_prompt(verdicts, credibility_map)

        # Single LLM call for the entire batch
        print("--- EXPLANATION GENERATOR ---")
        llm = _build_explanation_llm()
        chain = prompt | llm
        response = chain.invoke({"verdicts_text": verdicts_text})

        # Validate / coerce response
        if isinstance(response, ExplanationOutput):
            explanation_output = response
        else:
            explanation_output = ExplanationOutput.model_validate(response)

        print("--- EXPLANATIONS COMPLETE ---")
        print(f"--- OVERALL CREDIBILITY: {explanation_output.overall_credibility} ---")
        print(f"--- BOTTOM LINE: {explanation_output.bottom_line} ---")

        # Key by claim_text for easy frontend lookup
        explanations_by_claim: dict[str, Any] = {}
        for ve in explanation_output.verdicts_explained:
            explanations_by_claim[ve.claim_text] = ve.model_dump()

        updated_state = dict(state)
        updated_state["explanations"] = {
            "by_claim": explanations_by_claim,
            "overall_credibility": explanation_output.overall_credibility,
            "bottom_line": explanation_output.bottom_line,
        }
        return updated_state

    except Exception:
        print("[explanation_generator_node] ERROR — using fallback explanations")
        print(traceback.format_exc())

        verdicts = state.get("verdicts", [])
        fallback = _fallback_explanations(verdicts)

        explanations_by_claim = {}
        for ve in fallback.verdicts_explained:
            explanations_by_claim[ve.claim_text] = ve.model_dump()

        updated_state = dict(state)
        updated_state["explanations"] = {
            "by_claim": explanations_by_claim,
            "overall_credibility": fallback.overall_credibility,
            "bottom_line": fallback.bottom_line,
        }
        return updated_state
