"""Fact-checking decision node for claim verification.

This module consumes claims with retrieved evidence snippets and emits strict,
UI-ready verdict objects for downstream presentation.
"""

from __future__ import annotations

import os
import importlib
import time
import traceback
from enum import Enum
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

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

_fc_cfg: dict = pipeline_config.get("fact_checker", {})
MAX_CITATIONS: int = _fc_cfg.get("max_citations", 3)

try:
    from utils.helpers import read_env_var as _read_env_var
except ModuleNotFoundError:
    import sys as _sys2
    from pathlib import Path as _Path2
    _sys2.path.insert(0, str(_Path2(__file__).resolve().parent.parent))
    from utils.helpers import read_env_var as _read_env_var
NO_EVIDENCE_EXPLANATION = (
    "Insufficient reliable evidence was retrieved for this claim. "
    "The claim remains unverified at this time."
)
LLM_ERROR_EXPLANATION = (
    "The model could not produce a reliable fact-check verdict for this claim. "
    "The claim is marked unverified pending further evidence."
)


class VerdictLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    MISLEADING = "MISLEADING"
    UNVERIFIED = "UNVERIFIED"


class FactCheckVerdict(BaseModel):
    claim_text: str = Field(min_length=1)
    verdict: VerdictLabel
    truth_score: int = Field(ge=0, le=100)
    explanation: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    reasoning_summary: str = Field(
        default="",
        description="Key points from the reasoning process",
    )
    confidence_level: str = Field(
        default="MEDIUM",
        description="HIGH / MEDIUM / LOW based on evidence coverage",
    )
    evidence_gaps: list[str] = Field(
        default_factory=list,
        description="Sub-parts of the claim not addressed by evidence",
    )

    @field_validator("evidence_gaps", "citations", mode="before")
    @classmethod
    def _coerce_str_to_list(cls, value: object) -> object:
        """Allow the model to return a plain string instead of a JSON array."""
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        return value


REASONING_PROMPT = """
You are a fact-checking analyst. Analyze the claim against the provided evidence snippets only — do not use outside knowledge.

Rules:
- Only cite information present in the snippets
- Mark anything absent from evidence as "Gap"
- Do NOT output a JSON or a verdict label

Your output must cover:
1. Evidence summary: one line per snippet — "[Domain] (score/100): <what it says>"
2. Sub-claim check: break the claim into checkable parts; for each: CONFIRMED / CONTRADICTED / INFERRED / GAP
3. Contradictions: note any conflicting snippets; higher credibility score wins
4. Verdict reasoning: 2-3 sentences — what the evidence shows, what gaps remain, what verdict this points to
""".strip()


VERDICT_PROMPT = """
You are a fact-check verdict recorder.

A senior analyst has already completed the reasoning process.
Your job is to convert that reasoning into a precise structured verdict.

Rules:
- Base your verdict ONLY on the reasoning document provided
- Do not add new analysis or override the analyst's conclusions
- Select the verdict label that best matches the reasoning
- Set truth_score based on confidence level in reasoning:
    HIGH confidence + SUPPORTED → 85-100
    MEDIUM confidence + SUPPORTED → 60-84
    LOW confidence → 40-59
    CONTRADICTED → 0-30
    MISLEADING → 30-60
- Citations must only come from URLs in the evidence snippets
- Explanation must summarize the reasoning in 2-3 sentences

Verdict labels:
- SUPPORTED: core claim confirmed by credible evidence
- CONTRADICTED: core claim directly refuted by evidence
- MISLEADING: technically true but deceptive framing or missing context
- UNVERIFIED: insufficient evidence to confirm or deny

You MUST respond with a valid JSON object only. No explanation outside the JSON.
The JSON must contain exactly these keys:
  claim_text, verdict, truth_score, explanation, citations,
  reasoning_summary, confidence_level, evidence_gaps
""".strip()





def _build_fact_checker_llm(model: str | None = None) -> Any:
    from langchain_groq import ChatGroq

    groq_api_key = _read_env_var("GROQ_API_KEY", "groq_api_key")
    if not groq_api_key:
        raise RuntimeError("Missing Groq API key. Set GROQ_API_KEY in .env")

    chosen_model = model or _fc_cfg.get("verdict_model", "llama-3.3-70b-versatile")
    llm = ChatGroq(
        model=chosen_model,
        api_key=groq_api_key,
        temperature=_fc_cfg.get("verdict_temperature", 0),
    )
    return llm.with_structured_output(
        FactCheckVerdict,
        method=_fc_cfg.get("verdict_method", "json_mode"),
    )


def _build_gemini_fact_checker_llm() -> Any:
    """Gemini Flash fallback when Groq daily token quota is exhausted."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    gemini_api_key = _read_env_var("GEMINI_API_KEY", "gemini_api_key")
    if not gemini_api_key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY in .env")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_api_key,
        temperature=0,
    )
    return llm.with_structured_output(
        FactCheckVerdict,
        method="json_mode",
    )


def _build_reasoning_llm() -> Any:
    """
    Plain LLM with no structured output constraint.
    Used for Call 1 (free-form reasoning). Intentionally unconstrained
    so the model can think in natural language without schema pressure.
    """
    from langchain_groq import ChatGroq

    groq_api_key = _read_env_var("GROQ_API_KEY", "groq_api_key")
    if not groq_api_key:
        raise RuntimeError("Missing Groq API key. Set GROQ_API_KEY in .env")

    return ChatGroq(
        model=_fc_cfg.get("reasoning_model", "llama-3.3-70b-versatile"),
        api_key=groq_api_key,
        temperature=_fc_cfg.get("reasoning_temperature", 0),
        max_tokens=_fc_cfg.get("reasoning_max_tokens", 1500),
    )


def _run_reasoning_step(
    claim_text: str,
    evidence_context: str,
    llm: Any,
) -> str:
    """
    Call 1: Free-form chain-of-thought reasoning.
    Returns reasoning document as plain text.
    Raises ValueError if output is suspiciously short (model skipped steps).
    Retries up to 3 times on 429 rate-limit errors with exponential backoff.
    """
    reasoning_prompt = ChatPromptTemplate.from_messages([
        ("system", REASONING_PROMPT),
        (
            "human",
            "Claim to analyze:\n{claim_text}\n\n"
            "Evidence snippets:\n{evidence_context}",
        ),
    ])

    chain = reasoning_prompt | llm

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response = chain.invoke({
                "claim_text": claim_text,
                "evidence_context": evidence_context,
            })
            reasoning_text = response.content
            if len(reasoning_text.strip()) < 100:
                raise ValueError(
                    f"Reasoning output too short — model may have skipped steps: "
                    f"{reasoning_text}"
                )
            return reasoning_text
        except Exception as exc:  # noqa: BLE001
            err_str = str(exc)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait = (2 ** attempt) * 5  # 5s, 10s, 20s
                print(f"--- RATE LIMIT (reasoning, attempt {attempt + 1}) — waiting {wait}s ---")
                time.sleep(wait)
                last_exc = exc
            else:
                raise
    raise last_exc  # type: ignore[misc]


def _is_http_url(value: str) -> bool:
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _build_claim_context(
    claim_text: str, snippets: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    normalized_blocks: list[str] = []
    allowed_urls: list[str] = []

    for idx, snippet in enumerate(snippets, start=1):
        url = str(snippet.get("url", "")).strip()
        content = str(snippet.get("content", "")).strip()
        if not content or not _is_http_url(url):
            continue

        allowed_urls.append(url)
        cred: dict[str, Any] | None = snippet.get("credibility")  # type: ignore[assignment]
        if cred:
            cred_line = (
                f"Credibility: {cred.get('score', '?')}/100 "
                f"| {cred.get('label', 'Unknown')} "
                f"| Bias: {cred.get('bias', 'unknown')}"
            )
            normalized_blocks.append(
                f"Snippet {idx}:\nURL: {url}\n{cred_line}\nContent: {content}"
            )
        else:
            normalized_blocks.append(
                f"Snippet {idx}:\nURL: {url}\nContent: {content}"
            )

    evidence_context = "\n\n".join(normalized_blocks)
    return evidence_context, allowed_urls


def _canonicalize_url(url: str) -> str:
    """Normalise URL for comparison: lowercase scheme+host, strip trailing slash, drop fragment."""
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/") or "/"
        normalized = f"{scheme}://{netloc}{path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        # Fragment intentionally dropped — #section variants should match the base URL.
        return normalized
    except Exception:  # noqa: BLE001
        return url.strip()


def _normalize_citations(citations: list[str], allowed_urls: list[str]) -> list[str]:
    # Build canonical-form → original evidence URL mapping (first occurrence wins).
    canonical_to_original: dict[str, str] = {}
    for original in allowed_urls:
        canon = _canonicalize_url(original)
        if canon not in canonical_to_original:
            canonical_to_original[canon] = original

    cleaned: list[str] = []
    seen: set[str] = set()  # tracks canonical forms already accepted

    for raw in citations:
        candidate = str(raw or "").strip()
        if not candidate or not _is_http_url(candidate):
            continue
        canon = _canonicalize_url(candidate)
        if canon not in canonical_to_original or canon in seen:
            continue
        # Emit the evidence-side URL, not the model's potentially-mutated variant.
        cleaned.append(canonical_to_original[canon])
        seen.add(canon)
        if len(cleaned) >= MAX_CITATIONS:
            break

    return cleaned


def _fallback_unverified(claim_text: str, reason: str) -> dict[str, Any]:
    explanation = NO_EVIDENCE_EXPLANATION
    if reason == "llm_error":
        explanation = LLM_ERROR_EXPLANATION

    return FactCheckVerdict(
        claim_text=str(claim_text or "").strip() or "Unknown claim",
        verdict=VerdictLabel.UNVERIFIED,
        truth_score=50,
        explanation=explanation,
        citations=[],
    ).model_dump()


def fact_checker_node(state: "AgentState") -> "AgentState":
    try:
        claims = state.get("claims", [])
        evidence_map = state.get("evidence_map", {})
        verdicts: list[dict[str, Any]] = []

        reasoning_llm = None
        structured_llm = None

        verdict_prompt = ChatPromptTemplate.from_messages([
            ("system", VERDICT_PROMPT),
            (
                "human",
                "Claim:\n{claim_text}\n\n"
                "Evidence snippets (for citation reference only):\n"
                "{evidence_context}\n\n"
                "Analyst reasoning document:\n{reasoning_text}\n\n"
                "Available citation URLs (use only these):\n{available_urls}",
            ),
        ])

        for claim in claims:
            claim_text = str(claim.get("claim_text", "")).strip()
            print(f"--- FACT-CHECKING CLAIM: {claim_text} ---")

            snippets = evidence_map.get(claim_text, [])
            if not snippets:
                fallback = _fallback_unverified(claim_text, reason="no_evidence")
                verdicts.append(fallback)
                print(
                    f"--- VERDICT: {fallback['verdict']} "
                    f"({fallback['truth_score']}) ---"
                )
                continue

            try:
                evidence_context, allowed_urls = _build_claim_context(
                    claim_text, snippets
                )
                if not evidence_context:
                    fallback = _fallback_unverified(claim_text, reason="no_evidence")
                    verdicts.append(fallback)
                    print(
                        f"--- VERDICT: {fallback['verdict']} "
                        f"({fallback['truth_score']}) ---"
                    )
                    continue

                # Lazy init both LLMs
                if reasoning_llm is None:
                    reasoning_llm = _build_reasoning_llm()
                if structured_llm is None:
                    structured_llm = _build_fact_checker_llm()

                # ── CALL 1: Free-form reasoning ──────────────────────────
                print("--- REASONING STEP (CALL 1) ---")
                reasoning_text = _run_reasoning_step(
                    claim_text=claim_text,
                    evidence_context=evidence_context,
                    llm=reasoning_llm,
                )
                print(
                    f"--- REASONING COMPLETE "
                    f"({len(reasoning_text)} chars) ---"
                )

                # ── CALL 2: Structured verdict using reasoning ────────────
                print("--- VERDICT STEP (CALL 2) ---")
                chain = verdict_prompt | structured_llm

                # Retry with backoff on 429; fall back to 8b if 70b daily limit exhausted
                last_exc: Exception | None = None
                response = None
                for attempt in range(3):
                    try:
                        response = chain.invoke({
                            "claim_text": claim_text,
                            "evidence_context": evidence_context,
                            "reasoning_text": reasoning_text,
                            "available_urls": "\n".join(allowed_urls),
                        })
                        break
                    except Exception as exc:  # noqa: BLE001
                        err_str = str(exc)
                        if "429" in err_str or "rate_limit" in err_str.lower():
                            if attempt < 2:
                                wait = (2 ** attempt) * 5  # 5s, 10s
                                print(f"--- RATE LIMIT (verdict, attempt {attempt + 1}) — waiting {wait}s ---")
                                time.sleep(wait)
                                last_exc = exc
                            else:
                                # Final attempt: fall back to Gemini
                                print("--- RATE LIMIT on Groq — falling back to Gemini Flash ---")
                                structured_llm = _build_gemini_fact_checker_llm()
                                chain = verdict_prompt | structured_llm
                                response = chain.invoke({
                                    "claim_text": claim_text,
                                    "evidence_context": evidence_context,
                                    "reasoning_text": reasoning_text,
                                    "available_urls": "\n".join(allowed_urls),
                                })
                                break
                        else:
                            raise

                if response is None:
                    if last_exc is not None:
                        raise last_exc  # type: ignore[misc]
                    raise ValueError(
                        "Structured LLM returned None — model produced no output for this claim."
                    )

                if isinstance(response, FactCheckVerdict):
                    parsed = response
                else:
                    parsed = FactCheckVerdict.model_validate(response)

                normalized = parsed.model_dump()
                normalized["claim_text"] = claim_text
                normalized["citations"] = _normalize_citations(
                    normalized.get("citations", []),
                    allowed_urls,
                )

                validated = FactCheckVerdict.model_validate(normalized)
                verdict_payload = validated.model_dump()
                verdicts.append(verdict_payload)
                print(
                    f"--- VERDICT: {verdict_payload['verdict']} "
                    f"({verdict_payload['truth_score']}) "
                    f"[{verdict_payload.get('confidence_level', '?')}] ---"
                )

            except Exception:  # noqa: BLE001
                tb_str = traceback.format_exc()
                print(
                    f"--- FACT-CHECK ERROR ---\n"
                    f"  Claim: {claim_text!r}\n"
                    f"  {tb_str}"
                )
                fallback = _fallback_unverified(claim_text, reason="llm_error")
                verdicts.append(fallback)
                print(
                    f"--- VERDICT: {fallback['verdict']} "
                    f"({fallback['truth_score']}) ---"
                )

        updated_state = dict(state)
        updated_state["verdicts"] = verdicts
        return updated_state

    except Exception as exc:  # noqa: BLE001
        updated_state = dict(state)
        updated_state["verdicts"] = updated_state.get("verdicts", [])
        updated_state["error"] = str(exc)
        return updated_state
