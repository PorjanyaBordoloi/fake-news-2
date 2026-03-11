"""Claim Extraction Agent (Parser) built with LangGraph.

This module exposes a small single-node graph that:
1) Scrapes article content via Jina Reader
2) Sanitizes article text and extracts metadata
3) Extracts structured, checkable claims with Groq
4) Ranks claims by checkworthiness and keeps top-N

Designed to be extended with additional nodes (e.g., evidence retrieval).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, NotRequired, Optional, TypedDict
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import AliasChoices, BaseModel, Field, ValidationError, field_validator

try:
    from utils.helpers import read_env_var as _read_env_var
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from utils.helpers import read_env_var as _read_env_var

try:
    from agents.evidence_retrieval import evidence_retrieval_node
except ModuleNotFoundError as exc:
    # Support direct script execution from backend/agents where "agents" isn't on sys.path.
    if exc.name != "agents":
        raise
    from evidence_retrieval import evidence_retrieval_node

try:
    from agents.fact_checker import fact_checker_node
except ModuleNotFoundError as exc:
    if exc.name != "agents":
        raise
    from fact_checker import fact_checker_node

try:
    from agents.source_credibility import source_credibility_node
except ModuleNotFoundError as exc:
    if exc.name != "agents":
        raise
    from source_credibility import source_credibility_node

try:
    from agents.explanation_generator import explanation_generator_node
except ModuleNotFoundError:
    from explanation_generator import explanation_generator_node

try:
    from agents.agent0_multilingual import agent0_pre_node, agent0_post_node
except ModuleNotFoundError:
    from agent0_multilingual import agent0_pre_node, agent0_post_node  # type: ignore[no-redef]

load_dotenv()

try:
    from core.config import pipeline_config
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from core.config import pipeline_config

_ce_cfg: dict = pipeline_config.get("claim_extraction", {})


class AgentState(TypedDict):
    user_input: str
    input_url: NotRequired[str]
    raw_markdown: str
    claims: list[dict[str, Any]]
    evidence_map: dict[str, list[dict[str, str]]]
    credibility_map: NotRequired[dict[str, list[dict[str, Any]]]]
    explanations: NotRequired[dict[str, Any]]
    verdicts: list[dict[str, Any]]
    top_n: Optional[int]
    pub_date: Optional[str]
    author: Optional[str]
    source_domain: Optional[str]
    error: str
    # --- Multilingual Agent 0 fields ---
    source_language: NotRequired[str]       # Sarvam language code e.g. "bn-IN"
    original_text: NotRequired[str]         # Raw input before translation
    is_translated: NotRequired[bool]        # True if Agent 0 translated input
    translated_input: NotRequired[str]      # English version passed to pipeline
    localized_output: NotRequired[dict]     # Verdicts translated back to source lang


class Claim(BaseModel):
    claim_text: str = Field(
        min_length=1,
        validation_alias=AliasChoices("claim_text", "text", "claim"),
    )
    checkworthiness_score: int = Field(ge=1, le=10)
    reasoning: str = Field(min_length=1)


class ClaimExtractionResult(BaseModel):
    claims: list[Claim]

    @field_validator("claims")
    @classmethod
    def validate_claims_non_empty(cls, value: list[Claim]) -> list[Claim]:
        if not value:
            raise ValueError("At least one claim is required")
        return value


CLAIM_PROMPT = """
You are a claim extraction engine for fact-checking.

Task:
- Extract checkable, factual claims from the article.

NEGATIVE CONSTRAINTS

STOP: Do not extract standalone entities.

A valid claim MUST:
- Contain a subject
- Contain a verb
- Contain a factual assertion

The claim must be answerable with TRUE or FALSE.

Invalid examples:
- "Odisha CM Mohan Majhi"
- "India"
- "The Prime Minister"

Valid examples:
- "Odisha CM Mohan Majhi stated that the state will allocate ₹500 crore for flood relief."
- "The government announced a new tax reform policy."

Rank each claim by how important it is to fact-check.

Prioritize claims that:
- Influence public perception
- Relate to politics or government actions
- Contain numbers, statistics, or policy announcements
- Could spread misinformation if false

CO-REFERENCE RESOLUTION

Rewrite every claim so it is fully self-contained.

Replace all pronouns (he, she, they, it, this, that) with the explicit person, organization, or entity mentioned in the article.

The claim must remain understandable even when isolated from the original article.

Output requirements:
- Respond with a valid JSON object following the structured schema exactly.
- Return claims with checkworthiness_score in the range 1 to 10.
- Keep reasoning concise and evidence-oriented.
""".strip()





URL_RE = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)


def is_url(text: str) -> bool:
    """Return True when text is an http(s) URL."""
    return bool(URL_RE.match(str(text or "").strip()))


def _is_low_context_text(text: str) -> bool:
    """Heuristic low-context gate for very short text rumors/headlines."""
    return len(str(text or "").strip().split()) < 3


def scrape_article(url: str) -> str:
    """Fetch article content from Jina Reader as markdown."""
    print(f"--- SCRAPING URL --- {url}")

    if not url:
        raise ValueError("input_url is required")

    jina_api_key = _read_env_var("JINA_READER_API_KEY", "jina_reader_api_key")
    if not jina_api_key:
        raise RuntimeError("Missing Jina API key. Set JINA_READER_API_KEY in .env")

    full_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {jina_api_key}",
        "X-Return-Format": "markdown",
        "Accept": "text/plain",
    }

    try:
        response = requests.get(full_url, headers=headers, timeout=45)
        if response.status_code in (404, 500):
            raise RuntimeError(
                f"Jina Reader returned HTTP {response.status_code} for URL: {url}"
            )
        response.raise_for_status()
        print("--- SCRAPING COMPLETE ---")
        return response.text
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise RuntimeError(f"Jina HTTP error ({status}) while scraping: {url}") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error while scraping article: {exc}") from exc


def sanitize_article_text(raw_text: str) -> str:
    """Strip leading website chrome/navigation noise from article text."""
    if not raw_text:
        return raw_text

    marker_positions: list[int] = []

    heading_match = re.search(r"(?m)^#\s+", raw_text)
    if heading_match:
        marker_positions.append(heading_match.start())

    marker_patterns = [
        r"(?i)\bPublished\b",
        r"(?i)\bUpdated\b",
        # Line-start anchor prevents mid-sentence matches (e.g. "Sort By" in nav).
        r"(?im)^\s*Written\s+[Bb]y\b",
        # Named-month dateline: "March 9, 2026" — reliable article-start anchor.
        r"(?im)^\s*(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}",
    ]
    for pattern in marker_patterns:
        match = re.search(pattern, raw_text)
        if match:
            marker_positions.append(match.start())

    if marker_positions:
        start_index = min(marker_positions)
        return raw_text[start_index:].strip()

    # Fallback: return from the first line with ≥25 words — article body sentences are
    # almost always this long; nav/menu items rarely are.
    for m in re.finditer(r"(?m)^(.+)$", raw_text):
        line = m.group(1).strip()
        if len(line.split()) >= 25:
            return raw_text[m.start():].strip()

    return raw_text.strip()


def extract_article_metadata(article_text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract publish/update date and author from sanitized article text."""
    if not article_text:
        return None, None

    pub_date: Optional[str] = None
    author: Optional[str] = None

    pub_patterns = [
        r"(?im)^\s*Published on\s*[:\-]\s*(.+)$",
        r"(?im)^\s*Published\s*[:\-]\s*(.+)$",
        r"(?im)^\s*Updated on\s*[:\-]\s*(.+)$",
        r"(?im)^\s*Updated\s*[:\-]\s*(.+)$",
    ]
    for pattern in pub_patterns:
        match = re.search(pattern, article_text)
        if match:
            pub_date = match.group(1).strip()
            break

    author_patterns = [
        r"(?im)^\s*By\s+([^\n\|]+)",
        r"(?im)^\s*Written by\s+([^\n\|]+)",
    ]
    for pattern in author_patterns:
        match = re.search(pattern, article_text)
        if match:
            author = match.group(1).strip(" -:")
            break

    return pub_date, author


def _truncate_markdown(markdown: str, limit: int = 6000) -> str:
    if len(markdown) > limit:
        print(f"--- TRUNCATING MARKDOWN TO {limit} CHARS ---")
    return markdown[:limit]


def _build_structured_llm() -> Any:
    """Build Groq LLM client configured for strict structured output."""
    groq_api_key = _read_env_var("GROQ_API_KEY", "groq_api_key")
    if not groq_api_key:
        raise RuntimeError("Missing Groq API key. Set GROQ_API_KEY in .env")

    llm = ChatGroq(
        model=_ce_cfg.get("model", "llama-3.1-8b-instant"),
        api_key=groq_api_key,
        temperature=_ce_cfg.get("temperature", 0),
    )
    # llama-3.1-8b-instant produces valid JSON but fails the tool-call wire
    # protocol, resulting in Groq HTTP 400 "tool_use_failed". json_mode uses
    # response_format={"type":"json_object"} instead of tool-use, which this
    # model handles correctly.
    return llm.with_structured_output(
        ClaimExtractionResult,
        method=_ce_cfg.get("structured_output_method", "json_mode"),
    )


def _rank_and_select_claims(claims: list[Claim], top_n: int) -> list[Claim]:
    sorted_claims = sorted(claims, key=lambda item: item.checkworthiness_score, reverse=True)
    normalized_top_n = max(1, top_n)
    return sorted_claims[:normalized_top_n]


def _extract_claims_with_llm(markdown: str, top_n: int) -> list[dict[str, Any]]:
    print("--- EXTRACTING CLAIMS WITH GROQ ---")

    structured_llm = _build_structured_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                CLAIM_PROMPT,
            ),
            (
                "human",
                "Article markdown (possibly truncated):\n{article_markdown}",
            ),
        ]
    )

    chain = prompt | structured_llm
    response = chain.invoke({"article_markdown": markdown})

    try:
        if isinstance(response, ClaimExtractionResult):
            parsed = response
        else:
            parsed = ClaimExtractionResult.model_validate(response)
    except ValidationError as exc:
        raise RuntimeError(f"Structured output validation failed: {exc}") from exc

    selected_claims = _rank_and_select_claims(parsed.claims, top_n=top_n)
    if not selected_claims:
        raise RuntimeError("Structured output returned zero claims after ranking")

    claims_payload = [claim.model_dump() for claim in selected_claims]

    print("--- CLAIM EXTRACTION COMPLETE ---")
    return claims_payload


def extraction_node(state: AgentState) -> AgentState:
    """LangGraph node: route URL/TEXT -> extract -> rank top-N."""
    resolved_input = str(state.get("user_input") or state.get("input_url") or "").strip()
    top_n = int(state.get("top_n", 3))
    evidence_map = state.get("evidence_map", {})
    verdicts = state.get("verdicts", [])
    truncated_markdown = ""
    pub_date: Optional[str] = None
    author: Optional[str] = None
    source_domain: Optional[str] = None
    input_is_url = is_url(resolved_input)

    if not resolved_input:
        return {
            "user_input": "",
            "raw_markdown": "",
            "claims": [],
            "evidence_map": evidence_map,
            "verdicts": verdicts,
            "pub_date": None,
            "author": None,
            "source_domain": None,
            "error": "user_input is required",
        }

    try:
        if input_is_url:
            print("--- DETECTED INPUT TYPE: URL ---")
            try:
                raw_markdown = scrape_article(resolved_input)
            except Exception as exc:  # noqa: BLE001
                return {
                    "user_input": resolved_input,
                    "raw_markdown": "",
                    "claims": [],
                    "evidence_map": evidence_map,
                    "verdicts": verdicts,
                    "pub_date": None,
                    "author": None,
                    "source_domain": None,
                    "error": f"URL scrape failed: {exc}",
                }

            source_domain = urlparse(resolved_input).netloc or None
            sanitized_markdown = sanitize_article_text(raw_markdown)
            pub_date, author = extract_article_metadata(sanitized_markdown)
            truncated_markdown = _truncate_markdown(
                sanitized_markdown,
                limit=_ce_cfg.get("markdown_truncation_limit", 10000),
            )
        else:
            print("--- DETECTED INPUT TYPE: TEXT ---")
            truncated_markdown = _truncate_markdown(
                resolved_input,
                limit=_ce_cfg.get("markdown_truncation_limit", 10000),
            )
            pub_date, author = extract_article_metadata(truncated_markdown)
            source_domain = None

            if _is_low_context_text(truncated_markdown):
                return {
                    "user_input": resolved_input,
                    "raw_markdown": truncated_markdown,
                    "claims": [],
                    "evidence_map": evidence_map,
                    "verdicts": verdicts,
                    "pub_date": pub_date,
                    "author": author,
                    "source_domain": source_domain,
                    "error": "Low Context",
                }

        claims = _extract_claims_with_llm(truncated_markdown, top_n=top_n)
        return {
            "user_input": resolved_input,
            "raw_markdown": truncated_markdown,
            "claims": claims,
            "evidence_map": evidence_map,
            "verdicts": verdicts,
            "pub_date": pub_date,
            "author": author,
            "source_domain": source_domain,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - we want robust terminal visibility in demo
        error_message = str(exc)
        print(f"--- EXTRACTION ERROR --- {error_message}")
        return {
            "user_input": resolved_input,
            "raw_markdown": truncated_markdown,
            "claims": [],
            "evidence_map": evidence_map,
            "verdicts": verdicts,
            "pub_date": pub_date,
            "author": author,
            "source_domain": source_domain,
            "error": error_message,
        }


def build_claim_extraction_graph():
    """Build and compile the full pipeline graph with multilingual support."""
    builder = StateGraph(AgentState)

    # Agent 0 — pre (language detection + translation)
    builder.add_node("agent0_pre",            agent0_pre_node)
    # Agents 1–5 — unchanged
    builder.add_node("claim_extraction",      extraction_node)
    builder.add_node("evidence_retrieval",    evidence_retrieval_node)
    builder.add_node("source_credibility",    source_credibility_node)
    builder.add_node("fact_checker",          fact_checker_node)
    builder.add_node("explanation_generator", explanation_generator_node)
    # Agent 0 — post (localize output back to source language)
    builder.add_node("agent0_post",           agent0_post_node)

    builder.add_edge(START,                   "agent0_pre")
    builder.add_edge("agent0_pre",            "claim_extraction")
    builder.add_edge("claim_extraction",      "evidence_retrieval")
    builder.add_edge("evidence_retrieval",    "source_credibility")
    builder.add_edge("source_credibility",    "fact_checker")
    builder.add_edge("fact_checker",          "explanation_generator")
    builder.add_edge("explanation_generator", "agent0_post")
    builder.add_edge("agent0_post",           END)

    return builder.compile()


claim_extraction_graph = build_claim_extraction_graph()


if __name__ == "__main__":
    sample_state: AgentState = {
        "user_input": "https://x.com/i/trending/2030881009990856970",
        "raw_markdown": "",
        "claims": [],
        "evidence_map": {},
        "credibility_map": {},
        "explanations": {},
        "verdicts": [],
        "top_n": 3,
        "pub_date": None,
        "author": None,
        "source_domain": None,
        "error": "",
    }

    print("--- RUNNING CLAIM EXTRACTION GRAPH ---")
    result = claim_extraction_graph.invoke(sample_state)

    print("--- FINAL STATE ---")
    print(json.dumps(result, indent=2))