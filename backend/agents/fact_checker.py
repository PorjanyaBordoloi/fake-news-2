"""Fact Checker Agent — LangGraph node.

Purpose: Convert claim + evidence into final user-facing verdicts.

What it does:
  1. For each claim, builds evidence context from valid HTTP/HTTPS snippets.
  2. Uses Groq LLM with structured JSON schema for FactCheckVerdict.
  3. Enforces one of 4 labels: SUPPORTED, CONTRADICTED, MISLEADING, UNVERIFIED.
  4. Returns per claim: verdict, truth_score (0-100), 2-sentence explanation,
     cleaned citations (deduplicated/canonicalized from evidence URLs only).
  5. Falls back to safe UNVERIFIED output if no evidence or LLM error.

Technologies: Groq LLM (structured schema output)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

VALID_VERDICTS = {"SUPPORTED", "CONTRADICTED", "MISLEADING", "UNVERIFIED"}

FACT_CHECK_PROMPT = """You are a fact-checking agent. Given the CLAIM and EVIDENCE below, produce a structured verdict.

CLAIM:
{claim}

EVIDENCE:
{evidence}

You MUST respond with a single valid JSON object (no markdown fences, no extra text) with exactly these keys:
- "verdict": one of "SUPPORTED", "CONTRADICTED", "MISLEADING", "UNVERIFIED"
- "truth_score": integer 0-100
- "explanation": exactly 2 sentences explaining your verdict
- "citations": array of evidence URLs that support your verdict (only from the evidence above, deduplicated)

Rules:
- If evidence strongly confirms the claim → SUPPORTED (truth_score 70-100)
- If evidence directly contradicts the claim → CONTRADICTED (truth_score 0-30)
- If evidence partially supports but key details are wrong or exaggerated → MISLEADING (truth_score 30-60)
- If evidence is insufficient or inconclusive → UNVERIFIED (truth_score 40-60)
- Only include URLs that actually appear in the EVIDENCE section above
"""


def _read_env_var(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"').strip("'")
    return ""


def _canonicalize_url(url: str) -> str:
    """Normalize a URL: strip trailing slashes, fragments, lowercase scheme+host."""
    try:
        parsed = urlparse(url.strip())
        clean = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            parsed.params,
            parsed.query,
            "",  # drop fragment
        ))
        return clean
    except Exception:  # noqa: BLE001
        return url.strip()


def _deduplicate_urls(urls: list[str], evidence_urls: set[str]) -> list[str]:
    """Deduplicate and filter citations to only those present in retrieved evidence."""
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        canonical = _canonicalize_url(url)
        if canonical not in seen and canonical in evidence_urls:
            seen.add(canonical)
            result.append(canonical)
    return result


def _build_evidence_context(snippets: list[dict[str, str]]) -> tuple[str, set[str]]:
    """Build a text block of evidence snippets and collect canonical evidence URLs."""
    if not snippets:
        return "", set()

    evidence_urls: set[str] = set()
    lines: list[str] = []

    for i, snippet in enumerate(snippets, 1):
        url = snippet.get("url", "")
        content = snippet.get("content", "")
        if not url.startswith(("http://", "https://")):
            continue
        canonical = _canonicalize_url(url)
        evidence_urls.add(canonical)
        lines.append(f"[Source {i}] {url}\n{content}")

    return "\n\n".join(lines), evidence_urls


def _safe_unverified(claim_text: str) -> dict[str, Any]:
    """Fallback verdict when no evidence or LLM error."""
    return {
        "claim_text": claim_text,
        "verdict": "UNVERIFIED",
        "truth_score": 50,
        "explanation": "Insufficient evidence was found to verify this claim. The claim remains unverified pending further information.",
        "citations": [],
    }


def _parse_llm_response(raw_text: str) -> dict[str, Any]:
    """Parse Groq response, stripping markdown fences if present."""
    cleaned = raw_text.strip()

    # Strip ```json ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    return json.loads(cleaned)


def _generate_verdict_for_claim(
    client: Groq,
    claim_text: str,
    evidence_context: str,
    evidence_urls: set[str],
) -> dict[str, Any]:
    """Call Groq for a single claim and return a validated verdict dict."""
    prompt = FACT_CHECK_PROMPT.format(claim=claim_text, evidence=evidence_context)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ Groq API error for claim \"{claim_text[:50]}\": {exc}")
        return _safe_unverified(claim_text)

    try:
        parsed = _parse_llm_response(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"  ⚠️ Groq JSON parse error: {exc}")
        return _safe_unverified(claim_text)

    # Validate and sanitize verdict
    verdict = str(parsed.get("verdict", "UNVERIFIED")).upper().strip()
    if verdict not in VALID_VERDICTS:
        verdict = "UNVERIFIED"

    truth_score = parsed.get("truth_score", 50)
    if not isinstance(truth_score, (int, float)):
        truth_score = 50
    truth_score = max(0, min(100, int(truth_score)))

    explanation = str(parsed.get("explanation", ""))
    if not explanation:
        explanation = "The LLM did not provide an explanation. The claim remains unverified."

    raw_citations = parsed.get("citations", [])
    if not isinstance(raw_citations, list):
        raw_citations = []
    citations = _deduplicate_urls(
        [str(c) for c in raw_citations if isinstance(c, str)],
        evidence_urls,
    )

    return {
        "claim_text": claim_text,
        "verdict": verdict,
        "truth_score": truth_score,
        "explanation": explanation,
        "citations": citations,
    }


def fact_checker_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: generate verdicts for each claim using evidence + Groq."""
    print("--- FACT CHECKER START ---")

    claims: list[dict[str, Any]] = state.get("claims", [])
    evidence_map: dict[str, list[dict[str, str]]] = state.get("evidence_map", {})
    error = state.get("error", "")

    # If upstream had a fatal error or no claims, return empty verdicts
    if error or not claims:
        print(f"--- FACT CHECKER SKIPPED (error={error!r}, claims={len(claims)}) ---")
        return {
            "verdicts": [],
        }

    groq_api_key = _read_env_var("GROQ_API_KEY", "groq_api_key")
    if not groq_api_key:
        print("--- FACT CHECKER ERROR: Missing GROQ_API_KEY ---")
        return {
            "verdicts": [_safe_unverified(c["claim_text"]) for c in claims],
            "error": "Missing Groq API key. Set GROQ_API_KEY in .env",
        }

    client = Groq(api_key=groq_api_key)
    verdicts: list[dict[str, Any]] = []

    for claim in claims:
        claim_text = claim.get("claim_text", "")
        if not claim_text:
            continue

        print(f"  🔍 Processing claim: \"{claim_text[:80]}...\"")

        # Get evidence for this claim
        snippets = evidence_map.get(claim_text, [])
        evidence_context, evidence_urls = _build_evidence_context(snippets)

        if not evidence_context:
            print(f"  ⚠️ No evidence found — defaulting to UNVERIFIED")
            verdicts.append(_safe_unverified(claim_text))
            continue

        verdict = _generate_verdict_for_claim(client, claim_text, evidence_context, evidence_urls)
        verdicts.append(verdict)

        print(f"  ✅ Verdict: {verdict['verdict']} (truth_score: {verdict['truth_score']})")

    # Summary
    verdict_counts = {}
    for v in verdicts:
        label = v["verdict"]
        verdict_counts[label] = verdict_counts.get(label, 0) + 1

    summary_parts = [f"{count} {label}" for label, count in sorted(verdict_counts.items())]
    print(f"--- FACT CHECKER COMPLETE: {', '.join(summary_parts)} ---")

    return {
        "verdicts": verdicts,
    }
