"""Evidence Retrieval Agent — LangGraph node.

Purpose: Gather supporting/refuting web evidence for the most important claims.

What it does:
  1. Takes extracted claims from AgentState.
  2. Sorts by checkworthiness_score; limits to top MAX_CLAIMS_PER_RUN.
  3. Uses Tavily search to fetch relevant web snippets per claim.
  4. Normalizes each result to {url, content} and truncates snippet length.
  5. Runs retrieval concurrently with a thread pool for speed.
  6. Outputs an evidence_map keyed by claim text.

Technologies: Tavily Search API, concurrent.futures ThreadPoolExecutor
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

MAX_CLAIMS_PER_RUN = 5
MAX_SNIPPET_LENGTH = 1500
MAX_RESULTS_PER_CLAIM = 5


def _read_env_var(*names: str) -> str:
    """Read the first non-empty environment variable from candidate names."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"').strip("'")
    return ""


def _tavily_search(claim_text: str, api_key: str) -> list[dict[str, str]]:
    """Synchronous Tavily search for a single claim (called inside thread pool)."""
    print(f"  🔎 Tavily search: \"{claim_text[:80]}...\"")

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": claim_text,
                "search_depth": "advanced",
                "max_results": MAX_RESULTS_PER_CLAIM,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        print(f"  ⚠️ Tavily error for claim: {exc}")
        return []

    results: list[dict[str, str]] = []
    for result in data.get("results", []):
        url = result.get("url", "")
        content = result.get("content", "")

        # Only keep valid HTTP/HTTPS URLs
        if not url or not url.startswith(("http://", "https://")):
            continue

        # Truncate snippet
        if len(content) > MAX_SNIPPET_LENGTH:
            content = content[:MAX_SNIPPET_LENGTH] + "…"

        results.append({"url": url, "content": content})

    print(f"  ✅ Found {len(results)} snippets for claim")
    return results


def evidence_retrieval_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: retrieve web evidence for top-N claims via Tavily."""
    print("--- EVIDENCE RETRIEVAL START ---")

    claims: list[dict[str, Any]] = state.get("claims", [])
    error = state.get("error", "")

    # If upstream had an error or produced no claims, pass through
    if error or not claims:
        print(f"--- EVIDENCE RETRIEVAL SKIPPED (error={error!r}, claims={len(claims)}) ---")
        return {
            "evidence_map": {},
        }

    tavily_api_key = _read_env_var("TAVILY_API_KEY", "tavily_api_key")
    if not tavily_api_key:
        print("--- EVIDENCE RETRIEVAL ERROR: Missing TAVILY_API_KEY ---")
        return {
            "evidence_map": {},
            "error": "Missing Tavily API key. Set TAVILY_API_KEY in .env",
        }

    # Sort by checkworthiness (descending) and limit
    sorted_claims = sorted(
        claims,
        key=lambda c: c.get("checkworthiness_score", 0),
        reverse=True,
    )
    top_claims = sorted_claims[:MAX_CLAIMS_PER_RUN]
    print(f"--- Processing {len(top_claims)} claims (of {len(claims)} total) ---")

    evidence_map: dict[str, list[dict[str, str]]] = {}

    # Concurrent retrieval using thread pool
    with ThreadPoolExecutor(max_workers=min(len(top_claims), 5)) as executor:
        future_to_claim = {
            executor.submit(_tavily_search, claim["claim_text"], tavily_api_key): claim["claim_text"]
            for claim in top_claims
        }

        for future in as_completed(future_to_claim):
            claim_text = future_to_claim[future]
            try:
                snippets = future.result()
                evidence_map[claim_text] = snippets
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠️ Thread error for claim \"{claim_text[:50]}\": {exc}")
                evidence_map[claim_text] = []

    total_snippets = sum(len(v) for v in evidence_map.values())
    print(f"--- EVIDENCE RETRIEVAL COMPLETE: {total_snippets} total snippets across {len(evidence_map)} claims ---")

    return {
        "evidence_map": evidence_map,
    }
