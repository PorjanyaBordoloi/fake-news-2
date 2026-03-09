"""Source Credibility Agent — LangGraph node.

Scores every evidence snippet by domain-tier registry + TLD heuristic,
re-ranks snippets best-first, and enriches each snippet with a `credibility`
key so the downstream fact-checker can weight sources intelligently.

Zero external API calls — pure dict lookup + urlparse.

Pipeline position:  evidence_retrieval → source_credibility → fact_checker
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from agents.claim_extraction import AgentState

try:
    from core.config import pipeline_config
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from core.config import pipeline_config

_sc_cfg: dict = pipeline_config.get("source_credibility", {})

# ---------------------------------------------------------------------------
# Tier score / label tables
# ---------------------------------------------------------------------------

TIER_SCORES: dict[int, int] = _sc_cfg.get(
    "tier_scores",
    {1: 95, 2: 82, 3: 70, 4: 35, 5: 10},
)

TIER_LABELS: dict[int, str] = {
    1: "Highly Credible",
    2: "Generally Credible",
    3: "Moderately Credible",
    4: "Low Credibility",
    5: "Social Media / Unverified",
}

# ---------------------------------------------------------------------------
# Domain registry  (60 entries)
# Each entry: {"tier": int, "bias": str}
# bias values: "neutral", "centre-left", "centre-right", "left", "right",
#               "pro-government", "tabloid"
# ---------------------------------------------------------------------------

DOMAIN_REGISTRY: dict[str, dict[str, Any]] = {
    # ── Tier 1 — Wire services & major fact-checkers ─────────────────────
    "reuters.com":         {"tier": 1, "bias": "neutral"},
    "apnews.com":          {"tier": 1, "bias": "neutral"},
    "afp.com":             {"tier": 1, "bias": "neutral"},
    "ptinews.com":         {"tier": 1, "bias": "neutral"},
    "ians.in":             {"tier": 1, "bias": "neutral"},
    "snopes.com":          {"tier": 1, "bias": "neutral"},
    "factcheck.org":       {"tier": 1, "bias": "neutral"},
    "politifact.com":      {"tier": 1, "bias": "neutral"},
    "fullfact.org":        {"tier": 1, "bias": "neutral"},
    "boomlive.in":         {"tier": 1, "bias": "neutral"},
    "altnews.in":          {"tier": 1, "bias": "neutral"},
    "thequint.com":        {"tier": 1, "bias": "neutral"},

    # ── Tier 2 — Major international newspapers & broadcasters ───────────
    "bbc.com":             {"tier": 2, "bias": "centre-left"},
    "bbc.co.uk":           {"tier": 2, "bias": "centre-left"},
    "theguardian.com":     {"tier": 2, "bias": "centre-left"},
    "nytimes.com":         {"tier": 2, "bias": "centre-left"},
    "washingtonpost.com":  {"tier": 2, "bias": "centre-left"},
    "economist.com":       {"tier": 2, "bias": "centre-right"},
    "ft.com":              {"tier": 2, "bias": "centre-right"},
    "wsj.com":             {"tier": 2, "bias": "centre-right"},
    "npr.org":             {"tier": 2, "bias": "centre-left"},
    "aljazeera.com":       {"tier": 2, "bias": "centre-left"},
    "dw.com":              {"tier": 2, "bias": "neutral"},
    "theatlantic.com":     {"tier": 2, "bias": "centre-left"},
    "time.com":            {"tier": 2, "bias": "centre-left"},
    "foreignpolicy.com":   {"tier": 2, "bias": "neutral"},

    # ── Tier 2 — Indian quality press ────────────────────────────────────
    "thehindu.com":        {"tier": 2, "bias": "centre-left"},
    "indianexpress.com":   {"tier": 2, "bias": "centre-left"},
    "livemint.com":        {"tier": 2, "bias": "neutral"},
    "hindustantimes.com":  {"tier": 2, "bias": "neutral"},
    "scroll.in":           {"tier": 2, "bias": "centre-left"},
    "thewire.in":          {"tier": 2, "bias": "left"},
    "ndtv.com":            {"tier": 2, "bias": "neutral"},
    "theprint.in":         {"tier": 2, "bias": "neutral"},
    "business-standard.com": {"tier": 2, "bias": "neutral"},
    "economictimes.indiatimes.com": {"tier": 2, "bias": "neutral"},

    # ── Tier 3 — Specialist, government-adjacent, or regional ────────────
    "eia.gov":             {"tier": 3, "bias": "neutral"},
    "who.int":             {"tier": 3, "bias": "neutral"},
    "worldbank.org":       {"tier": 3, "bias": "neutral"},
    "imf.org":             {"tier": 3, "bias": "neutral"},
    "statista.com":        {"tier": 3, "bias": "neutral"},
    "visualcapitalist.com": {"tier": 3, "bias": "neutral"},
    "ourworldindata.org":  {"tier": 3, "bias": "neutral"},
    "pewresearch.org":     {"tier": 3, "bias": "neutral"},
    "science.org":         {"tier": 3, "bias": "neutral"},
    "nature.com":          {"tier": 3, "bias": "neutral"},
    "msn.com":             {"tier": 3, "bias": "neutral"},
    "firstpost.com":       {"tier": 3, "bias": "centre-right"},
    "moneycontrol.com":    {"tier": 3, "bias": "neutral"},
    "newslaundry.com":     {"tier": 3, "bias": "centre-left"},

    # ── Tier 4 — Aggregators, partisan, low editorial standards ──────────
    "opindia.com":         {"tier": 4, "bias": "right"},
    "swarajyamag.com":     {"tier": 4, "bias": "right"},
    "postcard.news":       {"tier": 4, "bias": "right"},
    "sudarshannews.in":    {"tier": 4, "bias": "right"},
    "thefederal.com":      {"tier": 4, "bias": "left"},
    "republic.ru":         {"tier": 4, "bias": "pro-government"},
    "rt.com":              {"tier": 4, "bias": "pro-government"},
    "newsweek.com":        {"tier": 4, "bias": "neutral"},

    # ── Tier 5 — Social media (no editorial control) ─────────────────────
    "twitter.com":         {"tier": 5, "bias": "neutral"},
    "x.com":               {"tier": 5, "bias": "neutral"},
    "facebook.com":        {"tier": 5, "bias": "neutral"},
    "instagram.com":       {"tier": 5, "bias": "neutral"},
    "youtube.com":         {"tier": 5, "bias": "neutral"},
    "tiktok.com":          {"tier": 5, "bias": "neutral"},
    "reddit.com":          {"tier": 5, "bias": "neutral"},
    "t.me":                {"tier": 5, "bias": "neutral"},
    "whatsapp.com":        {"tier": 5, "bias": "neutral"},
    "telegram.org":        {"tier": 5, "bias": "neutral"},
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """Extract bare domain (no www., lowercase) from a URL string."""
    try:
        netloc = urlparse(str(url).strip()).netloc.lower()
        return netloc.removeprefix("www.")
    except Exception:
        return ""


def _heuristic_score(domain: str) -> dict[str, Any]:
    """Fall-back scoring based on TLD when domain not in registry."""
    if domain.endswith(".gov") or ".gov." in domain:
        return {"tier": 3, "score": 80, "label": TIER_LABELS[3], "bias": "neutral"}
    if domain.endswith(".edu") or ".edu." in domain:
        return {"tier": 3, "score": 75, "label": TIER_LABELS[3], "bias": "neutral"}
    if domain.endswith(".org"):
        return {"tier": 4, "score": 45, "label": TIER_LABELS[4], "bias": "neutral"}
    return {"tier": 4, "score": 35, "label": TIER_LABELS[4], "bias": "unknown"}


def score_domain(domain: str) -> dict[str, Any]:
    """Return a credibility dict for *domain* (bare, no www. prefix).

    Keys returned: domain, tier, score, label, bias
    """
    bare = domain.lower().removeprefix("www.")
    if bare in DOMAIN_REGISTRY:
        entry = DOMAIN_REGISTRY[bare]
        tier = entry["tier"]
        return {
            "domain": bare,
            "tier": tier,
            "score": TIER_SCORES[tier],
            "label": TIER_LABELS[tier],
            "bias": entry.get("bias", "unknown"),
        }
    # Not in registry — use heuristic
    result = _heuristic_score(bare)
    return {"domain": bare, **result}


def _score_snippet(snippet: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *snippet* with a ``credibility`` key added."""
    url = str(snippet.get("url", ""))
    domain = _extract_domain(url)
    credibility = score_domain(domain)
    return {**snippet, "credibility": credibility}


def _log_rerank(
    original: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    claim_text: str,
) -> None:
    """Print a one-line rerank notice for observability."""
    if not original or not ranked:
        return
    orig_domain = _extract_domain(str(original[0].get("url", "")))
    ranked_domain = _extract_domain(str(ranked[0].get("url", "")))
    if orig_domain != ranked_domain:
        print(
            f"[CREDIBILITY RERANK] '{claim_text[:60]}' — "
            f"moved '{ranked_domain}' above '{orig_domain}'"
        )
    else:
        print(
            f"[CREDIBILITY] '{claim_text[:60]}' — "
            f"'{orig_domain}' already leads (no rerank needed)"
        )


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def source_credibility_node(state: "AgentState") -> dict[str, Any]:
    """Score, re-rank and enrich evidence snippets by domain credibility.

    Reads  ``state["evidence_map"]``
    Writes ``state["evidence_map"]``  (enriched + re-ranked, in-place replacement)
           ``state["credibility_map"]``  (same structure, always present after node)

    On any unexpected error the original ``evidence_map`` is preserved and
    ``credibility_map`` is set to an empty dict so the pipeline can continue.
    """
    try:
        evidence_map: dict[str, list[dict[str, Any]]] = state.get("evidence_map", {})

        enriched_evidence_map: dict[str, list[dict[str, Any]]] = {}
        credibility_map: dict[str, list[dict[str, Any]]] = {}

        for claim_text, snippets in evidence_map.items():
            if not snippets:
                enriched_evidence_map[claim_text] = []
                credibility_map[claim_text] = []
                continue

            # Score every snippet
            scored: list[dict[str, Any]] = [_score_snippet(s) for s in snippets]

            # Sort descending by credibility score (stable sort preserves
            # original order for ties so highest-quality lead snippet wins)
            ranked = sorted(
                scored,
                key=lambda s: s.get("credibility", {}).get("score", 0),
                reverse=True,
            )

            _log_rerank(scored, ranked, claim_text)

            enriched_evidence_map[claim_text] = ranked
            credibility_map[claim_text] = [
                s["credibility"] for s in ranked
            ]

        return {
            "evidence_map": enriched_evidence_map,
            "credibility_map": credibility_map,
        }

    except Exception as exc:  # noqa: BLE001
        print(f"[source_credibility_node] ERROR — pipeline continues: {exc}")
        return {
            "credibility_map": {},
        }
