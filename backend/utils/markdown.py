"""
Markdown Report Generator

Formats evidence retrieval results as a readable markdown report.
Used for logging / debugging / optional export.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_evidence_report(evidence_map: dict[str, list[dict[str, str]]]) -> str:
    """
    Generate a markdown report from the evidence_map produced by Agent 2.

    Args:
        evidence_map: Dict keyed by claim_text, values are lists of {url, content} dicts.

    Returns:
        Formatted markdown report string.
    """
    if not evidence_map:
        return "# Evidence Report\n\nNo evidence was retrieved."

    lines: list[str] = ["# Evidence Report\n"]

    for i, (claim_text, snippets) in enumerate(evidence_map.items(), 1):
        lines.append(f"## Claim {i}: \"{claim_text}\"\n")
        lines.append(f"**Snippets Found:** {len(snippets)}\n")

        if not snippets:
            lines.append("- ⚠️ No evidence found for this claim\n")
            continue

        for j, snippet in enumerate(snippets, 1):
            url = snippet.get("url", "N/A")
            content = snippet.get("content", "N/A")
            lines.append(f"### Source {j}")
            lines.append(f"- **URL:** {url}")
            lines.append(f"- **Content:** {content[:500]}{'…' if len(content) > 500 else ''}")
            lines.append("")

        lines.append("---\n")

    return "\n".join(lines)
