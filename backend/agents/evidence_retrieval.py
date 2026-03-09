"""Evidence Retrieval Node for claim verification support.

This module queries Tavily for the top claims and returns compact evidence
payloads suitable for downstream fact-checking.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

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

_er_cfg: dict = pipeline_config.get("evidence_retrieval", {})
MAX_CLAIMS_PER_RUN: int = _er_cfg.get("max_claims_per_run", 3)
MAX_SNIPPET_CHARS: int = _er_cfg.get("max_snippet_chars", 500)


def _read_env_var(*names: str) -> str:
	for name in names:
		value = os.getenv(name)
		if value:
			return value.strip().strip('"').strip("'")
	return ""


def _truncate_snippet(content: str, limit: int = MAX_SNIPPET_CHARS) -> str:
	return content[:limit].strip()


def _normalize_results(raw_response: Any) -> list[dict[str, str]]:
	if isinstance(raw_response, list):
		candidates = raw_response
	elif isinstance(raw_response, dict) and isinstance(raw_response.get("results"), list):
		candidates = raw_response["results"]
	else:
		candidates = []

	normalized: list[dict[str, str]] = []
	for item in candidates:
		if not isinstance(item, dict):
			continue

		url = str(item.get("url", "")).strip()
		content = str(item.get("content", "")).strip()
		if not url or not content:
			continue

		normalized.append({"url": url, "content": _truncate_snippet(content)})

	return normalized


def get_evidence_for_claim(claim_text: str) -> list[dict[str, str]]:
	print(f"--- SEARCHING WEB FOR: {claim_text} ---")

	clean_claim = str(claim_text or "").strip()
	if not clean_claim:
		print("--- EVIDENCE RESULTS: 0 ---")
		return []

	tavily_api_key = _read_env_var("TAVILY_API_KEY", "tavily_api_key")
	if not tavily_api_key:
		print("--- EVIDENCE RESULTS: 0 ---")
		return []

	os.environ["TAVILY_API_KEY"] = tavily_api_key

	try:
		tool = TavilySearchResults(
			search_depth=_er_cfg.get("tavily_search_depth", "basic"),
			max_results=_er_cfg.get("tavily_max_results", 5),
		)
		raw_response = tool.invoke(clean_claim)
		normalized = _normalize_results(raw_response)
		print(f"--- EVIDENCE RESULTS: {len(normalized)} ---")
		return normalized
	except Exception as exc:  # noqa: BLE001
		print(f"--- EVIDENCE RESULTS: 0 --- ({exc})")
		return []


def _claim_sort_key(claim: dict[str, Any]) -> int:
	try:
		return int(claim.get("checkworthiness_score", 0))
	except (TypeError, ValueError):
		return 0


def evidence_retrieval_node(state: "AgentState") -> "AgentState":
	try:
		claims = state.get("claims", [])
		sorted_claims = sorted(claims, key=_claim_sort_key, reverse=True)
		selected_claims = sorted_claims[:MAX_CLAIMS_PER_RUN]

		selected_claim_texts = [str(claim.get("claim_text", "")).strip() for claim in selected_claims]

		evidence_map: dict[str, list[dict[str, str]]] = {}
		futures_by_claim: dict[str, Any] = {}

		with ThreadPoolExecutor(max_workers=_er_cfg.get("max_parallel_workers", 2)) as executor:
			for claim_text in selected_claim_texts:
				if claim_text:
					futures_by_claim[claim_text] = executor.submit(get_evidence_for_claim, claim_text)

			# Preserve deterministic order for the resulting map.
			for claim_text in selected_claim_texts:
				if not claim_text:
					evidence_map[claim_text] = []
					continue
				future = futures_by_claim.get(claim_text)
				if future is None:
					evidence_map[claim_text] = []
					continue
				try:
					evidence_map[claim_text] = future.result()
				except Exception:  # noqa: BLE001
					evidence_map[claim_text] = []

		updated_state = dict(state)
		updated_state["evidence_map"] = evidence_map
		return updated_state
	except Exception as exc:  # noqa: BLE001
		updated_state = dict(state)
		updated_state["evidence_map"] = updated_state.get("evidence_map", {})
		updated_state["error"] = str(exc)
		return updated_state
