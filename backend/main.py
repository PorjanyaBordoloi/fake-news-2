"""
main.py — FastAPI app entry point.

This file:
  1. Creates the FastAPI app
  2. Configures CORS for React frontend + Chrome Extension
  3. Defines the streaming and simple analysis endpoints
  4. Uses the compiled LangGraph pipeline from claim_extraction
       claim_extraction → evidence_retrieval → fact_checker
  5. Runs uvicorn
"""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

try:
    from agents.agent0_multilingual import SUPPORTED_LANGUAGES as _MULTILINGUAL_LANGUAGES
except Exception:  # noqa: BLE001
    _MULTILINGUAL_LANGUAGES: dict = {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lifespan — startup / shutdown hooks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_history()
    logger.info("🚀 Fake News Detector API starting up...")
    yield
    logger.info("🛑 Fake News Detector API shutting down...")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# App instance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = FastAPI(
    title="Fake News Detector API",
    version="1.0.0",
    lifespan=lifespan,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORS — allow React frontend + Chrome Extension origins
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# History — file-backed store
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_HISTORY_FILE = Path(__file__).parent / "history.json"
_history: list[dict] = []


def _load_history():
    global _history
    if _HISTORY_FILE.exists():
        try:
            with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
                _history = json.load(f)
        except Exception:
            _history = []


def _persist_history():
    with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(_history, f, indent=2)


def _make_history_entry(url: str, result: dict, source: str = "webapp") -> dict:
    verdicts = result.get("verdicts") or []
    explanations = result.get("explanations") or {}
    scores = [v.get("truth_score", 50) for v in verdicts if isinstance(v, dict)]
    avg_score = round(sum(scores) / len(scores)) if scores else 0
    overall = explanations.get("overall_credibility")
    if not overall:
        overall = (
            "CREDIBLE" if avg_score >= 70
            else "MIXED" if avg_score >= 50
            else "LOW CREDIBILITY" if avg_score >= 30
            else "UNRELIABLE"
        )
    return {
        "id": str(uuid.uuid4()),
        "url": url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_credibility": overall,
        "avg_score": avg_score,
        "bottom_line": explanations.get("bottom_line", ""),
        "verdicts": verdicts,
        "explanations": explanations,
        "source": source,
    }


def _append_history(entry: dict):
    _history.insert(0, entry)
    if len(_history) > 100:
        del _history[100:]
    _persist_history()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper: build initial state and invoke graph
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _build_initial_state(user_input: str, top_n: int = 3) -> dict:
    """Build a fresh AgentState dict for the LangGraph pipeline."""
    return {
        "user_input": user_input,
        "raw_markdown": "",
        "claims": [],
        "evidence_map": {},
        "credibility_map": {},
        "explanations": {},
        "verdicts": [],
        "top_n": top_n,
        "pub_date": None,
        "author": None,
        "source_domain": None,
        "error": "",
    }


def run_pipeline(user_input: str, top_n: int = 3) -> dict:
    """
    Run the full 5-agent LangGraph pipeline synchronously.
      1. claim_extraction     → "What should we fact-check?"
      2. evidence_retrieval   → "What evidence do we have?"
      3. source_credibility   → "How trustworthy are these sources?"
      4. fact_checker          → "Given this evidence, what is the verdict?"
      5. explanation_generator → "How do we explain this to a human?"
    """
    from agents.claim_extraction import claim_extraction_graph

    initial_state = _build_initial_state(user_input, top_n=top_n)
    result = claim_extraction_graph.invoke(initial_state)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.get("/api/analyze-stream")
async def analyze_stream(user_input: str):
    """
    Streaming endpoint for the website (shows chain of thought).
    Returns SSE stream of agent execution events.
    """
    from agents.claim_extraction import claim_extraction_graph

    async def event_generator():
        initial_state = _build_initial_state(user_input)
        _detected_lang_name: str | None = None  # track across agent0_pre → agent0_post
        # Stream events from the compiled LangGraph
        for event in claim_extraction_graph.stream(initial_state):
            for node_name, node_output in event.items():
                node_data = node_output or {}
                payload = {
                    "agent": node_name,
                    "status": "error" if node_data.get("error") else "success",
                    "data": node_data,
                }
                yield f"data: {json.dumps(payload)}\n\n"

                # Agent 0 pre — emit language detection log
                if node_name == "agent0_pre" and node_data.get("is_translated"):
                    src_lang = node_data.get("source_language", "")
                    _detected_lang_name = _MULTILINGUAL_LANGUAGES.get(src_lang, src_lang)
                    yield f"data: {json.dumps({'type': 'agent_log', 'symbol': '→', 'message': f'Detected {_detected_lang_name} — translating to English via Sarvam AI'})}\n\n"

                # Agent 0 post — emit localization log
                if node_name == "agent0_post" and node_data.get("localized_output") and _detected_lang_name:
                    yield f"data: {json.dumps({'type': 'agent_log', 'symbol': '\u2713', 'message': f'Verdicts localized back to {_detected_lang_name}'})}\n\n"

        yield f"data: {json.dumps({'status': 'complete'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@app.post("/api/analyze-simple")
async def analyze_simple(user_input: str):
    """
    Simple endpoint for Chrome extension (no streaming).
    Returns the full pipeline result and auto-saves to history.
    """
    result = run_pipeline(user_input)
    _append_history(_make_history_entry(user_input, result, source="extension"))
    return result


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# History endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HistorySaveRequest(BaseModel):
    url: str
    verdicts: list = []
    explanations: dict = {}
    source: str = "webapp"


@app.get("/api/history")
async def get_history():
    """Return all saved analysis history entries."""
    return _history


@app.post("/api/history")
async def save_history(req: HistorySaveRequest):
    """Save a history entry (called by frontend after stream completes)."""
    entry = _make_history_entry(
        req.url,
        {"verdicts": req.verdicts, "explanations": req.explanations},
        source=req.source,
    )
    _append_history(entry)
    return {"ok": True, "id": entry["id"]}


@app.delete("/api/history/{entry_id}")
async def delete_history_entry(entry_id: str):
    """Delete a single history entry by ID."""
    global _history
    _history = [e for e in _history if e.get("id") != entry_id]
    _persist_history()
    return {"ok": True}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Translate endpoint — used by frontend claim translate button
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TranslateRequest(BaseModel):
    text: str
    source_language: str = ""  # optional: if empty, auto-detected via Sarvam


@app.post("/api/translate")
async def translate_to_english(req: TranslateRequest):
    """
    Translate arbitrary text to English using Sarvam AI.
    Auto-detects source language if not provided.
    Used by the frontend 'Translate to English' button on claim cards.
    """
    try:
        from agents.agent0_multilingual import (
            SUPPORTED_LANGUAGES,
            _detect_language,
            _heuristic_detect,
            _translate_to_english,
        )

        text = req.text.strip()
        if not text:
            return {"translated_text": text, "source_language": "en-IN"}

        source_lang = req.source_language.strip()

        # Determine source language
        if not source_lang:
            source_lang, confidence = _detect_language(text)
            # If API + heuristic both say English or confidence too low, skip
            if source_lang.startswith("en"):
                return {"translated_text": text, "source_language": source_lang}
            # If API returned its fallback (en-IN, 1.0) but text has non-ASCII,
            # run the heuristic directly as a second opinion
            if confidence == 1.0 and source_lang == "en-IN":
                source_lang, confidence = _heuristic_detect(text)

        if source_lang.startswith("en"):
            return {"translated_text": text, "source_language": source_lang}

        translated = _translate_to_english(text, source_lang)
        lang_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
        return {
            "translated_text": translated,
            "source_language": source_lang,
            "source_language_name": lang_name,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[/api/translate] error: %s", exc)
        return {"translated_text": req.text, "source_language": "", "error": str(exc)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Uvicorn entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_includes=["*.yaml"])
