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
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lifespan — startup / shutdown hooks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@asynccontextmanager
async def lifespan(app: FastAPI):
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
        # Stream events from the compiled LangGraph
        for event in claim_extraction_graph.stream(initial_state):
            for node_name, node_output in event.items():
                payload = {
                    "agent": node_name,
                    "status": "error" if node_output.get("error") else "success",
                    "data": node_output,
                }
                yield f"data: {json.dumps(payload)}\n\n"

        yield f"data: {json.dumps({'status': 'complete'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@app.post("/api/analyze-simple")
async def analyze_simple(user_input: str):
    """
    Simple endpoint for Chrome extension (no streaming).
    Returns the full pipeline result including verdicts.
    """
    result = run_pipeline(user_input)
    return result


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Uvicorn entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
