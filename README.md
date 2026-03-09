<p align="center">
  <h1 align="center">🔍 Fake News Detector</h1>
  <p align="center">
    <strong>AI-Powered Multi-Agent Fact-Checking System</strong><br/>
    Built with LangGraph · Groq · Tavily · React · FastAPI
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Agents-5-blue?style=flat-square" alt="5 Agents"/>
  <img src="https://img.shields.io/badge/LLM-Groq_Llama_3-green?style=flat-square" alt="Groq"/>
  <img src="https://img.shields.io/badge/Framework-LangGraph-purple?style=flat-square" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Frontend-React_+_Vite-cyan?style=flat-square" alt="React"/>
  <img src="https://img.shields.io/badge/API-FastAPI-009688?style=flat-square" alt="FastAPI"/>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Why This Matters — Real-World Impact](#why-this-matters--real-world-impact)
- [System Architecture](#system-architecture)
- [The 5-Agent Pipeline — Deep Dive](#the-5-agent-pipeline--deep-dive)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Frontend Architecture](#frontend-architecture)
- [Chrome Extension](#chrome-extension)
- [Configuration Reference](#configuration-reference)
- [Error Handling & Resilience](#error-handling--resilience)
- [Performance & Rate Limits](#performance--rate-limits)

---

## Overview

The Fake News Detector is a production-grade, multi-agent AI system that autonomously fact-checks news articles in real time. Given any news URL or pasted text, it:

1. **Scrapes** the article content using Jina Reader
2. **Extracts** structured, checkable claims via LLM
3. **Retrieves** corroborating/contradicting evidence from the live web
4. **Scores** the credibility of each evidence source using domain-tier analysis
5. **Reasons** through the evidence using a 2-call Chain-of-Thought architecture
6. **Generates** plain-English explanations readable by any non-expert

The entire pipeline is orchestrated as a **LangGraph StateGraph** — a directed acyclic graph where each agent is a node with strict typed state contracts, ensuring deterministic execution, graceful fallbacks, and real-time SSE streaming to the frontend.

---

## Why This Matters — Real-World Impact

### The Misinformation Crisis

Misinformation is one of the defining challenges of the digital age. According to the World Economic Forum's 2024 Global Risks Report, **misinformation and disinformation rank as the #1 global risk** over the next two years — ahead of climate change, cyberattacks, and armed conflict.

- **UNESCO** estimates that 85% of the world's population is concerned about the impact of online disinformation.
- During the 2020 US elections alone, fact-checking organizations identified over **16,000 misleading claims** in a single quarter.
- In India, **WhatsApp misinformation** has been linked to mob violence, with at least 33 documented incidents between 2017-2019 (BBC).
- Health misinformation during COVID-19 — termed the "infodemic" by the WHO — led to at least **800 deaths worldwide** from consuming methanol or chlorine-based products promoted as cures.

### How This System Addresses the Problem

| Problem | How We Solve It |
|---------|-----------------|
| **Speed** — Fact-checking organizations take hours to days to verify a single claim | Our 5-agent pipeline returns verdicts in **30-60 seconds** |
| **Scale** — Manual fact-checkers cannot keep up with the volume of online content | The system processes any URL or text input **autonomously**, 24/7 |
| **Accessibility** — Fact-check reports are often published separately from viral content | The **Chrome Extension** puts the verdict directly where the user reads the article |
| **Source Bias** — Readers struggle to evaluate source credibility on their own | **Agent 3 (Source Credibility)** automatically scores 60+ domains and surfaces which tier each source belongs to |
| **Complexity** — Existing verdicts use jargon that non-experts cannot parse | **Agent 5 (Explanation Generator)** produces plain-English explanations at a 10th-grade reading level |
| **Transparency** — Black-box AI verdicts erode trust | Every verdict includes the **full reasoning chain**, evidence citations, confidence levels, and evidence gaps — the user sees exactly *why* a claim is supported or contradicted |

### Real-World Use Cases

- **Journalists** can instantly cross-reference claims in breaking news stories before publication
- **Educators** can use the tool to teach media literacy, showing students how claims map to evidence
- **Voters** can verify political claims during election cycles without waiting for traditional fact-checkers
- **Social media users** can check viral posts before sharing, breaking the misinformation chain at the individual level
- **Newsrooms** can integrate the API into their editorial workflows for semi-automated pre-publication verification
- **Researchers** studying misinformation can use the structured output (JSON verdicts with confidence scores) for large-scale analysis

### Impact in the Indian Context

India has **800+ million internet users** and the world's largest WhatsApp user base. Misinformation spreads rapidly through regional-language content on encrypted platforms where traditional fact-checkers have limited reach. This tool:

- Works with **any language** content accessible via URL (Jina Reader handles multilingual scraping)
- Supports India-specific news domains in its credibility registry (PTI, The Hindu, NDTV, OpIndia with appropriate tier ratings)
- Can be deployed as a **WhatsApp bot** or **Telegram bot** with minimal API integration, extending reach to non-English, mobile-first audiences

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                                │
│                                                                         │
│   ┌──────────────┐    ┌─────────────────┐    ┌───────────────────┐     │
│   │ React Web App│    │ Chrome Extension │    │ Direct API Call   │     │
│   │ (SSE Stream) │    │ (Simple Verdict) │    │ (POST/GET)       │     │
│   └──────┬───────┘    └────────┬─────────┘    └────────┬─────────┘     │
│          │                     │                        │               │
└──────────┼─────────────────────┼────────────────────────┼───────────────┘
           │                     │                        │
           ▼                     ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (main.py)                           │
│                                                                         │
│   GET /api/analyze-stream  ──→ SSE streaming (node-by-node)            │
│   POST /api/analyze-simple ──→ Full JSON response                      │
│   GET /health              ──→ Health check                            │
│                                                                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph Pipeline                        │
│                                                                         │
│   ┌─────────────┐    ┌──────────────┐    ┌────────────────┐            │
│   │   Agent 1    │    │   Agent 2     │    │    Agent 3     │            │
│   │   Claim      │───▶│   Evidence    │───▶│    Source      │            │
│   │   Extraction │    │   Retrieval   │    │    Credibility │            │
│   │   (Groq 8b)  │    │   (Tavily)    │    │    (No API)    │            │
│   └─────────────┘    └──────────────┘    └───────┬────────┘            │
│                                                    │                    │
│                                                    ▼                    │
│                      ┌──────────────┐    ┌────────────────┐            │
│                      │   Agent 5     │◀───│    Agent 4     │            │
│                      │   Explanation │    │    Fact        │            │
│                      │   Generator   │    │    Checker     │            │
│                      │   (Groq 8b)   │    │  (Groq 8b+70b)│            │
│                      └──────────────┘    └────────────────┘            │
│                                                                         │
│   State: AgentState (TypedDict)                                        │
│   ├── user_input, raw_markdown, claims, evidence_map                   │
│   ├── credibility_map, verdicts, explanations                          │
│   └── pub_date, author, source_domain, error                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The 5-Agent Pipeline — Deep Dive

### Agent 1: Claim Extraction

**File:** `backend/agents/claim_extraction.py`
**Model:** Groq `llama-3.1-8b-instant` (structured output via `json_mode`)
**Purpose:** Transform raw article content into a ranked list of checkable factual claims.

**Process:**
1. **Input routing** — Detects whether the input is a URL or raw text
2. **Scraping** — For URLs, calls Jina Reader API (`https://r.jina.ai/{url}`) to get clean markdown
3. **Sanitization** — Strips navigation chrome, headers, and boilerplate using regex heuristics
4. **Metadata extraction** — Pulls publication date and author from the article text
5. **Truncation** — Limits article to 10,000 characters to fit within LLM context windows
6. **LLM extraction** — Sends the truncated markdown to Groq with a carefully engineered prompt that enforces:
   - Co-reference resolution (replaces pronouns with explicit entities)
   - Checkworthiness scoring (1-10 scale)
   - Negative constraints (rejects standalone entities, requires subject + verb + assertion)
7. **Ranking** — Sorts claims by checkworthiness score, selects top-N (default: 3)

**Output schema (Pydantic):**
```python
class Claim(BaseModel):
    claim_text: str      # Self-contained factual assertion
    checkworthiness_score: int  # 1-10 priority ranking
    reasoning: str       # Why this claim is worth checking
```

---

### Agent 2: Evidence Retrieval

**File:** `backend/agents/evidence_retrieval.py`
**API:** Tavily Search API (web search optimized for AI agents)
**Purpose:** Gather supporting and contradicting evidence from the live web for each claim.

**Process:**
1. Receives ranked claims from Agent 1
2. Selects top-N claims (configurable, default: 3)
3. Runs **parallel web searches** using `ThreadPoolExecutor` (2 concurrent workers)
4. For each claim, queries Tavily with the full claim text
5. Normalizes results into `{url, content}` dicts, truncating snippets to 500 chars
6. Returns an `evidence_map: dict[claim_text → list[snippets]]`

**Key design decisions:**
- Parallel execution reduces wall-clock time by ~50% for multi-claim queries
- Snippet truncation prevents context window overflow in downstream agents
- Deterministic ordering preserved despite concurrent execution

---

### Agent 3: Source Credibility

**File:** `backend/agents/source_credibility.py`
**API calls:** None (zero external calls — pure Python)
**Purpose:** Score and re-rank evidence snippets by the trustworthiness of their source domains.

**Process:**
1. Extracts the domain from each evidence snippet URL
2. Looks up the domain against a **60-entry credibility registry** covering major global and Indian news sources
3. For unknown domains, applies **TLD-based heuristics**:
   - `.gov`, `.edu` → Tier 2 (score 82)
   - `.org` → Tier 3 (score 70)
   - `.com`, `.net` → Tier 4 (score 35)
   - Social media patterns → Tier 5 (score 10)
4. Enriches each snippet with a `credibility` object: `{score, label, tier, bias, domain}`
5. **Re-ranks snippets** per claim so higher-credibility sources appear first
6. Writes a `credibility_map` to state for downstream agents

**Tier System:**

| Tier | Score | Examples | Description |
|------|-------|----------|-------------|
| 1 | 90-100 | Reuters, AP, PTI, AFP, Snopes, PolitiFact | Wire services & primary fact-check orgs |
| 2 | 80-89 | BBC, NYT, The Hindu, Washington Post | Major national newspapers & broadcasters |
| 3 | 60-79 | WHO, Statista, Nature, Al Jazeera | Specialist, regional, government-adjacent |
| 4 | 30-59 | OpIndia, RT, Daily Mail, Breitbart | Low editorial standards / partisan bias |
| 5 | 0-29 | Twitter/X, Facebook, Reddit, TikTok | Social media — no editorial control |

---

### Agent 4: Fact Checker

**File:** `backend/agents/fact_checker.py`
**Model:** Groq `llama-3.1-8b-instant` (reasoning) + `llama-3.3-70b-versatile` (verdict)
**Purpose:** Analyze evidence against claims and produce structured verdicts.

**Architecture — 2-Call Chain-of-Thought:**

This agent uses a novel **dual-LLM architecture** that separates reasoning from judgment:

**Call 1 — Free-Form Reasoning (8b model, unconstrained)**
```
Input:  Claim + Evidence snippets (with credibility scores)
Output: Natural language reasoning document (unstructured)
Steps:  Evidence Inventory → Claim Decomposition → Evidence Matching →
        Contradiction Analysis → Confidence Assessment → Preliminary Verdict
```

**Call 2 — Structured Verdict (70b model, json_mode)**
```
Input:  Original claim + Evidence + Reasoning document from Call 1
Output: Strict Pydantic-validated JSON verdict
```

**Why two calls?**
- Single-call structured output forces the LLM to reason *and* format simultaneously, degrading both
- By separating reasoning from formatting, the model can "think freely" in Call 1, then a more capable model converts that reasoning into a precise schema in Call 2
- This produces measurably better verdicts with more nuanced confidence assessments

**Output schema:**
```python
class FactCheckVerdict(BaseModel):
    claim_text: str           # The claim being verified
    verdict: VerdictLabel     # SUPPORTED | CONTRADICTED | MISLEADING | UNVERIFIED
    truth_score: int          # 0-100 confidence score
    explanation: str          # 2-3 sentence justification
    citations: list[str]      # Source URLs (max 3, deduplicated)
    reasoning_summary: str    # Key points from CoT reasoning
    confidence_level: str     # HIGH / MEDIUM / LOW
    evidence_gaps: list[str]  # What the evidence didn't address
```

---

### Agent 5: Explanation Generator

**File:** `backend/agents/explanation_generator.py`
**Model:** Groq `llama-3.1-8b-instant` (structured output)
**Purpose:** Generate plain-English explanations for all verdicts in a single batched call.

**Process:**
1. Receives all verdicts + credibility data from Agents 3 & 4
2. Formats everything into a single prompt with strict writing rules:
   - 10th-grade reading level (no jargon)
   - Source quality described using exact tier phrases ("confirmed by major wire services")
   - Reader advisories triggered by specific conditions (low credibility, contradictions, MISLEADING)
3. Makes **one LLM call** for all claims (not one per claim) to minimize quota usage
4. Returns structured explanations keyed by claim text

**Output schema:**
```python
class ExplanationOutput(BaseModel):
    overall_credibility: str   # CREDIBLE / MOSTLY CREDIBLE / MIXED / LOW CREDIBILITY / UNRELIABLE
    bottom_line: str           # One-sentence article summary
    verdicts_explained: list[VerdictExplanation]  # Per-claim explanations

class VerdictExplanation(BaseModel):
    claim_text: str
    plain_english: str            # 3-4 sentence explanation
    confidence_statement: str     # How confident the reader should be
    source_quality_note: str      # Quality of sources used
    reader_advisory: str | None   # Warning if sources are weak
    evidence_gaps_plain: str | None  # What's still unknown
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Orchestration** | LangGraph StateGraph | DAG-based agent pipeline with typed state |
| **LLM Inference** | Groq Cloud (Llama 3.1 8b + 3.3 70b) | Fast inference (~200 tok/s) for structured + free-form output |
| **Web Search** | Tavily Search API | AI-optimized web search with snippet extraction |
| **Web Scraping** | Jina Reader API | Clean markdown extraction from any URL |
| **Backend** | FastAPI + Uvicorn | Async Python web server with SSE streaming |
| **Frontend** | React 18 + TypeScript + Vite | Real-time streaming UI with hot module reload |
| **Extension** | Chrome Manifest V3 | Browser-native popup with background service worker |
| **Validation** | Pydantic v2 | Strict schema validation for all LLM outputs |
| **Config** | pydantic-settings + PyYAML | Centralized config with env + yaml |

---

## Project Structure

```
FakeNews/
├── backend/
│   ├── agents/
│   │   ├── __init__.py                 # Package docs (5-agent registry)
│   │   ├── claim_extraction.py         # Agent 1 + LangGraph builder
│   │   ├── evidence_retrieval.py       # Agent 2 — Tavily search
│   │   ├── source_credibility.py       # Agent 3 — Domain tier scoring
│   │   ├── fact_checker.py             # Agent 4 — 2-call CoT verdicts
│   │   └── explanation_generator.py    # Agent 5 — Plain-English output
│   ├── core/
│   │   ├── config.py                   # Settings loader (env + yaml)
│   │   └── config.yaml                 # All tunable pipeline parameters
│   ├── models/
│   │   ├── requests.py                 # API request schemas
│   │   └── responses.py               # API response schemas
│   ├── services/
│   │   ├── jina_service.py             # Jina Reader wrapper
│   │   ├── tavily_service.py           # Tavily Search wrapper
│   │   ├── llm_service.py              # LLM client management
│   │   └── gemini_service.py           # DEPRECATED — kept as reference
│   ├── utils/
│   │   ├── helpers.py                  # Shared utilities (read_env_var)
│   │   ├── markdown.py                 # Markdown processing
│   │   └── scoring.py                  # Score computation helpers
│   ├── main.py                         # FastAPI app + endpoints
│   ├── requirements.txt                # Python dependencies
│   └── .env                            # API keys (not committed)
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                     # Main app with URL auto-start
│   │   ├── index.css                   # Full design system (dark theme)
│   │   ├── components/
│   │   │   ├── AgentChain.tsx          # Seamless streaming agent log
│   │   │   ├── ScoreDisplay.tsx        # Rich verdict + explanation UI
│   │   │   └── URLInput.tsx            # URL/text input component
│   │   └── hooks/
│   │       └── useAgentStream.ts       # SSE EventSource hook
│   ├── package.json
│   └── vite.config.ts
│
├── extension/
│   ├── manifest.json                   # Manifest V3 config
│   ├── popup.html                      # Extension popup UI
│   ├── popup.js                        # Pipeline progress + verdict display
│   ├── background.js                   # Service worker for API calls
│   └── content.js                      # Page URL extraction
│
├── README.md
└── .gitignore
```

---

## Getting Started

### Prerequisites

| Requirement | How to Get |
|------------|------------|
| **Python 3.11+** | `brew install python` or [python.org](https://python.org) |
| **Node.js 18+** | `brew install node` or [nodejs.org](https://nodejs.org) |
| **uv** (Python package manager) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Groq API Key** | Free at [console.groq.com](https://console.groq.com) |
| **Tavily API Key** | Free at [tavily.com](https://tavily.com) |
| **Jina Reader API Key** | Free at [jina.ai](https://jina.ai/reader) |

### 1. Clone & Configure

```bash
git clone https://github.com/PorjanyaBordoloi/fake-news-2.git
cd fake-news-2
```

Create `backend/.env`:
```env
GROQ_API_KEY=gsk_your_key_here
TAVILY_API_KEY=tvly-your_key_here
JINA_READER_API_KEY=jina_your_key_here
```

### 2. Start the Backend

```bash
cd backend
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### 3. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

### 4. Load the Chrome/Arc Extension

1. Open `chrome://extensions/` (Chrome) or `arc://extensions` (Arc)
2. Enable **Developer mode** (top right)
3. Click **Load unpacked** → select the `extension/` folder
4. Pin the extension to your toolbar
5. Navigate to any news article → click the extension icon → **Analyze This Page**

---

## API Reference

### `GET /api/analyze-stream`

**SSE streaming endpoint** for the web frontend. Pushes agent-by-agent updates as the pipeline executes.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_input` | string | Yes | URL or raw article text |

**Response:** `text/event-stream` with JSON payloads:
```json
data: {"agent": "claim_extraction", "status": "success", "data": {"claims": [...]}}
data: {"agent": "evidence_retrieval", "status": "success", "data": {"evidence_map": {...}}}
data: {"agent": "source_credibility", "status": "success", "data": {"credibility_map": {...}}}
data: {"agent": "fact_checker", "status": "success", "data": {"verdicts": [...]}}
data: {"agent": "explanation_generator", "status": "success", "data": {"explanations": {...}}}
data: {"status": "complete"}
```

### `POST /api/analyze-simple`

**Synchronous endpoint** for the Chrome Extension. Returns the full pipeline result.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_input` | string (query) | Yes | URL or raw article text |

**Response:** Full `AgentState` JSON including `verdicts`, `explanations`, `claims`, `evidence_map`, `credibility_map`.

### `GET /health`

Returns `{"status": "healthy"}`.

---

## Frontend Architecture

### Real-Time Streaming UI

The frontend uses **Server-Sent Events (SSE)** via the native `EventSource` API to receive pipeline updates in real time:

```
Backend (LangGraph stream) ──SSE──▶ useAgentStream hook ──state──▶ AgentChain + ScoreDisplay
```

**Key components:**

- **`useAgentStream.ts`** — Custom React hook that manages the EventSource lifecycle (open, collect, close). Tracks the connection in a `useRef` to prevent memory leaks on unmount.
- **`AgentChain.tsx`** — Seamless streaming text component. Lines appear one by one with staggered 120ms slide-in animations. Color-coded by type (blue=search, green=success, orange=verdict, gold=final). Shows animated "Thinking..." dots while waiting for the next agent.
- **`ScoreDisplay.tsx`** — Rich verdict display with overall credibility card, per-claim analysis, plain-English explanations, confidence badges, source quality notes, reader advisories, and evidence gap warnings.

### Design System

- **Background:** Deep gray `#212121` (Claude-inspired)
- **Chat bubbles:** `#2f2f2f` with subtle border
- **Typography:** System font stack (`-apple-system, BlinkMacSystemFont, Segoe UI`)
- **Color coding:** Green (#4ade80) = supported, Red (#fca5a5) = contradicted, Yellow (#fcd34d) = misleading, Slate (#94a3b8) = unverified

---

## Chrome Extension

### Architecture

```
User clicks icon → popup.html loads → popup.js runs
  ├── Shows current page URL
  ├── On "Analyze" click:
  │     ├── Shows pipeline progress animation (6 timed stages)
  │     ├── Calls POST /api/analyze-simple
  │     ├── Marks all stages complete
  │     └── Displays verdict card with score + bottom line
  └── "View Deep Analysis" → opens localhost:5173?url=<encoded_url>
                              → App.tsx auto-starts full streaming analysis
```

### Pipeline Progress Animation

Since the extension uses the non-streaming endpoint, it simulates pipeline progress with **timed stages** that match real-world execution times:

| Stage | Delay | What it represents |
|-------|-------|-------------------|
| Scraping article content | 0s | Jina Reader call |
| Extracting claims | 3s | Groq LLM structured output |
| Searching the web | 7s | Tavily parallel searches |
| Scoring source credibility | 13s | Domain tier lookup |
| Reasoning through evidence | 18s | 2-call fact-check per claim |
| Generating explanations | 28s | Batched explanation call |

---

## Configuration Reference

All tunable parameters live in `backend/core/config.yaml`:

```yaml
claim_extraction:
  model: "llama-3.1-8b-instant"
  temperature: 0
  markdown_truncation_limit: 10000

evidence_retrieval:
  max_claims_per_run: 3
  tavily_max_results: 5
  max_parallel_workers: 2

source_credibility:
  tier_scores: {1: 95, 2: 82, 3: 70, 4: 35, 5: 10}

fact_checker:
  reasoning_model: "llama-3.1-8b-instant"
  verdict_model: "llama-3.3-70b-versatile"
  max_citations: 3

explanation_generator:
  model: "llama-3.1-8b-instant"
  temperature: 0.2
```

---

## Error Handling & Resilience

The pipeline is designed to **never crash** — every agent has graceful fallback behavior:

| Agent | Failure Mode | Fallback |
|-------|-------------|----------|
| Claim Extraction | LLM returns invalid JSON | `RuntimeError` caught, returns empty claims + error message |
| Evidence Retrieval | Tavily API down or rate-limited | Returns empty evidence list per claim, pipeline continues |
| Source Credibility | Unknown domain | TLD heuristic scoring (`.gov` = Tier 2, `.com` = Tier 4, etc.) |
| Fact Checker | Groq rate limit (429) | Returns `UNVERIFIED` with truth_score 50, pipeline continues |
| Explanation Generator | LLM output malformed | `_fallback_explanations()` returns valid minimal explanations |

All `_read_env_var` calls are centralized in `utils/helpers.py` to prevent duplication and ensure consistent env variable handling across agents.

---

## Performance & Rate Limits

### Groq Free Tier Optimization

The pipeline is engineered to minimize `llama-3.3-70b-versatile` usage:

| Call | Model | Tokens/call | Calls/run |
|------|-------|-------------|-----------|
| Claim Extraction | 8b-instant | ~800 | 1 |
| Reasoning (per claim) | 8b-instant | ~1500 | 3 |
| Verdict (per claim) | **70b-versatile** | ~500 | **3** |
| Explanation | 8b-instant | ~800 | 1 |

**Total 70b tokens per run: ~1,500** (3 verdict calls × ~500 tokens)
**Free tier limit: 100,000 tokens/day** → **~66 full analyses per day**

### Typical Execution Time

| Stage | Duration |
|-------|----------|
| Jina scraping | 3-5s |
| Claim extraction | 2-4s |
| Evidence retrieval (parallel) | 5-8s |
| Source credibility scoring | <100ms |
| Fact-checking (3 claims × 2 calls) | 10-15s |
| Explanation generation | 3-5s |
| **Total pipeline** | **25-40s** |

---

<p align="center">
  <strong>Built for the Antigravity Agentic IDE Hackathon</strong><br/>
  <em>Fighting misinformation, one claim at a time.</em>
</p>
