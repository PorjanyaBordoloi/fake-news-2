# Fake News Detector - Multi-Agent System

> **AI-Powered Misinformation Detection using Agentic Workflows**
> 
> Built with FastAPI, React, and a 3-agent architecture for transparent, real-time fake news verification.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent System](#agent-system)
  - [Agent 1: Claim Extraction](#agent-1-claim-extraction)
  - [Agent 2: Evidence Retrieval](#agent-2-evidence-retrieval)
  - [Agent 3: Fact Checker](#agent-3-fact-checker)
- [Chain of Thought Display](#chain-of-thought-display)
- [Frontend Implementation](#frontend-implementation)
- [Backend Implementation](#backend-implementation)
- [Chrome Extension](#chrome-extension)
- [Setup & Installation](#setup--installation)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)

---

## 🎯 Overview

The Fake News Detector is a modular multi-agent fake-news verification pipeline that takes an input article/text and produces final explainable verdicts through three sequential agents:

1. **Claim Extraction Agent** — "What should we fact-check?" — Detects URL/text, scrapes via Jina Reader, extracts ranked claims via Groq LLM
2. **Evidence Retrieval Agent** — "What evidence do we have?" — Gathers supporting/refuting web evidence via Tavily search
3. **Fact Checker Agent** — "Given this evidence, what is the verdict?" — Produces structured verdicts per claim via Gemini LLM

### Key Features

✅ **Transparent Agent Workflow** - Users see real-time agent thinking process  
✅ **Chain of Thought Display** - Step-by-step reasoning shown on website  
✅ **Chrome Extension** - Quick fake/real verdict while browsing  
✅ **Multi-Source Verification** - Tavily web search for evidence gathering  
✅ **Structured Verdicts** - Per-claim verdicts with truth scores, explanations, and citations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         USER INPUT                          │
│  Website: URL / text input   |  Extension: Current page URL │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  FastAPI      │
         │  Backend      │
         └───────┬───────┘
                 │
    ┌────────────┴────────────┐
    ▼                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    AGENT PIPELINE                            │
│  Sequential execution with SSE streaming                     │
└────────────────┬─────────┬──────────┬────────────────────────┘
                 │         │          │
        ┌────────┘         │          └────────┐
        ▼                  ▼                   ▼
   ┌─────────────┐   ┌──────────────┐   ┌─────────────┐
   │  AGENT 1    │   │   AGENT 2    │   │  AGENT 3    │
   │  Claim      │──>│  Evidence    │──>│  Fact       │
   │  Extraction │   │  Retrieval   │   │  Checker    │
   └─────────────┘   └──────────────┘   └─────────────┘
        │                  │                   │
        ▼                  ▼                   ▼
   Jina Reader        Tavily API          Gemini LLM
   + Groq LLM         (web search)        (structured
   (structured                             verdicts)
    claims)
                    │
                    ▼
            ┌───────────────┐
            │  Per-Claim    │
            │  Verdicts +   │
            │  Truth Scores │
            └───────┬───────┘
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
   ┌─────────┐            ┌──────────────┐
   │ Website │            │  Extension   │
   │ (Full   │            │  (Verdicts + │
   │ Chain)  │            │  Scores)     │
   └─────────┘            └──────────────┘
```

---

## 🤖 Agent System

### Overall 3-Agent Design

The pipeline is built with **LangGraph** as a compiled `StateGraph` with three sequential nodes.
All three nodes share a single `AgentState` TypedDict and the graph is compiled once at module load:

| Agent | Question Answered | File | LangGraph Node |
|-------|-------------------|------|----------------|
| **Claim Extraction** | "What should we fact-check?" | `claim_extraction.py` | `extraction_node` |
| **Evidence Retrieval** | "What evidence do we have?" | `evidence_retrieval.py` | `evidence_retrieval_node` |
| **Fact Checker** | "Given this evidence, what is the verdict?" | `fact_checker.py` | `fact_checker_node` |

```
START → claim_extraction → evidence_retrieval → fact_checker → END
```

**Shared State (`AgentState`):**
```python
class AgentState(TypedDict):
    user_input: str
    input_url: NotRequired[str]
    raw_markdown: str
    claims: list[dict[str, Any]]
    evidence_map: dict[str, list[dict[str, str]]]
    verdicts: list[dict[str, Any]]
    top_n: Optional[int]
    pub_date: Optional[str]
    author: Optional[str]
    source_domain: Optional[str]
    error: str
```

---

### Agent 1: Claim Extraction

**Purpose:** Turn raw user input (URL or text) into a small list of high-value, checkable claims.

#### What it does:
1. Detects whether input is a URL or plain text
2. If URL, scrapes article markdown via **Jina Reader**
3. Cleans noisy page text and extracts basic metadata (`pub_date`, `author`, `source_domain`)
4. Sends text to a **Groq LLM** with structured output schema
5. Produces ranked claims with:
   - `claim_text`
   - `checkworthiness_score` (1-10)
   - `reasoning`
6. Keeps top-N claims and stores them in state

#### Technologies:
- **Jina Reader** (`r.jina.ai`) — URL-to-markdown scraping
- **LangChain ChatGroq** (`llama-3.1-8b-instant`) — Structured claim extraction with `json_mode`
- **LangGraph** — `StateGraph` with 3-node sequential pipeline
- **Pydantic** — `ClaimExtractionResult` schema for structured output validation
- **requests** — Synchronous HTTP client for Jina

#### Output (to Agent 2):
```json
{
  "success": true,
  "agent": "claim_extraction",
  "thought": "Detected URL input. Scraped article via Jina Reader. Extracted 5 claims, keeping top 3 by checkworthiness.",
  "data": {
    "input_type": "url",
    "metadata": {
      "pub_date": "2026-03-06",
      "author": "John Doe",
      "source_domain": "example.com"
    },
    "claims": [
      {
        "claim_text": "Biden announces new $2T infrastructure plan",
        "checkworthiness_score": 9,
        "reasoning": "Major policy claim with specific dollar amount, easily verifiable"
      },
      {
        "claim_text": "Study shows 80% increase in renewable energy",
        "checkworthiness_score": 8,
        "reasoning": "Specific statistical claim that can be cross-referenced"
      },
      {
        "claim_text": "Event occurred on March 5, 2026",
        "checkworthiness_score": 6,
        "reasoning": "Date claim, verifiable but lower priority"
      }
    ]
  },
  "timestamp": "2026-03-06T10:30:00Z"
}
```

#### Agent 1 Thought Process (displayed to user):
```
🔍 Agent 1: Claim Extraction
├─ Detecting input type...
│  └─ Input is a URL: example.com/news-article
├─ Scraping via Jina Reader...
│  └─ ✅ Received 2,450 words of markdown
├─ Extracting metadata...
│  ├─ Author: John Doe
│  ├─ Published: March 6, 2026
│  └─ Domain: example.com
├─ Sending to Groq LLM for claim extraction...
│  ├─ Extracted 5 candidate claims
│  ├─ Ranked by checkworthiness score
│  └─ Keeping top 3 claims
├─ Top Claims:
│  ├─ [9/10] "Biden announces new $2T infrastructure plan"
│  ├─ [8/10] "Study shows 80% increase in renewable energy"
│  └─ [6/10] "Event occurred on March 5, 2026"
└─ ✅ Claim extraction complete. Passing to Evidence Retrieval...
```

---

### Agent 2: Evidence Retrieval

**Purpose:** Gather supporting/refuting web evidence for the most important claims.

#### What it does:
1. Takes extracted claims and sorts by `checkworthiness_score`
2. Limits processing to top claims (`MAX_CLAIMS_PER_RUN`)
3. Uses **Tavily search** to fetch relevant web snippets per claim
4. Normalizes each result to `{url, content}` and truncates snippet length
5. Runs retrieval concurrently with a thread pool for speed
6. Outputs an `evidence_map` keyed by claim text

#### Technologies:
- **Tavily API** — Real-time web search with source citations
- **Concurrent thread pool** — Parallel evidence retrieval for speed

#### Input: Output from Agent 1 (claims list)

#### Output (to Agent 3):
```json
{
  "success": true,
  "agent": "evidence_retrieval",
  "thought": "Retrieved evidence for 3 claims using Tavily. Found 12 total snippets across all claims.",
  "data": {
    "claims_processed": 3,
    "evidence_map": {
      "Biden announces new $2T infrastructure plan": [
        {
          "url": "https://reuters.com/article/biden-infrastructure-2026",
          "content": "President Biden unveiled a $2 trillion infrastructure package on March 5..."
        },
        {
          "url": "https://apnews.com/article/white-house-infrastructure",
          "content": "The White House confirmed details of the new infrastructure spending plan..."
        }
      ],
      "Study shows 80% increase in renewable energy": [
        {
          "url": "https://nature.com/articles/renewable-growth-2026",
          "content": "Nature Journal reports actual renewable energy growth at approximately 40%, not 80%..."
        }
      ],
      "Event occurred on March 5, 2026": [
        {
          "url": "https://bbc.com/news/world-us-2026-03-05",
          "content": "The announcement was made on March 5, 2026 at the White House..."
        }
      ]
    }
  },
  "timestamp": "2026-03-06T10:30:10Z"
}
```

#### Agent 2 Thought Process (displayed to user):
```
🔎 Agent 2: Evidence Retrieval
├─ Received 3 claims sorted by checkworthiness
│  ├─ [9/10] "Biden announces new $2T infrastructure plan"
│  ├─ [8/10] "Study shows 80% increase in renewable energy"
│  └─ [6/10] "Event occurred on March 5, 2026"
│
├─ Searching Tavily for evidence (concurrent)...
│  ├─ Claim 1: "Biden infrastructure plan"
│  │  ├─ Found: Reuters (relevant) ✅
│  │  ├─ Found: AP News (relevant) ✅
│  │  └─ 2 snippets collected
│  │
│  ├─ Claim 2: "80% renewable energy increase"
│  │  ├─ Found: Nature Journal (contradicts) ❌
│  │  └─ 1 snippet collected
│  │
│  └─ Claim 3: "Event on March 5, 2026"
│     ├─ Found: BBC News (confirms) ✅
│     └─ 1 snippet collected
│
├─ Normalizing results to {url, content} format...
│  └─ 12 total evidence snippets across 3 claims
│
└─ ✅ Evidence retrieval complete. Passing to Fact Checker...
```

---

### Agent 3: Fact Checker

**Purpose:** Convert claim + evidence into final user-facing verdicts.

#### What it does:
1. For each claim, builds evidence context from valid HTTP/HTTPS snippets
2. Uses **Gemini** with structured schema `FactCheckVerdict`
3. Enforces one of 4 labels:
   - `SUPPORTED`
   - `CONTRADICTED`
   - `MISLEADING`
   - `UNVERIFIED`
4. Returns per claim:
   - `verdict`
   - `truth_score` (0-100)
   - 2-sentence `explanation`
   - Cleaned `citations` (only from retrieved evidence URLs, deduplicated/canonicalized)
5. Falls back to safe `UNVERIFIED` output if no evidence or LLM error

#### Technologies:
- **Google Gemini LLM** — Structured schema output for verdicts

#### Input: Output from Agent 1 (claims) + Agent 2 (evidence_map)

#### Output (Final Response):
```json
{
  "success": true,
  "agent": "fact_checker",
  "thought": "Generated verdicts for 3 claims. 1 SUPPORTED, 1 CONTRADICTED, 1 SUPPORTED.",
  "data": {
    "verdicts": [
      {
        "claim_text": "Biden announces new $2T infrastructure plan",
        "verdict": "SUPPORTED",
        "truth_score": 92,
        "explanation": "Multiple authoritative sources confirm this announcement. Reuters and AP News both reported the $2T infrastructure package on March 5.",
        "citations": [
          "https://reuters.com/article/biden-infrastructure-2026",
          "https://apnews.com/article/white-house-infrastructure"
        ]
      },
      {
        "claim_text": "Study shows 80% increase in renewable energy",
        "verdict": "CONTRADICTED",
        "truth_score": 18,
        "explanation": "Nature Journal reports the actual renewable energy growth at approximately 40%, not 80%. The article's statistic appears to be a significant exaggeration.",
        "citations": [
          "https://nature.com/articles/renewable-growth-2026"
        ]
      },
      {
        "claim_text": "Event occurred on March 5, 2026",
        "verdict": "SUPPORTED",
        "truth_score": 95,
        "explanation": "BBC News confirms the announcement was made on March 5, 2026. The date is consistent across all retrieved sources.",
        "citations": [
          "https://bbc.com/news/world-us-2026-03-05"
        ]
      }
    ]
  },
  "timestamp": "2026-03-06T10:30:25Z"
}
```

#### Agent 3 Thought Process (displayed to user):
```
✅ Agent 3: Fact Checker
├─ Received 3 claims + evidence map
│
├─ Processing Claim 1: "Biden $2T infrastructure plan"
│  ├─ Building evidence context from 2 HTTP snippets...
│  ├─ Sending to Gemini for structured verdict...
│  ├─ Verdict: SUPPORTED (truth_score: 92)
│  └─ Citations: reuters.com, apnews.com
│
├─ Processing Claim 2: "80% renewable energy increase"
│  ├─ Building evidence context from 1 HTTP snippet...
│  ├─ Sending to Gemini for structured verdict...
│  ├─ Verdict: CONTRADICTED (truth_score: 18)
│  └─ Citations: nature.com
│
├─ Processing Claim 3: "Event on March 5, 2026"
│  ├─ Building evidence context from 1 HTTP snippet...
│  ├─ Sending to Gemini for structured verdict...
│  ├─ Verdict: SUPPORTED (truth_score: 95)
│  └─ Citations: bbc.com
│
├─ Final Summary:
│  ├─ 2 claims SUPPORTED
│  ├─ 1 claim CONTRADICTED
│  └─ 0 claims UNVERIFIED
│
└─ ✅ Fact checking complete. Returning verdicts to user...
```

---

## 📺 Chain of Thought Display

### Website Implementation (Real-time Streaming)

The website displays the agent thinking process in real-time using **Server-Sent Events (SSE)**.

The compiled LangGraph uses `.stream()` to yield per-node outputs as they complete:

#### Backend: FastAPI SSE Endpoint (LangGraph)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

from agents.claim_extraction import claim_extraction_graph

app = FastAPI()

def _build_initial_state(user_input: str) -> dict:
    return {
        "user_input": user_input,
        "raw_markdown": "", "claims": [], "evidence_map": {},
        "verdicts": [], "top_n": 3, "pub_date": None,
        "author": None, "source_domain": None, "error": "",
    }

@app.post("/api/analyze-stream")
async def analyze_stream(user_input: str):
    async def event_generator():
        # LangGraph .stream() yields {node_name: node_output} per step
        for event in claim_extraction_graph.stream(_build_initial_state(user_input)):
            for node_name, node_output in event.items():
                payload = {
                    "agent": node_name,
                    "status": "error" if node_output.get("error") else "success",
                    "data": node_output,
                }
                yield f"data: {json.dumps(payload)}\n\n"

        yield f"data: {json.dumps({'status': 'complete'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

#### Frontend: React SSE Consumer

```jsx
import React, { useState, useEffect } from 'react';

function AgentThinkingDisplay({ url }) {
  const [thoughts, setThoughts] = useState([]);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    const eventSource = new EventSource(
      `http://localhost:8000/api/analyze-stream?url=${encodeURIComponent(url)}`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.status === 'complete') {
        setIsComplete(true);
        eventSource.close();
      } else {
        setThoughts(prev => [...prev, data]);
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
    };

    return () => eventSource.close();
  }, [url]);

  return (
    <div className="agent-thinking-container">
      <h2>Agent Analysis Progress</h2>
      
      {thoughts.map((thought, index) => (
        <div key={index} className={`thought-item agent-${thought.agent}`}>
          <div className="agent-badge">
            {thought.agent === 'claim_extraction' && '🔍 Agent 1: Claim Extraction'}
            {thought.agent === 'evidence_retrieval' && '🔎 Agent 2: Evidence Retrieval'}
            {thought.agent === 'fact_checker' && '✅ Agent 3: Fact Checker'}
          </div>
          
          <div className="thought-content">
            <span className={`status-${thought.status}`}>
              {thought.status === 'started' && '⏳'}
              {thought.status === 'progress' && '⚙️'}
              {thought.status === 'success' && '✅'}
            </span>
            {thought.thought}
          </div>
          
          {thought.data && (
            <div className="thought-data">
              {/* Render relevant data preview */}
            </div>
          )}
        </div>
      ))}
      
      {isComplete && (
        <div className="analysis-complete">
          ✅ Analysis Complete!
        </div>
      )}
    </div>
  );
}
```

#### CSS Styling (Claude-like Chain of Thought)

```css
.agent-thinking-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'Inter', sans-serif;
}

.thought-item {
  margin-bottom: 16px;
  padding: 16px;
  background: #f7f7f7;
  border-left: 4px solid #ccc;
  border-radius: 8px;
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.thought-item.agent-claim_extraction {
  border-left-color: #4a9eed;
  background: #e8f4fd;
}

.thought-item.agent-evidence_retrieval {
  border-left-color: #f59e0b;
  background: #fef3e2;
}

.thought-item.agent-fact_checker {
  border-left-color: #22c55e;
  background: #e8f8ed;
}

.agent-badge {
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 8px;
  color: #374151;
}

.thought-content {
  font-size: 15px;
  color: #1f2937;
  line-height: 1.5;
}

.status-started {
  display: inline-block;
  margin-right: 8px;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.status-progress {
  display: inline-block;
  margin-right: 8px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.analysis-complete {
  padding: 20px;
  text-align: center;
  font-size: 18px;
  font-weight: 600;
  color: #22c55e;
  background: #e8f8ed;
  border-radius: 8px;
  margin-top: 20px;
}
```

---

## 🌐 Frontend Implementation

### Website (Full Experience)

**Features:**
- URL input box
- Real-time agent thinking display (SSE)
- Final score and verdict
- Detailed breakdown of red flags
- Source citations with links
- Export report as PDF

**Tech Stack:**
- React 18+ with TypeScript
- Tailwind CSS
- EventSource API for SSE
- Framer Motion for animations

**Key Components:**
```
src/
├── components/
│   ├── URLInput.tsx           # URL submission form
│   ├── AgentChain.tsx          # Chain of thought display
│   ├── ScoreDisplay.tsx        # Final score visualization
│   ├── RedFlagsList.tsx        # Red flags breakdown
│   ├── SourceCitations.tsx     # Trusted sources list
│   └── ExportReport.tsx        # PDF export button
├── hooks/
│   ├── useAgentStream.ts       # SSE hook for agent updates
│   └── useAnalysis.ts          # Analysis state management
├── services/
│   └── api.ts                  # API client
└── App.tsx
```

---

## 🧩 Chrome Extension

### Extension Architecture

**Manifest V3 Structure:**
```json
{
  "manifest_version": 3,
  "name": "Fake News Detector",
  "version": "1.0.0",
  "permissions": ["activeTab", "storage"],
  "action": {
    "default_popup": "popup.html"
  },
  "background": {
    "service_worker": "background.js"
  },
  "content_scripts": [{
    "matches": ["<all_urls>"],
    "js": ["content.js"]
  }]
}
```

### Extension Flow

1. **User clicks extension icon** on any webpage
2. **Content script** extracts current page URL
3. **Background worker** sends URL to `/api/analyze-simple` endpoint
4. **Popup displays** simple verdict:

```html
<!-- popup.html -->
<div class="popup-container">
  <div class="header">
    <h1>Fake News Detector</h1>
  </div>
  
  <div id="loading" class="hidden">
    <div class="spinner"></div>
    <p>Analyzing article...</p>
  </div>
  
  <div id="result" class="hidden">
    <!-- FAKE verdict -->
    <div class="verdict verdict-fake">
      <div class="score-badge">Score: 28/100</div>
      <h2>⚠️ Likely Fake</h2>
      <p>Multiple red flags detected</p>
      <button id="view-details">View Full Analysis</button>
    </div>
    
    <!-- REAL verdict -->
    <div class="verdict verdict-real hidden">
      <div class="score-badge">Score: 87/100</div>
      <h2>✅ Likely Real</h2>
      <p>Verified by trusted sources</p>
      <button id="view-details">View Full Analysis</button>
    </div>
  </div>
</div>
```

**Extension displays ONLY:**
- Score (0-100)
- Verdict (Fake/Real/Misleading)
- Button to "View Full Analysis" (opens website)

**Extension does NOT show:**
- Chain of thought
- Detailed reasoning
- Source citations

This keeps the extension lightweight and fast.

---

## 🔧 Backend Implementation

### FastAPI Project Structure

```
backend/
├── main.py                    # FastAPI app entry point + pipeline orchestration
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
├── agents/
│   ├── __init__.py
│   ├── claim_extraction.py   # Agent 1: Claim Extraction (Jina + Groq)
│   ├── evidence_retrieval.py # Agent 2: Evidence Retrieval (Tavily)
│   └── fact_checker.py       # Agent 3: Fact Checker (Gemini)
├── services/
│   ├── jina_service.py       # Jina Reader API wrapper
│   ├── llm_service.py        # Groq LLM wrapper (claim extraction)
│   ├── tavily_service.py     # Tavily API wrapper (evidence search)
│   └── gemini_service.py     # Gemini LLM wrapper (verdicts)
├── models/
│   ├── requests.py           # Pydantic request models
│   └── responses.py          # Pydantic response models
└── utils/
    ├── scoring.py            # Score calculation logic
    └── markdown.py           # Markdown report generator
```

### API Endpoints

```python
# main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from agents.claim_extraction import claim_extraction_graph

app = FastAPI(title="Fake News Detector API")

def _build_initial_state(user_input: str, top_n: int = 3) -> dict:
    return {
        "user_input": user_input,
        "raw_markdown": "",
        "claims": [],
        "evidence_map": {},
        "verdicts": [],
        "top_n": top_n,
        "pub_date": None,
        "author": None,
        "source_domain": None,
        "error": "",
    }

@app.post("/api/analyze-stream")
async def analyze_stream(user_input: str):
    """SSE streaming — yields node outputs as events"""
    async def event_generator():
        for event in claim_extraction_graph.stream(_build_initial_state(user_input)):
            for node_name, node_output in event.items():
                yield f"data: {json.dumps({'agent': node_name, 'data': node_output})}\n\n"
        yield f"data: {json.dumps({'status': 'complete'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/analyze-simple")
async def analyze_simple(user_input: str):
    """Simple endpoint — returns full final state"""
    return claim_extraction_graph.invoke(_build_initial_state(user_input))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- API Keys:
  - Tavily API
  - Groq API
  - Google Gemini API
  - Jina Reader API

### Backend Setup

```bash
# Clone repository
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
TAVILY_API_KEY=your_tavily_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
JINA_READER_API_KEY=your_jina_key
EOF

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
# Navigate to frontend
cd ../frontend

# Install dependencies
npm install

# Create .env file
cat > .env << EOF
VITE_API_URL=http://localhost:8000
EOF

# Run development server
npm run dev
```

### Chrome Extension Setup

```bash
# Navigate to extension
cd ../extension

# Install dependencies
npm install

# Build extension
npm run build

# Load in Chrome:
# 1. Go to chrome://extensions
# 2. Enable "Developer mode"
# 3. Click "Load unpacked"
# 4. Select the `extension/dist` folder
```

---

## 📚 API Documentation

### Tavily API Integration

```python
# services/tavily_service.py
import httpx
import os

class TavilyService:
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com"
    
    async def search(self, query: str, max_results: int = 5):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "include_domains": [
                        "reuters.com",
                        "apnews.com",
                        "bbc.com",
                        "nytimes.com"
                    ],
                    "max_results": max_results
                }
            )
            return response.json()
```

### Groq LLM Integration (Claim Extraction)

```python
# services/llm_service.py
from groq import Groq
import os

class LLMService:
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
    
    async def extract_claims(self, article_text: str):
        prompt = f"""Extract factual claims from the following article text.

ARTICLE:
{article_text}

Return a JSON array of claims, each with:
- claim_text (the factual claim)
- checkworthiness_score (1-10, how important to verify)
- reasoning (why this score)
"""
        
        chat_completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return chat_completion.choices[0].message.content
```

### Gemini LLM Integration (Fact Checking)

```python
# services/gemini_service.py
from google import genai
import os

class GeminiService:
    def __init__(self):
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )
    
    async def generate_verdict(self, claim: str, evidence_context: str):
        prompt = f"""You are a fact-checking agent.

CLAIM: {claim}
EVIDENCE: {evidence_context}

Respond in JSON with:
- verdict: SUPPORTED | CONTRADICTED | MISLEADING | UNVERIFIED
- truth_score: 0-100
- explanation: exactly 2 sentences
- citations: array of evidence URLs
"""
        
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        
        return response.text
```

### Jina Reader Integration (URL Scraping)

```python
# services/jina_service.py
import httpx
import os

class JinaService:
    def __init__(self):
        self.api_key = os.getenv("JINA_READER_API_KEY")
        self.base_url = "https://r.jina.ai"
    
    async def scrape_url(self, url: str) -> str:
        headers = {"Accept": "text/markdown"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/{url}",
                headers=headers,
            )
            response.raise_for_status()
            return response.text
```

---

## 🎯 Environment Variables

```bash
# .env file
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxx
GEMINI_API_KEY=xxxxxxxxxxxxx
JINA_READER_API_KEY=jina_xxxxxxxxxxxxx

# Optional
ENVIRONMENT=development
PORT=8000
CORS_ORIGINS=http://localhost:5173,chrome-extension://*
```

---

## 📦 Deployment

### Backend (Railway/Render)

```bash
# railway.toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
```

### Frontend (Vercel)

```bash
# vercel.json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "env": {
    "VITE_API_URL": "https://your-api.railway.app"
  }
}
```

### Chrome Extension (Chrome Web Store)

1. Build production version: `npm run build`
2. Zip the `dist` folder
3. Upload to [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole)

---

## 🎨 UI/UX Design

### Chain of Thought Display (Website)

Visual inspiration: Claude's thinking process

```
┌─────────────────────────────────────────────┐
│  🔍 Agent 1: Claim Extraction               │
│  ├─ Detecting input type...                 │
│  ├─ Scraping via Jina Reader...             │
│  ├─ ✅ Extracted 3 checkable claims         │
│  └─ Passing to Agent 2...                   │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  🔎 Agent 2: Evidence Retrieval              │
│  ├─ Searching Tavily for evidence...        │
│  │  ├─ Claim 1: 2 snippets found ✅         │
│  │  ├─ Claim 2: 1 snippet found ❌          │
│  │  └─ Claim 3: 1 snippet found ✅          │
│  └─ ✅ Evidence map ready                   │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  ✅ Agent 3: Fact Checker                    │
│  ├─ Generating verdicts via Gemini...       │
│  ├─ Claim 1: SUPPORTED (92/100)             │
│  ├─ Claim 2: CONTRADICTED (18/100)          │
│  ├─ Claim 3: SUPPORTED (95/100)             │
│  └─ ✅ All verdicts complete                │
└─────────────────────────────────────────────┘
```

### Extension Popup (Simple)

```
┌──────────────────────┐
│ Fake News Detector   │
├──────────────────────┤
│                      │
│   Score: 28/100      │
│                      │
│   ⚠️ Likely Fake     │
│                      │
│ Multiple red flags   │
│ detected             │
│                      │
│ [View Full Analysis] │
│                      │
└──────────────────────┘
```

---

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest tests/

# Frontend tests
cd frontend
npm test

# E2E tests
npm run test:e2e
```

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

---

## 📧 Support

For issues or questions, please open a GitHub issue or contact support@fakenewsdetector.com

---

**Built with ❤️ for truth and transparency**