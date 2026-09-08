# TicketSolver

Enterprise-grade, multi-agent AI customer support automation platform built on a stateful **LangGraph** orchestration loop, deterministic **ML cost routing & triage**, hard-coded **policy guardrails**, multi-tier memory (**PostgreSQL + Qdrant**), and complete **trace auditability** with real-time WebSocket feeds and 3D trace visualization.

Engineered to run at **$0 on free tiers** (NVIDIA NIM free API, local Ollama embeddings, self-hosted Qdrant & PostgreSQL) with zero PII leaks (real-time regex + Luhn scrubbing).

---

## Key Architectural Highlights

- **Multi-Agent Orchestration (LangGraph):** Dynamic state graph executing `Investigate -> Decide -> Retry / Resolve | Escalate`. Adapts at runtime rather than following static pipelines (FR-4).
- **Hard-Coded Policy Guardrails (Code-Enforced):** Risky actions (such as refunds $\ge$ $50.00) are intercepted at the graph level and routed to a Human-in-the-Loop (HITL) approval queue. Guardrails are implemented in deterministic Python code, never delegated to LLM prompts (FR-3, FR-8).
- **Three-Tier Memory Architecture:**
  - **Short-Term Context:** In-memory conversational state and retry accumulation per ticket session (FR-5).
  - **Long-Term Memory:** Customer profile facts (tier, account age, notes) stored in PostgreSQL and injected into the system prompt (FR-6).
  - **Episodic Memory:** Past resolved incidents vectorized via Ollama (`nomic-embed-text`, 768-dim) into a dedicated Qdrant collection to retrieve similar historical resolutions and close the feedback loop upon terminal resolution (FR-7, FR-9).
- **Full Trace Auditability & Observability:** Every reasoning step, tool call, memory retrieval, model identifier, and latency metric is persisted sequentially to relational trace tables (FR-10).
- **NFR Security & PII Redaction:** Automatic filter scrubbing emails, phone numbers, and Luhn-valid credit card numbers from application logs and tracebacks (NFR: Security).

---

## Production Machine Learning Models & Real Metrics

TicketSolver employs deterministic ML models to triage incoming tickets before LLM invocation, saving token costs and reducing end-to-end latency.

### 1. Multi-Output XGBoost Ticket Classifier (FR-1) — `ticket_classifier.joblib`

Classifies ticket text into category and severity simultaneously via TF-IDF n-gram vectorization and dual XGBoost classification heads.

- **Status:** Trained, evaluated, committed, and serving live ahead of the orchestrator.
- **Dataset:** Hugging Face `Tobi-Bueck/customer-support-tickets` (CC BY-NC 4.0). Cleaned, English-filtered (`language == "en"`), null-dropped, yielding **13,731 usable rows**.
- **Data Splits (Stratified 70/15/15 on joint `category|severity`):**
  - **Training Set:** 9,611 rows
  - **Validation Set:** 2,060 rows
  - **Held-Out Test Set:** 2,060 rows (evaluated strictly once via the production loader)
- **Serving Latency:** ~2.6 ms / ticket (2,060 tickets evaluated in 5.29 s).

#### Real Held-Out Test Set Performance (n = 2,060)

| Task | Head | Metric | Test Score | Quality Gate Threshold | Status |
|---|---|---|---|---|---|
| **Category** | XGBoost (4 classes) | **Macro-F1** | **0.8501** | $\ge$ 0.60 | **PASS** |
| | | **Accuracy** | **85.53%** | — | — |
| **Severity** | XGBoost (3 classes) | **Macro-F1** | **0.5899** | $\ge$ 0.40 | **PASS** |
| | | **Accuracy** | **61.55%** | — | — |

#### Category Breakdown (Test Split)

| Class | Precision | Recall | F1-Score | Support (Tickets) |
|---|---|---|---|---|
| `change` | 0.9847 | 0.9234 | **0.9531** | 209 |
| `incident` | 0.7810 | 0.9193 | **0.8445** | 830 |
| `problem` | 0.7653 | 0.5220 | **0.6207** | 431 |
| `request` | 0.9798 | 0.9847 | **0.9822** | 590 |
| **Macro Average** | **0.8777** | **0.8373** | **0.8501** | 2,060 |

#### Severity Breakdown (Test Split)

| Class | Precision | Recall | F1-Score | Support (Tickets) |
|---|---|---|---|---|
| `low` | 0.7080 | 0.3756 | **0.4908** | 426 |
| `medium` | 0.5782 | 0.6969 | **0.6320** | 838 |
| `high` | 0.6359 | 0.6583 | **0.6469** | 796 |
| **Macro Average** | **0.6407** | **0.5769** | **0.5899** | 2,060 |

*(Note on Data Quality: Kaggle dataset `suraj520` was thoroughly tested and rejected on empirical evidence after proving to have synthetic random-noise labels independent of text. See [`Docs/CLASSIFIER_METRICS.md`](Docs/CLASSIFIER_METRICS.md) for full benchmark documentation).*

### 2. Complexity & Cost Router (FR-14) — `app/ml/router.py`
- Selects between cheap tier (`meta/llama-3.1-8b-instruct`) and strong tier (`meta/llama-3.1-70b-instruct`) models based on token length, syntax complexity, and predicted severity.
- Paired with a Logistic Regression baseline (SRS §6.3).

### 3. Decision Confidence Calibration (FR-15) — `app/ml/calibration.py`
- Learns mapping from observable execution signals (attempt count, proposal length, tool call volume, tool errors) to calibrated probabilities $P(\text{correct})$ to dynamically tune escalation thresholds.

---

## Repository Structure

```
TicketSolver/
├── backend/
│   ├── app/
│   │   ├── agents/          # LangGraph reasoning agent (Investigator/Resolver)
│   │   ├── api/             # REST routes (tickets, approvals, traces, dashboard) & WebSocket
│   │   ├── core/            # Resilience (CircuitBreaker, retries), Tracing, PII Redaction
│   │   ├── db/              # SQLAlchemy models (Postgres) & session management
│   │   ├── guardrails/      # Deterministic code-level policy engines (e.g. $50 refund limit)
│   │   ├── llm/             # NVIDIA NIM OpenAI-compatible client
│   │   ├── memory/          # Customer memory (Postgres) & Episodic memory (Qdrant)
│   │   ├── ml/              # Classifier, Router, Calibration, and Feature pipelines
│   │   ├── orchestration/   # Stateful LangGraph workflow and async background workers
│   │   ├── tools/           # Simulated account and order lookup tools
│   │   └── vector/          # Qdrant client & Ollama nomic-embed-text RAG client
│   ├── data/                # Data pipeline splits (train/val/test - gitignored)
│   ├── kb/                  # Markdown knowledge-base source documents
│   ├── models/              # Serialized joblib models and metrics.json
│   ├── scripts/             # Ingestion, initialization, data prep, and training scripts
│   └── tests/               # 136 automated unit, integration, and ML tests
├── frontend/
│   ├── src/
│   │   ├── api/             # REST and WebSocket client
│   │   ├── components/      # Dashboard panels (Volume, Cost, Quality, Approvals)
│   │   ├── hooks/           # WebSocket real-time subscription hooks
│   │   ├── pages/           # DashboardPage and TicketDetailPage
│   │   └── types/           # Shared TypeScript interfaces
├── Docs/                    # SRS, Data Provenance, Classifier Metrics, and Module Roadmaps
├── CLAUDE.md                # Locked architectural decisions and stack constraints
└── rules.md                 # Agent and development operating rules
```

---

## Test Suite & Verification

The backend includes a comprehensive automated test suite spanning unit, resilience, data pipeline, and integration tests:

```bash
backend/.venv/Scripts/python.exe -m pytest
```

**Test Status:** `136 passed in 15.35s` (100% passing across 14 test modules):
- `test_dashboard.py`: Summary aggregation, cost accounting, and queue metrics.
- `test_redaction.py`: PII regex rules (emails, phone numbers, Luhn credit cards) and log filtering.
- `test_ml_classifier.py` & `test_ml_features.py`: TF-IDF extraction and XGBoost prediction stability.
- `test_ml_integration.py`: End-to-end classification through FastAPI endpoints.
- `test_guardrails.py`: Boundary verification ($49.99 allowed, $50.00 blocked) and HITL queue creation.
- `test_orchestrator_unit.py`: Transition table, retry bounds, and resolution states.
- `test_resilience.py`: Exponential backoff, timeout handling, and CircuitBreaker trips.
- `test_episodic.py` & `test_long_term.py`: Vector retrieval and relational customer memory.
- `test_tools.py` & `test_tracing.py`: Account/order mock stores and relational trace logging.

---

## Implementation Roadmap & Status

| Module | Feature Area | Status | Key Deliverables |
|---|---|---|---|
| **Module 0** | Scaffold & Dev Infra | ✅ Complete | FastAPI, Docker Compose (Postgres, Qdrant), Vite + React |
| **Module 1** | Investigator Agent & KB RAG | ✅ Complete | LangGraph loop, Ollama embeddings, Qdrant KB search, Postgres tracing |
| **Module 2** | Long-Term Customer Memory | ✅ Complete | Relational customer facts injected into reasoning context |
| **Module 3** | Episodic Incident Memory | ✅ Complete | Semantic retrieval of past resolutions + closed-loop write-back |
| **Module 4** | Orchestrator & Resilience | ✅ Complete | Multi-attempt graph, CircuitBreaker, retries, async jobs & WebSocket |
| **Module 5** | Guardrails & HITL Queue | ✅ Complete | Hardcoded $50 refund limit, approval queue, human review actions |
| **Module 6** | ML Ticket Classifier | ✅ Complete | Dual-head XGBoost classifier (85.5% category, 61.6% severity accuracy) |
| **Module 7** | Cost Router & Calibration | 🟡 In Progress | Router & calibration modules implemented; wiring to graph in progress |
| **Module 8** | Evaluation Harness | 📋 Planned | 30–50 hand-verified golden test tickets & regression reporting |
| **Module 9** | Dashboard Frontend | 🟡 In Progress | Summary API complete; Recharts UI panels & live WS wiring next |
| **Module 10**| 3D Trace Visualizer | 📋 Planned | Three.js / react-three-fiber interactive agent-flow graph |
| **Module 11**| Cost Analytics & Explainability | 📋 Planned | Token cost savings math & XGBoost feature importance display |
| **Module 12**| Clustering & Anomaly Detection| 📋 Optional | HDBSCAN taxonomy discovery & volume spike detection |
| **Module 13**| Deployment & Hardening | 🟡 In Progress | PII log redaction complete; cloud hosting configuration next |

---

## Getting Started

### Prerequisites
- Python $\ge$ 3.11 (3.13 recommended)
- Node.js $\ge$ 18 & npm
- Docker Desktop (for local PostgreSQL & Qdrant)
- Ollama running locally with `nomic-embed-text` pulled (`ollama pull nomic-embed-text`)

### 1. Start Local Infrastructure
```bash
cd backend
docker compose up -d
```
*Spins up PostgreSQL on port `5433` and Qdrant on port `6333`.*

### 2. Backend Setup
```bash
cd backend
python -m venv .venv
# On Windows Git Bash / PowerShell:
.\.venv\Scripts\activate
pip install -e ".[dev]"

# Copy environment variables
cp .env.example .env

# Initialize database schema & seed data
python scripts/init_db.py
python scripts/seed_customer_memory.py --verify
python scripts/seed_episodes.py --verify
python scripts/ingest_kb.py

# Run development server
uvicorn app.main:app --reload --port 8000
```

### 3. Reproducing the ML Classifier
```bash
cd backend
# Download dataset & create stratified splits
python scripts/prepare_data.py --download

# Train XGBoost classifier and evaluate on held-out test split
python scripts/train_classifier.py
```

### 4. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Visit `http://localhost:5173` to access the dashboard.
