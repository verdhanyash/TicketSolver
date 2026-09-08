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

### 2. Complexity & Cost Router (FR-14) — `ticket_router.joblib`

Routes tickets to the cheap tier (`meta/llama-3.1-8b-instruct`) or strong tier (`meta/llama-3.1-70b-instruct`) based on 11 lexical, syntactic, and predicted category/severity features, paired with a Logistic Regression baseline (SRS §6.3).

- **Status:** Trained, evaluated, and serving live ahead of the orchestrator (`_route_ticket`).
- **Held-Out Test Set Performance ($n = 2,060$):**

| Model | Accuracy | Macro-F1 | Simple F1 | Complex F1 | Inference Latency | Quality Gate | Status |
|---|---|---|---|---|---|---|---|
| **XGBoost Classifier** | **78.79%** (0.7879) | **0.7857** | **0.7639** | **0.8074** | ~2.21 ms / ticket | Macro-F1 $\ge 0.70$ | **PASS** |
| **Logistic Regression Baseline** | **69.51%** (0.6951) | **0.6860** | **0.6323** | **0.7396** | ~1.45 ms / ticket | — | — |
| **Comparison Advantage** | **+9.28%** | **+9.97%** | **+13.16%** | **+6.78%** | — | XGBoost $\ge$ Baseline | **PASS** |

### 3. Decision Confidence Calibration (FR-15) — `confidence_calibration.joblib`

Maps runtime agent observables (`attempt`, `proposal_length`, `confidence`, `tool_calls`, `tool_errored`) to $P(\text{decision correct})$ to dynamically adjust orchestrator escalation thresholds between $0.30$ and $0.90$.

- **Status:** Trained, evaluated, and active in `decide_node` (`_calibrated_threshold`).
- **Held-Out Test Performance ($n = 450$):**
  - **Accuracy:** **1.0000** (Gate $\ge 0.80$ **PASS**)
  - **Brier Score:** **0.0125** (Gate $\le 0.15$ **PASS**)
  - **Log-Loss:** **0.0519**
  - **Inference Latency:** ~0.17 ms / run

### 4. System-Level Evaluation Harness (FR-12, FR-13, FR-17, §10) — `eval/report.json`

Evaluates end-to-end multi-agent orchestration across 35 curated golden support tickets (18 routine requests, 9 guardrail violations, and 8 unresolvable/high-risk edge cases).

- **Status:** Evaluator engine, golden dataset, CLI runner, and baseline report complete (`backend/eval/report.json`).
- **Real Benchmark Performance ($n = 35$):**

| Quality Gate / Benchmark | Metric | Target | Actual | Verdict |
|---|---|---|---|---|
| **Routine Resolution Rate (§10.3)** | Resolution Correctness | $\ge 80.0\%$ | **100.0%** (18/18) | **PASS** |
| **Guardrail Safety Intercept (§10.2)** | Intercept Correctness | $100.0\%$ | **100.0%** (9/9) | **PASS** |
| **Escalation F1 Score (FR-13)** | Precision / Recall F1 | $\ge 0.75$ | **1.0000** (8/8) | **PASS** |
| **Overall Terminal Accuracy** | Exact Outcome Match | — | **100.0%** (35/35) | **PASS** |
| **M6 Category Classifier Gate** | Category Macro-F1 | $\ge 0.60$ | **0.8501** | **PASS** |
| **M7 Router vs Baseline Margin (§6.3)** | XGBoost vs LR Advantage | Advantage $> 0$ | **+9.97%** | **PASS** |
| **M7 Calibration Brier Score** | Brier Reliability Score | $\le 0.15$ | **0.0125** | **PASS** |

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
│   │   ├── eval/            # Evaluation runner engine and metric calculators (FR-12, FR-13, FR-17)
│   │   ├── guardrails/      # Deterministic code-level policy engines (e.g. $50 refund limit)
│   │   ├── llm/             # NVIDIA NIM OpenAI-compatible client
│   │   ├── memory/          # Customer memory (Postgres) & Episodic memory (Qdrant)
│   │   ├── ml/              # Classifier, Router, Calibration, and Feature pipelines
│   │   ├── orchestration/   # Stateful LangGraph workflow and async background workers
│   │   ├── tools/           # Simulated account and order lookup tools
│   │   └── vector/          # Qdrant client & Ollama nomic-embed-text RAG client
│   ├── data/                # Data pipeline splits (train/val/test - gitignored)
│   ├── eval/                # Curated golden tickets dataset and baseline reports (JSON & Markdown)
│   ├── kb/                  # Markdown knowledge-base source documents
│   ├── models/              # Serialized joblib models and metrics JSON reports
│   ├── scripts/             # Ingestion, initialization, data prep, training, and eval scripts
│   └── tests/               # 157 automated unit, integration, ML, and evaluation tests
├── frontend/
│   ├── src/
│   │   ├── api/             # REST and WebSocket client
│   │   ├── components/      # Dashboard panels (Volume, Cost, Quality, Approvals)
│   │   ├── hooks/           # WebSocket real-time subscription hooks
│   │   ├── pages/           # DashboardPage and TicketDetailPage
│   │   └── types/           # Shared TypeScript interfaces
├── Docs/                    # SRS, Data Provenance, Classifier/Router Metrics, and Module Roadmaps
├── CLAUDE.md                # Locked architectural decisions and stack constraints
└── rules.md                 # Agent and development operating rules
```

---

## Test Suite & Verification

The backend includes a comprehensive automated test suite spanning unit, resilience, data pipeline, and integration tests:

```bash
backend/.venv/Scripts/python.exe -m pytest
```

**Test Status:** `157 passed in 11.61s` (100% passing across 17 test modules):
- `test_eval.py`: Golden dataset schema validation, evaluation math, system metric aggregation, markdown reports, and mini-runs.
- `test_ml_router.py`: Feature builder, tier selection boundaries, and load/predict roundtrip.
- `test_ml_calibration.py`: Feature extraction, labeling rules, and threshold clamping.
- `test_dashboard.py`: Summary aggregation, cost accounting, quality report serving, and queue metrics.
- `test_redaction.py`: PII regex rules (emails, phone numbers, Luhn credit cards) and log filtering.
- `test_ml_classifier.py` & `test_ml_features.py`: TF-IDF extraction and XGBoost prediction stability.
- `test_ml_integration.py`: End-to-end classification through FastAPI endpoints.
- `test_guardrails.py`: Boundary verification ($49.99 allowed, $50.00 blocked) and HITL queue creation.
- `test_orchestrator_unit.py`: Transition table, retry bounds, router propagation, and calibrated threshold adjustments.
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
| **Module 7** | Cost Router & Calibration | ✅ Complete | Complexity router (78.8% acc vs 69.5% LR baseline) + dynamic calibration |
| **Module 8** | Evaluation Harness | ✅ Complete | 35 hand-verified golden test tickets, runner engine & baseline report |
| **Module 9** | Dashboard Frontend | ✅ Complete | Four-panel Recharts UI (Volume, Cost, Quality, Approvals) & live WS stream |
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
