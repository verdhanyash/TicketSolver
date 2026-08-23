# Software Requirements Specification
## Enterprise Support Agent Platform (Multi-Agent + ML)

**Version:** 1.0
**Date:** August 2026

---

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for a production-oriented, multi-agent AI system that automates customer support ticket resolution. The system uses coordinated LLM agents for reasoning and tool use, backed by trained ML models for cost routing and decision-confidence calibration, with full observability, guardrails, and human-in-the-loop controls.

### 1.2 Scope
The system will:
- Accept support tickets and resolve them autonomously when safe/confident to do so
- Maintain short-term, long-term, and episodic memory per customer
- Enforce hard-coded guardrails on risky actions (e.g., refunds)
- Escalate uncertain or policy-violating cases to a human reviewer
- Log full agent traces for observability and auditing
- Evaluate agent performance against a labeled test set
- Optimize LLM cost via ML-based ticket routing
- Visualize agent reasoning traces interactively

### 1.3 Intended Audience
Portfolio/CV project — technical reviewers (interviewers), and as a demo case for non-technical stakeholders evaluating agentic AI trustworthiness.

### 1.4 Definitions
- **Agent**: An LLM-driven reasoning loop with tool access and a defined role
- **Orchestrator**: Logic that routes control between agents based on outcomes
- **Guardrail**: A hard-coded policy check enforced in application code, not by the LLM
- **HITL**: Human-in-the-loop approval step
- **Trace**: A logged record of an agent's decisions, tool calls, and reasoning

---

## 2. Overall Description

### 2.1 Product Perspective
A standalone full-stack web application: a FastAPI backend running the agent system, a Postgres + vector database for memory/traces, and a React + Three.js frontend for ticket management and trace visualization.

### 2.2 User Classes
- **Customer** (submits tickets, simulated/synthetic for this project)
- **Support Agent Team (AI)**: Resolver, Memory-aware reasoning, Guardrail-checked action execution
- **Human Reviewer**: Approves/rejects escalated or flagged actions via the approval queue
- **Admin/Analyst**: Views the dashboard (cost, resolution rate, eval scores)

### 2.3 Assumptions & Constraints
- Built and run at $0 cost using free-tier services (see Section 7)
- No real customer/PII data — all tickets are synthetic or self-generated
- Not intended for real production traffic; portfolio/demo scale only

---

## 3. System Features (Functional Requirements)

### 3.1 Ticket Classification (ML-based, not an LLM agent)
- **FR-1**: System shall classify each incoming ticket's category and severity using a trained ML classifier (not an LLM call) — see Section 6.1. Category/severity output then drives a deterministic routing table to decide which agents are engaged.

### 3.2 Multi-Agent Core
- **FR-2**: An Investigator/Resolver Agent shall gather evidence via tool calls (account lookup, order status, knowledge base search) before proposing an action
- **FR-3**: A Guardrail layer shall block any action exceeding policy thresholds (e.g., refund > $50) and route it to the HITL queue instead of executing it
- **FR-4**: An Orchestrator shall decide next steps (resolve, retry, escalate) based on agent outputs — not a fixed sequential pipeline

### 3.3 Memory
- **FR-5**: System shall maintain short-term memory (current conversation) in-context per session
- **FR-6**: System shall maintain long-term memory (customer facts: plan tier, account age) in Postgres, retrieved at session start
- **FR-7**: System shall maintain episodic memory (past resolved incidents) as embeddings in a vector store, retrieved via similarity search for related tickets

### 3.4 Human-in-the-Loop
- **FR-8**: Any guardrail-flagged action shall appear in an approval queue with the agent's full reasoning trace visible
- **FR-9**: Human decisions (approve/reject/correct) shall be logged and optionally written back into episodic memory

### 3.5 Observability & Tracing
- **FR-10**: Every agent decision, tool call, and memory retrieval shall be logged with timestamp, input, output, and reasoning
- **FR-11**: Traces shall be viewable per-ticket in the frontend, including an interactive Three.js graph visualization of the agent flow (nodes = agents/tool calls, edges = decision flow, animated replay)

### 3.6 Evaluation
- **FR-12**: System shall maintain a labeled test set (30–50 synthetic tickets) with expected outcomes
- **FR-13**: System shall report accuracy metrics (correct resolution rate, correct escalation rate) after each agent logic change

### 3.7 Cost Optimization (ML Component)
- **FR-14**: An XGBoost classifier shall predict ticket complexity from engineered features (length, sentiment, keywords, customer history) to route tickets to a cheap vs. strong LLM
- **FR-15**: A calibration model shall estimate confidence in the agent's decision (beyond the LLM's self-reported confidence) to inform escalation thresholds

### 3.8 Frontend Dashboard

The dashboard is the primary artifact for demoing the system to both technical and non-technical viewers — it must tell the "is this system working and trustworthy" story at a glance, then allow drill-down.

- **FR-16**: Dashboard shall display four top-level panels:
  1. **Ticket Volume & Outcomes** — total tickets processed, % auto-resolved vs. % escalated to human, trend over time (line/bar chart via Recharts)
  2. **Cost Panel** — cost per ticket, total spend, $ saved by ML-based cheap/strong model routing vs. a naive "always use strongest model" baseline (bar comparison)
  3. **Quality/Eval Panel** — current eval harness scores (resolution correctness %, escalation correctness %), XGBoost/calibration model precision/recall, with feature importance chart for explainability (FR-18)
  4. **Live Approval Queue** — real-time list of guardrail-flagged tickets awaiting human review, each with a one-click expand into the full reasoning trace
- **FR-19**: Clicking any ticket (resolved, escalated, or in-queue) shall open a detail view containing the Three.js interactive trace visualization (FR-11) alongside the plain-text reasoning summary, so non-technical viewers can read the summary while technical viewers explore the graph
- **FR-20**: Dashboard shall update in near real-time via WebSocket as new tickets are processed, rather than requiring a manual refresh
- **FR-21**: Dashboard shall be usable without any explanation for a first-time non-technical viewer — panel titles and metrics must be self-descriptive (e.g., "Tickets Resolved Automatically" not "Auto-Resolution Rate (FR-1 compliant)")

---

## 4. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Reliability** | Tool calls wrapped with retry/timeout/circuit-breaker logic |
| **Scalability** | Tickets processed asynchronously; concurrent sessions supported |
| **Security** | No hardcoded secrets; least-privilege tool access; PII redaction in logs |
| **Cost** | Entire system buildable and runnable at $0 using free tiers |
| **Auditability** | Every autonomous action must be traceable to a logged reasoning chain |
| **Usability** | Non-technical reviewer should understand system status from the dashboard alone |

---

## 5. System Architecture (Summary)

```
Ticket Input
     │
     ▼
ML Classifier (category + severity)   ← trained model, not LLM
     │
     ▼
Deterministic Routing Table
     │
     ▼
Investigator Agent → Fix/Resolution Proposal
     │                        │
     ▼                        ▼
Episodic + Long-term Memory     Guardrail Check
(Postgres + Qdrant)                    │
                          ┌─────────┴─────────┐
                       Pass                 Blocked
                          │                     │
                    Auto-resolve          HITL Approval Queue
                          │                     │
                          └─────────┬───────────┘
                                    ▼
                           Trace Logging (all steps)
                                    │
                                    ▼
                    Dashboard (volume/cost/eval panels)
                                    │
                                    ▼
                    Three.js Trace Visualizer (per-ticket drill-down)
```

---

## 6. Machine Learning Models

This section lists every ML/NLP component in the system, beyond the LLM agents themselves (which use a pretrained model via API, not a model trained in-house).

### 6.1 Core Models (Required)

| Model | Purpose | Type | Training Data |
|---|---|---|---|
| **Ticket Classifier** (XGBoost or fine-tuned lightweight text classifier) | Classifies ticket category (billing/technical/refund/etc.) and severity — replaces what was originally an LLM "Triage Agent" | Supervised, text/tabular | Kaggle support ticket dataset with existing category/priority labels |
| **XGBoost (gradient boosted trees)** | Ticket complexity classification — routes tickets to a cheap vs. strong LLM tier | Supervised, tabular | Kaggle support ticket dataset, features engineered from ticket text (length, keywords, sentiment) + existing priority/category labels |
| **Confidence Calibration Model** (logistic regression or small gradient-boosted model) | Predicts whether the agent's proposed decision is likely correct; informs the escalation threshold — more reliable than trusting the LLM's self-reported confidence | Supervised, tabular | Features from agent/tool-call outcomes + labels from the eval set (correct vs. incorrect resolutions) |
| **Sentence Embedding Model** (sentence-transformers, or NVIDIA NIM's hosted embedding endpoint) | Converts ticket text and past incidents into vectors for retrieval — powers both episodic memory lookup and knowledge base RAG | Pretrained (used, not trained) | N/A — pretrained model, applied to your own ticket/KB text |

### 6.2 Optional / Stretch Models

| Model | Purpose | Type |
|---|---|---|
| **Clustering (k-means or HDBSCAN)** on ticket embeddings | Auto-discovers ticket categories/taxonomy not manually defined — useful analytics/insight feature | Unsupervised |
| **Anomaly Detection (Isolation Forest)** | Flags unusual spikes in a specific ticket category (e.g., signals a possible product bug) | Unsupervised |

### 6.3 Baseline for Comparison
- **Logistic Regression** shall be trained as a documented baseline for the complexity classifier, to be compared against XGBoost's performance (justifies the model choice with evidence rather than assumption).

### 6.4 Model Evaluation Requirements
- **FR-17**: The XGBoost classifier and calibration model shall each be evaluated with standard classification metrics (precision, recall, F1) against a held-out test split, reported alongside the agent-level eval scores (Section 3.5).
- **FR-18**: Feature importance for the XGBoost classifier shall be surfaced in the dashboard to support explainability for non-technical stakeholders.

---

## 7. Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| **LLM Inference** | NVIDIA NIM (build.nvidia.com) | Free tier, OpenAI-compatible, ~40 RPM |
| **Agent Orchestration** | LangGraph (Python) | Stateful multi-agent flow control |
| **Backend API** | FastAPI (Python) | Serves agent system + REST/WebSocket endpoints |
| **Relational DB** | PostgreSQL (Supabase or Neon free tier) | Long-term memory, tickets, traces |
| **Vector DB** | Qdrant (self-hosted, free) | Episodic memory + knowledge base RAG |
| **ML Models** | XGBoost, scikit-learn | Cost-routing classifier, confidence calibration |
| **Embeddings** | Sentence-transformers / NIM embedding models | For memory + retrieval |
| **Frontend Framework** | React | Ticket queue, approval UI, dashboard |
| **3D/Interactive Visualization** | Three.js | Interactive agent trace graph (nodes/edges, animated replay) |
| **Charts** | Recharts | Dashboard metrics (cost, resolution rate) |
| **Observability** | Custom Postgres trace tables (or self-hosted Langfuse) | Full decision/tool-call logging |
| **Hosting (Backend)** | Railway / Render free tier | |
| **Hosting (Frontend)** | Vercel free tier | |

---

## 8. Data Requirements

- **Primary ticket dataset**: Sourced from Kaggle (e.g., "Customer Support Ticket Dataset," ~8,469 rows with priority/category/satisfaction labels, or a larger 100k+ synthetic SaaS helpdesk dataset). Used to train the XGBoost complexity classifier and to seed realistic ticket examples. License to be verified per dataset before any public use.
- **Synthetic ticket dataset**: Additional tickets generated via LLM (NVIDIA NIM) to cover edge cases and difficulty levels not well represented in the Kaggle data, manually validated for realism
- **Eval set**: 30–50 hand-verified tickets (drawn from Kaggle data + synthetic) with known correct outcomes (resolve vs. escalate, correct action)
- **Knowledge base**: Small set of self-authored policy/FAQ documents (10–20 pages) for RAG grounding — company-specific, cannot come from Kaggle
- **Episodic memory seed data**: Subset of Kaggle tickets paired with plausible resolutions (self-written or LLM-generated) to give retrieval something real to match against

---

## 9. Out of Scope
- Real customer/PII data
- Real payment processing (refund tool is simulated)
- Multi-tenant auth/billing (optional stretch goal only)
- Production-scale traffic handling

---

## 10. Success Criteria
- End-to-end ticket flow works: submission → agent resolution or escalation → trace visible
- Guardrails demonstrably block unsafe actions in test cases
- Eval harness reports measurable accuracy (e.g., resolution correctness %, escalation correctness %)
- Cost-routing shows measurable $ savings vs. a naive "always use strongest model" baseline
- Three.js trace visualization renders and animates a real agent run
