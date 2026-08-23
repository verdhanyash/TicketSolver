# CLAUDE.md — TicketSolver

Locked tech-stack decisions and conventions for the TicketSolver project. **Read this
before adding any dependency or service.** If a need arises that isn't covered here, ask
before introducing a new library/framework (see `rules.md`).

Full requirements: `Docs/SRS_Enterprise_Support_Agent_Platform.md` (referred to elsewhere
as "the SRS"; note the request called it `docs/SRS.md` — the actual path is the former).

## Project in one line
Multi-agent (LLM) + ML support-automation platform: classify → investigate → guardrail →
auto-resolve or escalate (HITL) → trace everything → visualize + report. Portfolio/demo
scale, built to run at **$0 on free tiers**, with **no real PII** (synthetic tickets only).

## Locked stack

| Layer | Decision | Notes |
|---|---|---|
| LLM inference | **NVIDIA NIM** (build.nvidia.com) | OpenAI-compatible API (use the `openai` client pointed at NIM base URL); free tier ~40 RPM. Two tiers: a cheap and a strong model, chosen per ticket by the ML router. |
| Agent orchestration | **LangGraph** (Python) | Stateful multi-agent flow; the Orchestrator is graph-driven, not a fixed pipeline (FR-4). |
| Backend API | **FastAPI** (Python ≥3.11) | REST + WebSocket. Entry: `backend/app/main.py`. |
| Relational DB | **PostgreSQL** (Supabase / Neon free tier) | Tickets, long-term customer memory (FR-6), and trace logs (FR-10). Access via SQLAlchemy; migrations via Alembic. |
| Vector DB | **Qdrant** (self-hosted, free) | Episodic memory (FR-7) + knowledge-base RAG. |
| ML — classification | **XGBoost** + **scikit-learn** | Ticket category/severity classifier (FR-1), complexity→cost router (FR-14), confidence calibration (FR-15), logistic-regression baseline (SRS §6.3). |
| Embeddings | **sentence-transformers** (fallback: NIM embedding endpoint) | Powers episodic memory + KB retrieval. Pretrained, not trained in-house. |
| Frontend framework | **React + TypeScript** (Vite) | Dashboard, approval UI, ticket detail. |
| 3D / trace viz | **Three.js via react-three-fiber + @react-three/drei** | Interactive animated agent-trace graph (FR-11, FR-19). |
| Charts | **Recharts** | Dashboard metrics (FR-16). |
| Observability | Custom Postgres trace tables | Full decision/tool-call logging (FR-10/11). Langfuse optional later. |
| Hosting | Backend: Railway/Render free tier · Frontend: Vercel free tier | Deferred. |

**Do not** swap these for equivalents (e.g., don't add OpenAI/Anthropic SDKs for the LLM,
Pinecone for vectors, LightGBM for XGBoost, Plotly/D3 for charts, or plain three.js without
react-three-fiber) without explicit sign-off.

## Repository layout

```
backend/
  app/
    main.py            FastAPI app: CORS, routers, WebSocket, /api/health
    config.py          pydantic-settings (env-driven; no hardcoded secrets)
    api/               routers (routes/*) + websocket manager
    core/              logging + PII redaction (NFR: Security)
    db/                SQLAlchemy session + ORM models (Postgres)
    vector/            Qdrant client
    llm/               NVIDIA NIM (OpenAI-compatible) client
    agents/            LLM reasoning agents — Investigator/Resolver (FR-2)  [no logic yet]
    orchestration/     LangGraph flow control (FR-4)                         [no logic yet]
    guardrails/        hard-coded policy checks, e.g. refund > $50 (FR-3)    [no logic yet]
    memory/            short-term / long-term (Postgres) / episodic (Qdrant) (FR-5–7) [stub]
    ml/                classifier, cost router, calibration (FR-1/14/15)     [no logic yet]
    eval/              labeled test-set harness + metrics (FR-12/13/17)      [no logic yet]
    schemas/           pydantic request/response models
  tests/               pytest (asyncio); run relevant tests before "done"
frontend/
  src/
    api/               REST + WebSocket client
    hooks/             e.g. useWebSocket (FR-20 live dashboard)
    types/             shared Ticket / Trace / Approval types
    components/
      dashboard/       4 panels: TicketVolume, Cost, QualityEval, ApprovalQueue (FR-16)
      trace/           react-three-fiber trace viewer (FR-11/19)
      approvals/       HITL approval queue items (FR-8)
    pages/             DashboardPage, TicketDetailPage
```

## Conventions
- **Config/secrets**: everything through `app/config.py` (pydantic-settings) / `.env`.
  Never hardcode secrets; `.env` is gitignored, `.env.example` is the template.
- **Guardrails are code, not prompts** (FR-3): risky-action limits live in `app/guardrails/`,
  enforced in application code — never delegated to the LLM.
- **Auditability** (NFR): every autonomous action must trace to a logged reasoning chain.
- **PII redaction** in logs (NFR: Security); tickets are synthetic regardless.
- **Reliability**: wrap tool calls with retry/timeout/circuit-breaker when implemented.
- **Frontend for humans** (FR-21): panel titles/metrics must be self-descriptive for a
  non-technical first-time viewer.
- **Dependency versions**: declared in `backend/pyproject.toml` and `frontend/package.json`;
  pin to exact versions on first install.

## Data (SRS §8)
Kaggle support-ticket dataset (train XGBoost + seed examples) + LLM-generated synthetic
edge cases + a 30–50 ticket hand-verified eval set + a small self-authored KB for RAG.
No dataset committed until license is verified.
