# TicketSolver

Production-oriented, multi-agent AI platform that automates customer-support ticket
resolution — with ML-based cost routing, hard-coded guardrails, human-in-the-loop
approval, full trace observability, and an interactive 3D trace visualizer.

Full spec: [`Docs/SRS_Enterprise_Support_Agent_Platform.md`](Docs/SRS_Enterprise_Support_Agent_Platform.md).
Stack decisions and repo conventions: [`CLAUDE.md`](CLAUDE.md).
Operating rules for AI coding sessions: [`rules.md`](rules.md).

## Layout

```
TicketSolver/
├── backend/     FastAPI app + agents, orchestration, ML, memory (Python)
├── frontend/    React + TypeScript + Vite dashboard & 3D trace viewer
├── Docs/        SRS and design docs
├── CLAUDE.md    Locked tech-stack decisions
└── rules.md     Coding-agent operating rules
```

## Status

Session 1 — repository scaffold and governance docs only. **No agent/ML/orchestration
logic is implemented yet.** See each `backend/app/<package>/__init__.py` for the intended
responsibility of that module and the SRS requirements it will satisfy.

## Getting started (next session)

```bash
# Backend
cd backend
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -e ".[dev]"
uvicorn app.main:app --reload
pytest

# Frontend
cd frontend
npm install
npm run dev
```
