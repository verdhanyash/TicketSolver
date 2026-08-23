# TicketSolver — Backend

FastAPI service hosting the multi-agent system, ML models, memory, and trace APIs.
See `../CLAUDE.md` for stack decisions and `../rules.md` for operating rules.

## Setup (deps not installed during scaffold session)

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash;  .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
cp .env.example .env                # then fill in NIM / Postgres / Qdrant values
```

## Run

```bash
uvicorn app.main:app --reload       # http://127.0.0.1:8000  (health: /api/health)
```

## Test

```bash
pytest                              # run relevant tests only when iterating (see rules.md)
```

## Structure
Each package under `app/` owns one SRS concern; see its `__init__.py` docstring for the
intended responsibility. No agent/ML/orchestration logic is implemented yet.
