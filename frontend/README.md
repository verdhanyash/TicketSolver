# TicketSolver — Frontend

React + TypeScript + Vite dashboard, HITL approval UI, and the interactive 3D agent-trace
viewer (Three.js via react-three-fiber). See `../CLAUDE.md` for stack decisions.

## Setup (deps not installed during scaffold session)

```bash
npm install
cp .env.example .env      # optional; defaults proxy to the backend on :8000
```

## Run

```bash
npm run dev               # http://localhost:5173
npm run build             # type-check + production build
npm run typecheck         # tsc --noEmit
```

## Structure
- `components/dashboard/` — four self-descriptive panels (FR-16)
- `components/trace/` — react-three-fiber trace viewer (FR-11/19)
- `components/approvals/` — HITL approval queue (FR-8)
- `pages/` — DashboardPage, TicketDetailPage
- `api/` + `hooks/` — REST client and live-update WebSocket (FR-20)

All components are placeholders; no data wiring yet.
