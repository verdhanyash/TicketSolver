"""FastAPI application entrypoint for TicketSolver.

Wires up CORS, the REST API router, and the dashboard WebSocket. No agent, ML, or
orchestration logic lives here — routes are stubs until later sessions (see CLAUDE.md).

Run: `uvicorn app.main:app --reload`
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.api.websocket import register_websocket
from app.config import settings

app = FastAPI(
    title="TicketSolver API",
    version="0.1.0",
    description="Multi-agent support-automation platform backend.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
register_websocket(app)
