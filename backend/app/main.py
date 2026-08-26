"""FastAPI application entrypoint for TicketSolver.

Wires up CORS, the REST API router, and the dashboard WebSocket. Agent logic lives in
`app.agents`, flow control in `app.orchestration`; this module only composes them.

Run: `uvicorn app.main:app --reload`
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.api.websocket import capture_main_loop, register_websocket
from app.config import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Worker threads broadcast WS events onto this loop (M4).
    capture_main_loop()
    yield


app = FastAPI(
    title="TicketSolver API",
    version="0.1.0",
    description="Multi-agent support-automation platform backend.",
    lifespan=lifespan,
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
