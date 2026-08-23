"""Aggregates all REST route modules into a single router mounted at /api."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import approvals, dashboard, health, tickets, traces

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(tickets.router, prefix="/tickets", tags=["tickets"])
api_router.include_router(approvals.router, prefix="/approvals", tags=["approvals"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(traces.router, prefix="/traces", tags=["traces"])
