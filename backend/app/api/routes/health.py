"""Health check — used by tests and uptime probes."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ticketsolver-backend"}
