"""Dashboard metrics endpoints (FR-16: volume/outcomes, cost, eval, queue).

Stubs only — metric aggregation arrives in later sessions.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

router = APIRouter()


@router.get("/summary")
def dashboard_summary():
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented: dashboard summary")
