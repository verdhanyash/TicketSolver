"""Per-ticket trace retrieval endpoints (FR-10, FR-11).

Returns the logged reasoning/tool-call graph that the frontend Three.js viewer renders.
Stubs only — trace persistence and serialization arrive in later sessions.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

router = APIRouter()


@router.get("/{ticket_id}")
def get_trace(ticket_id: str):
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented: ticket trace")
