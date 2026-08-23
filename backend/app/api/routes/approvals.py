"""Human-in-the-loop approval queue endpoints (FR-8, FR-9).

Stubs only — guardrail-flagged actions and reviewer decisions arrive in later sessions.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

router = APIRouter()


@router.get("")
def list_pending_approvals():
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented: approval queue")


@router.post("/{ticket_id}/decision")
def submit_decision(ticket_id: str):
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented: approval decision")
