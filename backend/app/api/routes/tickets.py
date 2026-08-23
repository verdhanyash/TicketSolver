"""Ticket submission and retrieval endpoints (FR-1, FR-2 flow entrypoint).

Stubs only — classification, agent processing, and persistence arrive in later sessions.
"""

from __future__ import annotations

from fastapi import APIRouter, status

router = APIRouter()


@router.get("")
def list_tickets():
    raise _not_implemented("list tickets")


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def submit_ticket():
    raise _not_implemented("submit ticket")


@router.get("/{ticket_id}")
def get_ticket(ticket_id: str):
    raise _not_implemented("get ticket")


def _not_implemented(what: str):
    from fastapi import HTTPException

    return HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=f"Not implemented: {what}")
