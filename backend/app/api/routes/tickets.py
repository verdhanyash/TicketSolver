"""Ticket submission, retrieval, and investigation endpoints (FR-1, FR-2 entrypoint).

Investigation is job-based (FR-4): `POST .../investigate` enqueues background work
(202) and returns immediately; progress is available via the status endpoint and a
WebSocket broadcast on completion.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import Ticket
from app.db.session import get_session
from app.orchestration.worker import enqueue_investigation, get_job
from app.schemas.ticket import InvestigationJobOut, TicketCreate, TicketOut

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]


@router.post("", response_model=TicketOut, status_code=status.HTTP_201_CREATED)
def submit_ticket(payload: TicketCreate, session: SessionDep) -> Ticket:
    ticket = Ticket(customer_id=payload.customer_id, subject=payload.subject, body=payload.body)
    session.add(ticket)
    session.commit()
    return ticket


@router.get("/{ticket_id}", response_model=TicketOut)
def get_ticket(ticket_id: str, session: SessionDep) -> Ticket:
    ticket = session.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    return ticket


@router.post(
    "/{ticket_id}/investigate",
    response_model=InvestigationJobOut,
    status_code=status.HTTP_202_ACCEPTED,
)
def investigate_ticket(ticket_id: str, session: SessionDep) -> InvestigationJobOut:
    """Enqueue orchestrated processing; poll `GET .../investigate` or watch the WS."""
    ticket = session.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    job = enqueue_investigation(ticket.id, ticket.subject, ticket.body, ticket.customer_id)
    return InvestigationJobOut(
        ticket_id=job.ticket_id, status=job.status, outcome=job.outcome,
        attempts=job.attempts, error=job.error,
    )


@router.get("/{ticket_id}/investigate", response_model=InvestigationJobOut)
def investigation_status(ticket_id: str) -> InvestigationJobOut:
    job = get_job(ticket_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No investigation job for ticket")
    return InvestigationJobOut(
        ticket_id=job.ticket_id, status=job.status, outcome=job.outcome,
        attempts=job.attempts, error=job.error,
    )
