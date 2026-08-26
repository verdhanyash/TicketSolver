"""Per-ticket trace retrieval (FR-10, FR-11): feeds the frontend trace viewer."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.tracing import TraceWriter
from app.db.session import get_session, get_session_factory
from app.schemas.ticket import TraceStepOut

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/{ticket_id}", response_model=list[TraceStepOut])
def get_trace(ticket_id: str, session: SessionDep) -> list[TraceStepOut]:
    steps = TraceWriter(get_session_factory()).get_trace(ticket_id)
    if not steps:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No trace for ticket")
    return [TraceStepOut(**s) for s in steps]
