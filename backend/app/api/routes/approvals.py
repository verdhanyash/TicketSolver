"""Human-in-the-loop approval queue endpoints (FR-8, FR-9).

Reviewers see guardrail-blocked actions with the ticket's full trace; their decisions
are traced, update the ticket outcome, and feed episodic memory (FR-9 write-back,
completing the M3 loop).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.websocket import broadcast_from_thread
from app.core.tracing import TraceWriter
from app.db.models import Approval, Ticket
from app.db.session import get_session, get_session_factory
from app.memory.episodic import record_episode
from app.schemas.approval import ApprovalDetailOut, ApprovalOut, DecisionIn
from app.schemas.ticket import TraceStepOut

logger = logging.getLogger(__name__)
router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]


def _to_out(approval: Approval) -> ApprovalOut:
    return ApprovalOut(
        id=approval.id,
        ticket_id=approval.ticket_id,
        action_kind=approval.action_kind,
        amount_usd=approval.amount_usd,
        reason=approval.reason,
        status=approval.status,
        decided_by=approval.decided_by,
        decision_note=approval.decision_note,
        corrected_action=approval.corrected_action,
        created_at=approval.created_at,
        decided_at=approval.decided_at,
    )


@router.get("", response_model=list[ApprovalOut])
def list_approvals(
    session: SessionDep, status_filter: str = Query("pending", alias="status")
) -> list[Approval]:
    """Queue listing; defaults to pending items (`?status=all` for everything)."""
    stmt = select(Approval).order_by(Approval.created_at)
    if status_filter != "all":
        stmt = stmt.where(Approval.status == status_filter)
    return list(session.scalars(stmt).all())


@router.get("/{approval_id}", response_model=ApprovalDetailOut)
def get_approval(approval_id: str, session: SessionDep) -> ApprovalDetailOut:
    approval = session.get(Approval, approval_id)
    if approval is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Approval not found")
    trace_steps = TraceWriter(get_session_factory()).get_trace(approval.ticket_id)
    base = _to_out(approval)
    return ApprovalDetailOut(
        **base.model_dump(),
        trace=[TraceStepOut(**{k: v for k, v in step.items() if k in TraceStepOut.model_fields}) for step in trace_steps],
    )


@router.post("/{approval_id}/decision", response_model=ApprovalOut)
def submit_decision(
    approval_id: str, payload: DecisionIn, session: SessionDep
) -> ApprovalOut:
    """Record a reviewer decision (FR-9): trace it, update the ticket, write back."""
    approval = session.get(Approval, approval_id)
    if approval is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Approval not found")
    if approval.status != "pending":
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail=f"Approval already decided ({approval.status})"
        )
    if payload.decision == "corrected" and not (payload.corrected_action or "").strip():
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="corrected decisions require corrected_action",
        )

    ticket = session.get(Ticket, approval.ticket_id)
    if ticket is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Ticket for approval not found")

    approval.status = payload.decision
    approval.decided_by = payload.decided_by
    approval.decision_note = payload.note
    approval.corrected_action = (
        payload.corrected_action.strip() if payload.corrected_action else None
    )
    approval.decided_at = datetime.now(UTC)
    session.commit()

    final_resolution = approval.corrected_action or ticket.proposed_resolution
    ticket_status = {
        "approved": "resolved",
        "rejected": "rejected",
        "corrected": "resolved",
    }[payload.decision]
    ticket.status = ticket_status
    if payload.decision == "corrected":
        ticket.proposed_resolution = approval.corrected_action
    session.commit()
    # FR-9 write-back: human-approved outcomes become retrievable episodes (M3 loop).
    if final_resolution and ticket_status == "resolved":
        try:
            record_episode(ticket, f"{final_resolution} [human-{payload.decision}]")
        except Exception:
            logger.exception("episodic write-back after decision failed")

    try:
        TraceWriter(get_session_factory()).log_step(
            ticket_id=approval.ticket_id,
            step_type="decision",
            name="human_review",
            input_payload={
                "approval_id": approval.id,
                "decision": payload.decision,
                "action_kind": approval.action_kind,
                "amount_usd": approval.amount_usd,
            },
            reasoning=payload.note or approval.reason,
        )
    except Exception:
        logger.exception("failed to trace human_review decision")

    broadcast_from_thread(
        {
            "type": "approval_decided",
            "approval_id": approval.id,
            "ticket_id": approval.ticket_id,
            "decision": payload.decision,
        }
    )
    return _to_out(approval)
