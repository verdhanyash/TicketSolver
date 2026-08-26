"""Pydantic schemas for the HITL approval queue (FR-8, FR-9)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.ticket import TraceStepOut


class ApprovalOut(BaseModel):
    id: str
    ticket_id: str
    action_kind: str
    amount_usd: float | None = None
    reason: str
    status: str  # pending | approved | rejected | corrected
    decided_by: str | None = None
    decision_note: str | None = None
    corrected_action: str | None = None
    created_at: datetime | None = None
    decided_at: datetime | None = None


class ApprovalDetailOut(ApprovalOut):
    """Queue item plus the full ticket trace (FR-8: visible trace for reviewers)."""

    trace: list[TraceStepOut] = []


class DecisionIn(BaseModel):
    decision: Literal["approved", "rejected", "corrected"]
    note: str | None = Field(default=None, max_length=4000)
    corrected_action: str | None = Field(default=None, max_length=2000)
    decided_by: str = Field(min_length=1, max_length=64)
