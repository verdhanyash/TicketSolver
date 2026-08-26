"""Pydantic schemas for ticket intake, investigation results, and traces."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TicketCreate(BaseModel):
    customer_id: str = Field(min_length=1, max_length=64)
    subject: str = Field(min_length=1, max_length=255)
    body: str = Field(min_length=1)


class TicketOut(BaseModel):
    id: str
    customer_id: str
    subject: str
    body: str
    status: str
    category: str | None = None  # filled by the ML classifier (FR-1)
    severity: str | None = None
    proposed_resolution: str | None = None


class InvestigateResult(BaseModel):
    ticket_id: str
    proposed_resolution: str
    trace_step_count: int


class InvestigationJobOut(BaseModel):
    """Async investigation job state (FR-4): submit returns 202 with this shape."""

    ticket_id: str
    status: str  # queued | running | completed | failed
    outcome: str | None = None  # resolved | escalated once completed
    attempts: int | None = None
    error: str | None = None


class TraceStepOut(BaseModel):
    seq: int
    step_type: str
    name: str
    input_payload: dict | None = None
    output_payload: dict | None = None
    reasoning: str | None = None
    model: str | None = None
    latency_ms: int | None = None
