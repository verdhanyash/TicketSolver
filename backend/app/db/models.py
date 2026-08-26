"""SQLAlchemy ORM models (PostgreSQL).

Holds ticket records, long-term customer memory (FR-6), and the full per-ticket decision
traces (FR-10): every agent step — memory retrieval, tool call, LLM call, or decision —
is persisted with its input, output, reasoning, model, and latency.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

# Single shared declarative base; re-exported here for convenience.
from app.db.base import Base


class Ticket(Base):
    """An inbound customer support ticket."""

    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(32))  # filled by the ML classifier
    severity: Mapped[str | None] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="open")
    proposed_resolution: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class CustomerMemory(Base):
    """Long-term per-customer facts (FR-6), loaded into agent context at session start."""

    __tablename__ = "customer_memory"

    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    plan_tier: Mapped[str | None] = mapped_column(String(32))
    account_age_days: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[list | None] = mapped_column(JSON)  # ordered free-text facts
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Approval(Base):
    """A guardrail-blocked action awaiting (or having received) human review (FR-8/9)."""

    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)
    ticket_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tickets.id"), nullable=False, index=True
    )
    action_kind: Mapped[str] = mapped_column(String(32), nullable=False)  # refund | ...
    amount_usd: Mapped[float | None] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="pending")
    decided_by: Mapped[str | None] = mapped_column(String(64))
    decision_note: Mapped[str | None] = mapped_column(Text)
    corrected_action: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TraceStep(Base):
    """One logged agent step (memory_retrieval | tool_call | llm_call | decision) — FR-10."""

    __tablename__ = "trace_steps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)
    ticket_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tickets.id"), nullable=False, index=True
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)  # ordering within a ticket
    step_type: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. tool name / "investigator_llm"
    input_payload: Mapped[dict | None] = mapped_column(JSON)
    output_payload: Mapped[dict | None] = mapped_column(JSON)
    reasoning: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(128))  # producing LLM/model, if applicable
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
