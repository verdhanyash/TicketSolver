"""Persistence for agent observability (FR-10).

`TraceWriter` appends ordered `TraceStep` rows so every agent decision, tool call, LLM
call, and memory retrieval is recorded with its input, output, reasoning, model, and
latency. Callers pass their own `sessionmaker`; the module-level `trace_writer` is a
convenience bound to the app's default factory (constructed lazily enough that merely
importing this module never opens a connection).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import TraceStep
from app.db.session import get_session_factory


class TraceWriter:
    """Writes and reads per-ticket trace steps (FR-10)."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def log_step(
        self,
        *,
        ticket_id: str,
        step_type: str,
        name: str,
        input_payload: dict | None = None,
        output_payload: dict | None = None,
        reasoning: str | None = None,
        model: str | None = None,
        latency_ms: int | None = None,
    ) -> str:
        """Append one trace step for `ticket_id` and return its id."""
        with self._session_factory() as session:
            max_seq = session.scalar(
                select(func.coalesce(func.max(TraceStep.seq), 0)).where(
                    TraceStep.ticket_id == ticket_id
                )
            )
            step = TraceStep(
                id=uuid.uuid4().hex,
                ticket_id=ticket_id,
                seq=int(max_seq or 0) + 1,
                step_type=step_type,
                name=name,
                input_payload=input_payload,
                output_payload=output_payload,
                reasoning=reasoning,
                model=model,
                latency_ms=latency_ms,
            )
            session.add(step)
            session.commit()
            return step.id

    def get_trace(self, ticket_id: str) -> list[dict]:
        """Return all steps for `ticket_id`, ordered by seq, as plain dicts."""
        with self._session_factory() as session:
            steps = session.scalars(
                select(TraceStep)
                .where(TraceStep.ticket_id == ticket_id)
                .order_by(TraceStep.seq)
            ).all()
            return [
                {
                    "id": s.id,
                    "ticket_id": s.ticket_id,
                    "seq": s.seq,
                    "step_type": s.step_type,
                    "name": s.name,
                    "input_payload": s.input_payload,
                    "output_payload": s.output_payload,
                    "reasoning": s.reasoning,
                    "model": s.model,
                    "latency_ms": s.latency_ms,
                    "created_at": s.created_at.isoformat() if isinstance(s.created_at, datetime) else None,
                }
                for s in steps
            ]


try:  # pragma: no cover - depends on env; never connects at import time
    trace_writer = TraceWriter(get_session_factory())
except RuntimeError:  # DATABASE_URL unset (e.g. bare unit-test environment)
    trace_writer = None  # type: ignore[assignment]
