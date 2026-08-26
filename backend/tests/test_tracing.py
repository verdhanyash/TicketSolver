"""Unit tests for TraceWriter (FR-10) against SQLite in-memory — no Postgres needed."""

from __future__ import annotations

import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.tracing import TraceWriter
from app.db.models import Base, Ticket


def _make_writer() -> tuple[TraceWriter, str]:
    """Fresh in-memory DB with one seeded ticket; returns the writer and the ticket id."""
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    tid = uuid.uuid4().hex
    with factory() as session:
        session.add(
            Ticket(
                id=tid,
                customer_id="cust-001",
                subject="Cannot log in",
                body="Password reset email never arrives.",
            )
        )
        session.commit()
    return TraceWriter(factory), tid


def test_log_step_assigns_sequential_seq() -> None:
    writer, tid = _make_writer()

    id1 = writer.log_step(ticket_id=tid, step_type="memory_retrieval", name="qdrant_search")
    id2 = writer.log_step(ticket_id=tid, step_type="tool_call", name="kb_lookup")

    assert id1 != id2
    trace = writer.get_trace(tid)
    assert [s["seq"] for s in trace] == [1, 2]
    assert [s["name"] for s in trace] == ["qdrant_search", "kb_lookup"]
    assert [s["step_type"] for s in trace] == ["memory_retrieval", "tool_call"]


def test_get_trace_fields_and_payload_round_trip() -> None:
    writer, tid = _make_writer()
    payload_in = {"query": "password reset", "top_k": 3}
    payload_out = {"hits": [{"chunk": "reset-guide", "score": 0.91}]}

    writer.log_step(
        ticket_id=tid,
        step_type="llm_call",
        name="investigator_llm",
        input_payload=payload_in,
        output_payload=payload_out,
        reasoning="Retrieved KB chunks before drafting.",
    )

    (step,) = writer.get_trace(tid)
    assert step["id"] and step["ticket_id"] == tid
    assert step["seq"] == 1
    assert step["step_type"] == "llm_call"
    assert step["name"] == "investigator_llm"
    assert step["input_payload"] == payload_in
    assert step["output_payload"] == payload_out
    assert step["reasoning"] == "Retrieved KB chunks before drafting."
    assert isinstance(step["created_at"], str) and "T" in step["created_at"]


def test_latency_ms_and_model_stored() -> None:
    writer, tid = _make_writer()

    writer.log_step(
        ticket_id=tid,
        step_type="decision",
        name="severity_classifier",
        model="meta/llama-3.1-8b-instruct",
        latency_ms=142,
    )

    (step,) = writer.get_trace(tid)
    assert step["model"] == "meta/llama-3.1-8b-instruct"
    assert step["latency_ms"] == 142
