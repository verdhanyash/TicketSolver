"""Tests for long-term customer memory (FR-6) — no network, no live services.

Formatter cases are pure unit tests. Persistence round-trips use an in-memory SQLite
DB via a StaticPool-backed sessionmaker (same pattern as test_tracing.py /
test_investigator_unit.py). Investigator-wiring tests script the LLM and assert the
customer facts actually reach the system prompt and the FR-10 trace.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.memory.long_term import (
    CustomerMemoryRecord,
    format_customer_memory,
    get_customer_memory,
    upsert_customer_memory,
)


# --- Formatter (pure unit) -----------------------------------------------------
def test_format_none_is_empty():
    assert format_customer_memory(None) == ""


def test_format_all_fields_unset_is_empty():
    # A row that exists but carries no facts is omitted gracefully too.
    empty = CustomerMemoryRecord(customer_id="CUST-1001")
    assert format_customer_memory(empty) == ""


def test_format_populated_orders_facts():
    record = CustomerMemoryRecord(
        customer_id="CUST-1003",
        plan_tier="enterprise",
        account_age_days=1287,
        notes=["Escalation path is their named CSM.", "Renewal window opens Nov 2026."],
    )
    block = format_customer_memory(record)
    lines = block.splitlines()
    assert lines[0] == "KNOWN CUSTOMER FACTS (CUST-1003, from long-term memory):"
    assert "- Plan tier: enterprise" in lines
    assert "- Account age: 1287 days" in lines
    assert "- Note 1: Escalation path is their named CSM." in lines
    assert "- Note 2: Renewal window opens Nov 2026." in lines
    # fixed fact order: tier before age before notes, notes in stored order
    assert block.index("Plan tier") < block.index("Account age") < block.index("Note 1")


def test_format_partial_record_includes_only_present_fields():
    block = format_customer_memory(CustomerMemoryRecord(customer_id="X", plan_tier="free"))
    assert "- Plan tier: free" in block
    assert "Account age" not in block
    assert "Note" not in block


# --- Persistence round-trip (in-memory SQLite) ----------------------------------
@pytest.fixture()
def memory_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_upsert_then_get_round_trips_identically(memory_factory):
    record = CustomerMemoryRecord(
        customer_id="CUST-1001",
        plan_tier="pro",
        account_age_days=412,
        notes=["Prefers email follow-ups.", "Goodwill discount applied once."],
    )
    upsert_customer_memory(record, session_factory=memory_factory)

    loaded = get_customer_memory("CUST-1001", session_factory=memory_factory)
    assert loaded == record  # dataclass eq covers every field incl. note order


def test_upsert_is_idempotent_and_get_unknown_returns_none(memory_factory):
    record = CustomerMemoryRecord(
        customer_id="CUST-1002", plan_tier="free", account_age_days=23, notes=["a"]
    )
    upsert_customer_memory(record, session_factory=memory_factory)
    updated = CustomerMemoryRecord(
        customer_id="CUST-1002", plan_tier="pro", account_age_days=24, notes=["a", "b"]
    )
    upsert_customer_memory(updated, session_factory=memory_factory)

    from sqlalchemy import func, select

    from app.db.models import CustomerMemory

    with memory_factory() as session:
        count = session.scalar(
            select(func.count()).select_from(CustomerMemory).where(
                CustomerMemory.customer_id == "CUST-1002"
            )
        )
    assert count == 1  # updated in place, not duplicated
    assert get_customer_memory("CUST-1002", session_factory=memory_factory) == updated
    assert get_customer_memory("CUST-9999", session_factory=memory_factory) is None


# --- Investigator wiring: facts must reach prompt AND trace ----------------------
@pytest.fixture()
def inv_env(monkeypatch):
    """Investigator against SQLite: real TraceWriter, scripted memory, ticket seeded."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    TestFactory = sessionmaker(bind=engine, expire_on_commit=False)

    from app.agents import investigator as inv
    from app.core.tracing import TraceWriter
    from app.db.models import Ticket

    with TestFactory() as session:
        session.add(Ticket(id="t-9", customer_id="CUST-1001", subject="s", body="b"))
        session.commit()

    writer = TraceWriter(TestFactory)
    monkeypatch.setattr(inv, "_writer", lambda: writer)
    monkeypatch.setattr(inv, "get_session_factory", lambda: TestFactory)

    class ScriptedLLM:
        def __init__(self):
            self.calls = []

        def __call__(self, messages, tools=None):
            self.calls.append(list(messages))
            return AIMessage(content="PROPOSED ACTION: ok\nCUSTOMER REPLY: hi\nEVIDENCE: x")

    scripted = ScriptedLLM()
    monkeypatch.setattr(inv, "_call_model", scripted)
    monkeypatch.setattr(inv, "_to_ai_message", lambda response: response)
    # Episodic memory is exercised in test_episodic.py; neutralize here.
    monkeypatch.setattr(inv, "_find_similar_incidents", lambda text, ticket_id: [])
    return inv, writer, scripted


def test_facts_reach_system_prompt_and_trace(inv_env, monkeypatch):
    inv, writer, scripted = inv_env
    monkeypatch.setattr(
        inv,
        "_load_customer_memory",
        lambda customer_id: CustomerMemoryRecord(
            customer_id=customer_id,
            plan_tier="pro",
            account_age_days=412,
            notes=["Prefers email follow-ups."],
        ),
    )

    inv.run_investigation("t-9", "s", "b", "CUST-1001")

    system_msg = scripted.calls[0][0].content
    assert "KNOWN CUSTOMER FACTS (CUST-1001, from long-term memory):" in system_msg
    assert "- Plan tier: pro" in system_msg
    assert "Prefers email follow-ups." in system_msg

    steps = writer.get_trace("t-9")
    mem_step = next(s for s in steps if s["step_type"] == "memory_retrieval")
    assert mem_step["seq"] == 1  # retrieval logged before any LLM/tool activity
    assert mem_step["input_payload"] == {"customer_id": "CUST-1001"}
    assert mem_step["output_payload"]["found"] is True
    assert "Prefers email follow-ups." in mem_step["output_payload"]["facts_block"]
    # llm_call input payload records the assembled context for auditability (FR-6/10)
    llm_step = next(s for s in steps if s["step_type"] == "llm_call")
    assert "KNOWN CUSTOMER FACTS" in llm_step["input_payload"]["system_prompt"]


def test_no_memory_omits_block_and_traces_found_false(inv_env, monkeypatch):
    inv, writer, scripted = inv_env
    monkeypatch.setattr(inv, "_load_customer_memory", lambda customer_id: None)

    inv.run_investigation("t-9", "s", "b", "CUST-1001")

    assert "KNOWN CUSTOMER FACTS" not in scripted.calls[0][0].content
    mem_step = next(s for s in writer.get_trace("t-9") if s["step_type"] == "memory_retrieval")
    assert mem_step["output_payload"]["found"] is False


def test_memory_failure_degrades_without_breaking_flow(inv_env, monkeypatch):
    inv, _, scripted = inv_env

    def boom(customer_id):
        raise RuntimeError("postgres down")

    monkeypatch.setattr(inv, "_load_customer_memory", boom)

    proposal = inv.run_investigation("t-9", "s", "b", "CUST-1001")

    assert proposal.startswith("PROPOSED ACTION")  # flow completed despite failure
    assert "KNOWN CUSTOMER FACTS" not in scripted.calls[0][0].content
