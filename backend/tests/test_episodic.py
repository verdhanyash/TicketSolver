"""Tests for episodic memory (FR-7) — no network, no live services.

Serialization and formatting are pure unit tests. Investigator-wiring tests script the
LLM and stub the retrieval seam, asserting past cases reach the prompt and retrieval
failures degrade without breaking flow; episodic *write-back* belongs to the terminal
resolution paths (orchestrator resolve node / approval endpoint) and is covered in
test_orchestrator_unit.py. (Live Qdrant/Ollama integration runs via
`scripts/seed_episodes.py --verify`; results are recorded in Docs/MODULES.md.)
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import Ticket  # importing also registers ORM tables for create_all
from app.memory.episodic import (
    episode_from_payload,
    episode_point_id,
    episode_to_payload,
    format_similar_incidents,
)


# --- Pure serialization / formatting -------------------------------------------
def test_episode_payload_round_trips():
    stored = episode_to_payload(
        ticket_id="T-1",
        customer_id="CUST-1001",
        subject="Kit never arrived",
        body="Ordered three weeks ago...",
        resolution="Refunded $89.99",
        category="shipping_issue",
    )
    loaded = episode_from_payload(dict(stored))
    assert loaded == {
        "ticket_id": "T-1",
        "customer_id": "CUST-1001",
        "subject": "Kit never arrived",
        "body": "Ordered three weeks ago...",
        "resolution": "Refunded $89.99",
        "category": "shipping_issue",
    }


def test_episode_payload_normalizes_and_truncates():
    stored = episode_to_payload(
        ticket_id="T-2",
        customer_id=None,
        subject="s",
        body="x" * 5000,
        resolution="y" * 5000,
        category=None,
    )
    loaded = episode_from_payload(stored)
    assert len(loaded["body"]) == 2000 and len(loaded["resolution"]) == 2000
    # unknown keys don't leak back out of a stored payload
    raw = {**stored, "vector_version": 9}
    assert "vector_version" not in episode_from_payload(raw)


def test_format_empty_cases_is_empty():
    assert format_similar_incidents([]) == ""


def test_format_lists_cases_in_order_with_scores():
    cases = [
        {"ticket_id": "A", "subject": "Lost kit", "resolution": "Refund", "score": 0.87},
        {"ticket_id": "B", "subject": "Duplicate charge", "resolution": "Refund $12.99", "score": 0.61},
    ]
    block = format_similar_incidents(cases)
    lines = block.splitlines()
    assert lines[0].startswith("SIMILAR PAST INCIDENTS")
    assert '1. "Lost kit" -> Resolution: Refund (similarity 0.87)' in lines
    assert '2. "Duplicate charge"' in block
    assert block.index("Lost kit") < block.index("Duplicate charge")


def test_episode_point_ids_deterministic_per_ticket():
    assert episode_point_id("T-1") == episode_point_id("T-1")
    assert episode_point_id("T-1") != episode_point_id("T-2")


# --- Investigator wiring ---------------------------------------------------------
HIT_A = {
    "ticket_id": "T-SEED-001",
    "customer_id": "CUST-1001",
    "subject": "Onboarding kit never arrived",
    "body": "...",
    "resolution": "Confirmed lost in transit; refunded $89.99.",
    "category": "shipping_issue",
    "score": 0.83,
}


@pytest.fixture()
def inv_env(monkeypatch):
    """Investigator against SQLite traces + scripted LLM; episodic seams stubbed."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    TestFactory = sessionmaker(bind=engine, expire_on_commit=False)

    from app.agents import investigator as inv
    from app.core.tracing import TraceWriter

    with TestFactory() as session:
        session.add(Ticket(id="t-5", customer_id="CUST-1001", subject="s", body="b"))
        session.commit()

    writer = TraceWriter(TestFactory)
    monkeypatch.setattr(inv, "_writer", lambda: writer)
    monkeypatch.setattr(inv, "get_session_factory", lambda: TestFactory)
    monkeypatch.setattr(inv, "_load_customer_memory", lambda customer_id: None)

    calls: dict = {"retrieved": []}

    def fake_find(text, ticket_id):
        calls["retrieved"].append((text, ticket_id))
        return []

    monkeypatch.setattr(inv, "_find_similar_incidents", fake_find)

    class ScriptedLLM:
        def __init__(self):
            self.calls = []

        def __call__(self, messages, tools=None):
            self.calls.append(list(messages))
            return AIMessage(content="PROPOSED ACTION: refund\nCUSTOMER REPLY: sorry\nEVIDENCE: x")

    scripted = ScriptedLLM()
    monkeypatch.setattr(inv, "_call_model", scripted)
    monkeypatch.setattr(inv, "_to_ai_message", lambda response: response)
    return inv, writer, scripted, calls


def test_similar_cases_reach_prompt_and_trace(inv_env, monkeypatch):
    inv, writer, scripted, calls = inv_env

    def find_with_hit(text, ticket_id):
        calls["retrieved"].append((text, ticket_id))
        return [HIT_A]

    monkeypatch.setattr(inv, "_find_similar_incidents", find_with_hit)

    inv.run_investigation("t-5", "Package lost", "never arrived", "CUST-1001")

    system_msg = scripted.calls[0][0].content
    assert "SIMILAR PAST INCIDENTS" in system_msg
    assert "Confirmed lost in transit; refunded $89.99." in system_msg
    # retrieval was queried with subject+body, excluding this ticket's own episodes
    text, excluded_id = calls["retrieved"][0]
    assert "Package lost" in text and excluded_id == "t-5"

    steps = writer.get_trace("t-5")
    epi = next(s for s in steps if s["name"] == "episodic_memory")
    assert epi["step_type"] == "memory_retrieval"
    assert epi["output_payload"]["hits"][0]["ticket_id"] == HIT_A["ticket_id"]


def test_investigation_alone_never_writes_back(inv_env):
    """FR-7 scope: episodic memory holds *resolved* incidents. A proposal only becomes
    resolved after the orchestrator's guardrail/decision layer (M4/M5), so recording
    happens in the resolve node / approval endpoint — never inside the investigator."""
    inv, writer, _, _ = inv_env

    proposal = inv.run_investigation("t-5", "s", "b", "CUST-1001")

    assert proposal.startswith("PROPOSED ACTION")
    assert not [s for s in writer.get_trace("t-5") if s["step_type"] == "memory_writeback"]


def test_retrieval_failure_degrades_without_breaking_flow(inv_env, monkeypatch):
    inv, _, scripted, _ = inv_env

    def boom(text, ticket_id):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(inv, "_find_similar_incidents", boom)

    proposal = inv.run_investigation("t-5", "s", "b", "CUST-1001")

    assert proposal.startswith("PROPOSED ACTION")
    assert "SIMILAR PAST INCIDENTS" not in scripted.calls[0][0].content
