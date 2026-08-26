"""Unit tests for the Investigator agent — no network, no live services.

The LLM is replaced with a scripted sequence of responses (tool call, then final
answer); Qdrant/Ollama and the trace writer are swapped for in-memory fakes.
Verifies the graph loops through tools, produces a proposal, and writes FR-10 traces.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import Ticket


def _ai(content: str = "", tool_calls: list | None = None) -> AIMessage:
    return AIMessage(content=content, tool_calls=tool_calls or [])


class ScriptedLLM:
    """Returns canned AIMessages in order; records prompts it saw."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.calls: list[list] = []

    def __call__(self, messages, tools=None):
        self.calls.append([m.content for m in messages])
        return self.responses.pop(0)


def script_llm(monkeypatch, inv, responses: list[AIMessage]) -> ScriptedLLM:
    """Patch both the LLM call and the OpenAI-response unwrap for tests."""
    scripted = ScriptedLLM(responses)
    monkeypatch.setattr(inv, "_call_model", scripted)
    monkeypatch.setattr(inv, "_to_ai_message", lambda response: response)
    return scripted


@pytest.fixture()
def trace_env(monkeypatch):
    """SQLite-backed DB + real TraceWriter; patches investigator internals."""
    from sqlalchemy.pool import StaticPool

    # StaticPool keeps every session on the SAME in-memory sqlite DB.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    TestFactory = sessionmaker(bind=engine, expire_on_commit=False)

    with TestFactory() as session:
        session.add(Ticket(id="t-1", customer_id="CUST-1001", subject="s", body="b"))
        session.commit()

    from app.agents import investigator as inv
    from app.core.tracing import TraceWriter

    writer = TraceWriter(TestFactory)
    monkeypatch.setattr(inv, "_writer", lambda: writer)
    # finalize_node calls get_session_factory()() — return the factory so the
    # extra call yields a Session.
    monkeypatch.setattr(inv, "get_session_factory", lambda: TestFactory)
    # Keep memory loading hermetic/deterministic; specific tests override this.
    monkeypatch.setattr(inv, "_load_customer_memory", lambda customer_id: None)
    # Episodic memory (M3) exercised in test_episodic.py; neutralize here.
    # (Write-back lives in the orchestrator's resolve node, not the investigator.)
    monkeypatch.setattr(inv, "_find_similar_incidents", lambda text, ticket_id: [])
    # Tool reliability policy (M4) has its own tests; run tools directly here.
    monkeypatch.setattr(inv, "_execute_tool", lambda fn, **kwargs: fn(**kwargs))
    return inv, writer, TestFactory


def test_agent_loops_through_tools_and_proposes(trace_env, monkeypatch):
    inv, writer, factory = trace_env

    scripted = script_llm(monkeypatch, inv, [
        _ai(tool_calls=[{"name": "kb_search", "args": {"query": "refund window"}, "id": "call_1"}]),
        _ai("PROPOSED ACTION: Reship kit\nCUSTOMER REPLY: Sorry!\nEVIDENCE: policy says X"),
    ])
    monkeypatch.setitem(
        inv.TOOL_REGISTRY,
        "kb_search",
        lambda query: [{"text": "Refunds within 14 days.", "source": "refund-policy.md", "score": 0.9}],
    )

    proposal = inv.run_investigation("t-1", "Where is my refund?", "I want a refund", "CUST-1001")

    assert "PROPOSED ACTION" in proposal
    # LLM was called twice: once before tools, once after results fed back.
    assert len(scripted.calls) == 2
    # Tool result message reached the model on the second call.
    assert any("kb_search" in str(m) or "Refunds" in str(m) for m in [scripted.calls[1]])

    steps = writer.get_trace("t-1")
    types = [(s["step_type"], s["name"]) for s in steps]
    assert ("llm_call", "investigator_llm") in types
    assert ("tool_call", "kb_search") in types
    assert ("decision", "proposed_resolution") in types
    # seq ordering is 1..n
    assert [s["seq"] for s in steps] == list(range(1, len(steps) + 1))
    # proposal persisted to the ticket row
    with factory() as session:
        assert "PROPOSED ACTION" in session.get(Ticket, "t-1").proposed_resolution


def test_unknown_tool_returns_error_to_model(trace_env, monkeypatch):
    inv, writer, _ = trace_env
    script_llm(monkeypatch, inv, [
        _ai(tool_calls=[{"name": "does_not_exist", "args": {}, "id": "call_x"}]),
        _ai("PROPOSED ACTION: Apologize\nCUSTOMER REPLY: ...\nEVIDENCE: none"),
    ])
    monkeypatch.setitem(inv.TOOL_REGISTRY, "does_not_exist", None)

    proposal = inv.run_investigation("t-1", "s", "b", "CUST-1001")
    assert proposal.startswith("PROPOSED ACTION")
    tool_steps = [s for s in writer.get_trace("t-1") if s["step_type"] == "tool_call"]
    assert len(tool_steps) == 1


def test_tool_exception_is_caught_and_traced(trace_env, monkeypatch):
    inv, writer, _ = trace_env

    def boom(**kwargs):
        raise RuntimeError("qdrant down")

    script_llm(monkeypatch, inv, [
        _ai(tool_calls=[{"name": "kb_search", "args": {"query": "q"}, "id": "c"}]),
        _ai("PROPOSED ACTION: Escalate\nCUSTOMER REPLY: ...\nEVIDENCE: retrieval failed"),
    ])
    monkeypatch.setitem(inv.TOOL_REGISTRY, "kb_search", boom)

    inv.run_investigation("t-1", "s", "b", "CUST-1001")
    tool_steps = [s for s in writer.get_trace("t-1") if s["step_type"] == "tool_call"]
    assert len(tool_steps) == 1  # failure still traced, not raised


def test_tool_schemas_are_well_formed():
    from app.agents.investigator import TOOL_REGISTRY, TOOL_SCHEMAS

    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == set(TOOL_REGISTRY) == {"kb_search", "lookup_account", "get_order_status"}
