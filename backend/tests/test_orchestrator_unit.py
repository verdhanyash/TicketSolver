"""Unit tests for the orchestrator (FR-4) — no network, no live services.

The transition table and confidence parsing are pure. Full flows run the real LangGraph
with a scripted investigation seam, a SQLite trace writer, and a SQLite ticket store,
asserting resolve / retry / escalate behavior end-to-end at graph level.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.db.base import Base
from app.db.models import Ticket
from app.orchestration import graph as orch


# --- Pure decision logic -----------------------------------------------------------
def test_decide_next_transition_table():
    strong = "PROPOSED ACTION: refund"
    # confident proposal → resolve regardless of attempt budget left
    assert orch.decide_next(attempt=1, confidence=0.9, proposal=strong) == orch.RESOLVE
    # weak outcome with attempts remaining → retry (defaults: max 2 attempts, 0.6 bar)
    assert orch.decide_next(attempt=1, confidence=0.59, proposal=strong) == orch.RETRY
    # weak outcome out of attempts → escalate (terminal safety state)
    assert orch.decide_next(attempt=2, confidence=0.59, proposal=strong) == orch.ESCALATE
    # empty proposal never resolves, even with self-reported high confidence
    assert orch.decide_next(attempt=1, confidence=1.0, proposal="") == orch.RETRY
    assert orch.decide_next(attempt=2, confidence=1.0, proposal="   ") == orch.ESCALATE


def test_decide_next_boundaries_and_overrides():
    strong = "PROPOSED ACTION: x"
    # exactly at threshold counts as confident
    assert (
        orch.decide_next(attempt=1, confidence=settings.confidence_threshold, proposal=strong)
        == orch.RESOLVE
    )
    # explicit overrides beat settings (used by tests/future calibration)
    assert (
        orch.decide_next(attempt=3, confidence=0.1, proposal=strong, max_attempts=5)
        == orch.RETRY
    )
    assert (
        orch.decide_next(attempt=1, confidence=0.9, proposal=strong, threshold=0.95)
        == orch.RETRY
    )


def test_confidence_extraction_and_stripping():
    raw = "PROPOSED ACTION: reship\nEVIDENCE: policy\nCONFIDENCE: 0.8"
    proposal, conf = orch.strip_confidence(raw)
    assert conf == 0.8
    assert "CONFIDENCE" not in proposal and proposal.endswith("policy")
    # case/space tolerant form
    assert orch.extract_confidence("confidence = 0.55") == 0.55
    # missing line → default (treated as confident; the meta-line is optional)
    assert orch.extract_confidence("PROPOSED ACTION: x") == 1.0
    # numeric out-of-range values clamp into [0, 1]...
    assert orch.extract_confidence("CONFIDENCE: 4.2") == 1.0
    # ...but non-numeric garbage never parses → documented default
    assert orch.extract_confidence("CONFIDENCE: -3") == 1.0
    assert orch.extract_confidence("CONFIDENCE: high") == 1.0


# --- Full orchestrated flows --------------------------------------------------------
@pytest.fixture()
def orch_env(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    TestFactory = sessionmaker(bind=engine, expire_on_commit=False)

    from app.core.tracing import TraceWriter

    with TestFactory() as session:
        session.add(Ticket(id="tk-1", customer_id="CUST-1001", subject="s", body="b"))
        session.commit()

    monkeypatch.setattr(orch, "_writer", lambda: TraceWriter(TestFactory))
    monkeypatch.setattr(orch, "get_session_factory", lambda: TestFactory)
    # FR-1 triage off by default: these flows pin pure orchestrator behavior.
    monkeypatch.setattr(orch, "_classify_ticket", lambda subject, body: None)

    # Episodic write-back (FR-7) recorder: lets tests assert exactly which outcomes
    # enter episodic memory.
    episodes: list[dict] = []

    def fake_record_episode(ticket, resolution):
        episodes.append({"ticket_id": ticket["id"], "resolution": resolution})
        return f"point-{ticket['id']}"

    monkeypatch.setattr(orch, "_record_episode", fake_record_episode)

    class ScriptedInvestigations:
        """Returns canned investigator outputs in order; records what it was told."""

        def __init__(self, answers: list[str]):
            self.answers = list(answers)
            self.calls: list[dict] = []

        def __call__(self, state):
            self.calls.append(
                {"session_turns": list(state.get("session_turns", [])), "nudges": list(state.get("nudges", []))}
            )
            return self.answers.pop(0)

    return TestFactory, ScriptedInvestigations, episodes


def _script(monkeypatch, orch_env, answers) -> tuple:
    factory, scripted_cls, episodes = orch_env
    scripted = scripted_cls(answers)
    monkeypatch.setattr(orch, "_run_investigation", scripted)
    return factory, scripted, episodes


def test_confident_ticket_resolves_first_pass(orch_env, monkeypatch):
    from app.core.tracing import TraceWriter

    factory, _, episodes = _script(
        monkeypatch,
        orch_env,
        ["PROPOSED ACTION: refund $12\nCUSTOMER REPLY: ok\nEVIDENCE: e\nCONFIDENCE: 0.9"],
    )

    result = orch.process_ticket("tk-1", "s", "b", "CUST-1001")

    assert result == {
        "outcome": "resolved",
        "attempts": 1,
        "proposal": "PROPOSED ACTION: refund $12\nCUSTOMER REPLY: ok\nEVIDENCE: e",
    }
    with factory() as session:
        ticket = session.get(Ticket, "tk-1")
        assert ticket.status == "auto_resolved"
        assert ticket.proposed_resolution.startswith("PROPOSED ACTION")
        assert "CONFIDENCE" not in ticket.proposed_resolution
    decisions = [s for s in TraceWriter(factory).get_trace("tk-1") if s["step_type"] == "decision"]
    assert any(s["name"] == "orchestrator_resolve" for s in decisions)
    # FR-7 closure: the auto-resolution (and only it) enters episodic memory, traced.
    assert episodes == [
        {
            "ticket_id": "tk-1",
            "resolution": "PROPOSED ACTION: refund $12\nCUSTOMER REPLY: ok\nEVIDENCE: e",
        }
    ]
    wb = next(s for s in TraceWriter(factory).get_trace("tk-1") if s["step_type"] == "memory_writeback")
    assert wb["name"] == "episode_writeback"
    assert wb["output_payload"]["point_id"] == "point-tk-1"


def test_low_confidence_retries_then_escalates(orch_env, monkeypatch):
    from app.core.tracing import TraceWriter

    factory, scripted, episodes = _script(
        monkeypatch,
        orch_env,
        [
            "PROPOSED ACTION: guess\nEVIDENCE: none\nCONFIDENCE: 0.2",
            "PROPOSED ACTION: another guess\nEVIDENCE: still none\nCONFIDENCE: 0.1",
        ],
    )

    result = orch.process_ticket("tk-1", "s", "b", "CUST-1001")

    assert result["outcome"] == "escalated"
    assert result["attempts"] == 2  # bounded by orchestrator_max_attempts
    with factory() as session:
        assert session.get(Ticket, "tk-1").status == "escalated"
    # escalation is not a resolution: nothing enters episodic memory (FR-7 scope)
    assert episodes == []
    steps = TraceWriter(factory).get_trace("tk-1")
    decision_names = [s["name"] for s in steps if s["step_type"] == "decision"]
    assert decision_names == ["orchestrator_retry", "orchestrator_escalate"]
    # the retry nudge reached the second investigation as accumulated context
    assert len(scripted.calls) == 2
    assert len(scripted.calls[0]["nudges"]) == 0
    assert "[orchestrator]" in scripted.calls[1]["nudges"][0]


def test_resolve_node_survives_episodic_write_failure(orch_env, monkeypatch):
    """A Qdrant outage must not turn a resolution into a failed run."""
    factory, _, episodes = _script(
        monkeypatch, orch_env, ["PROPOSED ACTION: Reship lost kit\nCONFIDENCE: 0.9"]
    )

    def boom(ticket, resolution):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(orch, "_record_episode", boom)

    result = orch.process_ticket("tk-1", "s", "b", "CUST-1001")

    assert result["outcome"] == "resolved"
    with factory() as session:
        assert session.get(Ticket, "tk-1").status == "auto_resolved"
    assert episodes == []


def test_session_turns_reach_investigation_fr5(orch_env, monkeypatch):
    _, scripted, _ = _script(
        monkeypatch, orch_env, ["PROPOSED ACTION: Reship lost kit\nCONFIDENCE: 0.9"]
    )
    orch.record_session_turn("tk-fr5", "customer", "also my email is broken")

    result = orch.process_ticket(
        "tk-fr5", "s", "b", "CUST-1001", session_turns=orch.get_session_context("tk-fr5")
    )

    assert result["outcome"] == "resolved"
    assert scripted.calls[0]["session_turns"] == ["customer: also my email is broken"]


def test_follow_up_accumulates_short_term_context(orch_env, monkeypatch):
    _, scripted, _ = _script(
        monkeypatch,
        orch_env,
        [
            "PROPOSED ACTION: Reship lost kit first time\nCONFIDENCE: 0.9",
            "PROPOSED ACTION: Reship lost kit again\nCONFIDENCE: 0.9",
        ],
    )
    args = ("tk-fu", "subject about billing", "body", "CUST-1001")

    first = orch.process_ticket(*args)
    second = orch.follow_up("tk-fu", "it happened again today", *args[1:])

    assert first["outcome"] == "resolved" and second["outcome"] == "resolved"
    # second run saw the recorded customer turn as session context
    assert any("it happened again today" in t for t in scripted.calls[1]["session_turns"])
    ctx = orch.get_session_context("tk-fu")
    assert ctx[0].startswith("customer:")
    assert any(turn.startswith("agent:") for turn in ctx)


# --- FR-1 ML triage ------------------------------------------------------------------
def test_ml_triage_persists_columns_and_traces_ahead_of_routing(orch_env, monkeypatch):
    from app.core.tracing import TraceWriter

    factory, _, episodes = _script(
        monkeypatch, orch_env, ["PROPOSED ACTION: refund $12\nEVIDENCE: e\nCONFIDENCE: 0.9"]
    )

    def fake_classify(subject, body):
        assert (subject, body) == ("s", "b")  # classified from the raw intake fields
        return SimpleNamespace(
            category="incident",
            severity="high",
            category_confidence=0.87,
            severity_confidence=0.44,
        )

    monkeypatch.setattr(orch, "_classify_ticket", fake_classify)

    result = orch.process_ticket("tk-1", "s", "b", "CUST-1001")

    assert result["outcome"] == "resolved"
    with factory() as session:
        ticket = session.get(Ticket, "tk-1")
        assert ticket.category == "incident"
        assert ticket.severity == "high"
        assert ticket.status == "auto_resolved"  # triage left the graph's fields alone
        assert ticket.proposed_resolution.startswith("PROPOSED ACTION")
    steps = TraceWriter(factory).get_trace("tk-1")
    ml = next(s for s in steps if s["step_type"] == "ml_prediction")
    assert ml["name"] == "ticket_classification"
    assert ml["output_payload"] == {
        "category": "incident",
        "severity": "high",
        "category_confidence": 0.87,
        "severity_confidence": 0.44,
    }
    assert ml["reasoning"] == "XGBoost triage from subject+body (FR-1)"
    assert isinstance(ml["latency_ms"], int)
    # FR-1 ordering: the prediction lands before every routing/decision step
    assert ml["seq"] < min(s["seq"] for s in steps if s["step_type"] == "decision")
    # the run itself was untouched by triage: same resolution + episodic closure (FR-7)
    assert episodes and episodes[0]["ticket_id"] == "tk-1"


def test_ml_triage_unavailable_leaves_row_and_trace_untouched(orch_env, monkeypatch):
    """Graceful degradation locked in: no model → NULL columns, no trace, same outcome."""
    from app.core.tracing import TraceWriter

    factory, _, episodes = _script(
        monkeypatch, orch_env, ["PROPOSED ACTION: reship\nEVIDENCE: e\nCONFIDENCE: 0.9"]
    )
    monkeypatch.setattr(orch, "_classify_ticket", lambda subject, body: None)

    result = orch.process_ticket("tk-1", "s", "b", "CUST-1001")

    assert result["outcome"] == "resolved"
    with factory() as session:
        ticket = session.get(Ticket, "tk-1")
        assert ticket.category is None
        assert ticket.severity is None
    steps = TraceWriter(factory).get_trace("tk-1")
    assert not any(s["step_type"] == "ml_prediction" for s in steps)
    assert episodes  # the resolution path is unchanged


def test_ml_triage_failure_is_swallowed_not_fatal(orch_env, monkeypatch):
    """Even a crashing classifier cannot break orchestration (FR-7 guard mirrored)."""
    from app.core.tracing import TraceWriter

    factory, _, _ = _script(
        monkeypatch, orch_env, ["PROPOSED ACTION: reship\nEVIDENCE: e\nCONFIDENCE: 0.9"]
    )

    def boom(subject, body):
        raise RuntimeError("classifier crashed")

    monkeypatch.setattr(orch, "_classify_ticket", boom)

    result = orch.process_ticket("tk-1", "s", "b", "CUST-1001")

    assert result["outcome"] == "resolved"
    with factory() as session:
        ticket = session.get(Ticket, "tk-1")
        assert ticket.category is None and ticket.severity is None
    steps = TraceWriter(factory).get_trace("tk-1")
    assert not any(s["step_type"] == "ml_prediction" for s in steps)
