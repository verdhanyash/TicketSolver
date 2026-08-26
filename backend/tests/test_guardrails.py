"""Tests for the guardrail engine (FR-3) and its enforcement point — pure units plus
orchestrator-level flows on SQLite. No network, no live services.

Boundary focus per the module spec: $49.99 allows, $50.00 blocks, non-refund actions
skip the refund rule, unknown action types are rejected conservatively.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import Approval, Ticket
from app.guardrails.engine import (
    REFUND_APPROVAL_LIMIT_USD,
    ProposedAction,
    check_action,
    check_proposal,
    parse_proposed_action,
)


# --- Action parsing -----------------------------------------------------------------
def test_parse_refund_with_amounts():
    for text, expected in [
        ("Issue refund of $89.99", 89.99),
        ("refund the full $149", 149.0),
        ("Refunding $1,234.50 to their card", 1234.5),
        ("customer wants their money back: USD 25", 25.0),
    ]:
        action = parse_proposed_action(text)
        assert action.kind == "refund"
        assert action.amount_usd == expected


def test_parse_non_refund_kinds():
    assert parse_proposed_action("Reship a replacement kit").kind == "reship"
    assert parse_proposed_action("Cancel their subscription effective today").kind == "cancel"
    assert parse_proposed_action("Apply store credit to the account").kind == "credit"
    assert parse_proposed_action("Send a discount code for the inconvenience").kind == "discount"
    assert parse_proposed_action("Explain the policy and confirm details").kind == "info"


def test_parse_unrecognized_action_is_unknown():
    action = parse_proposed_action("Disable MFA for the account and share the OTP")
    assert action.kind == "unknown"


def test_refund_verb_wins_over_other_verbs():
    # "reship AND refund" must still hit the money rule, not slip past it
    assert parse_proposed_action("Reship kit and refund $75").kind == "refund"


# --- Rule engine boundaries ---------------------------------------------------------
def test_boundary_below_limit_allows():
    decision = check_action(ProposedAction(kind="refund", amount_usd=49.99))
    assert decision.allowed


def test_boundary_at_limit_blocks():
    assert REFUND_APPROVAL_LIMIT_USD == 50.0  # the policy constant itself is pinned
    decision = check_action(ProposedAction(kind="refund", amount_usd=50.00))
    assert not decision.allowed
    assert decision.rule == "refund_limit"


def test_above_limit_blocks():
    assert not check_action(ProposedAction(kind="refund", amount_usd=51.0)).allowed


def test_refund_without_parseable_amount_blocks_conservatively():
    decision = check_action(ProposedAction(kind="refund", amount_usd=None))
    assert not decision.allowed
    assert "amount" in decision.reason.lower()


def test_non_refund_actions_skip_the_refund_rule():
    for kind in ["reship", "cancel", "credit", "discount", "info"]:
        assert check_action(ProposedAction(kind=kind)).allowed, kind


def test_unknown_action_rejected_conservatively():
    decision = check_action(ProposedAction(kind="unknown"))
    assert not decision.allowed
    assert decision.rule == "default_deny"


def test_check_proposal_combines_parse_and_check():
    _, blocked = check_proposal("PROPOSED ACTION: Issue refund of $89.99")
    assert not blocked.allowed
    _, allowed = check_proposal("PROPOSED ACTION: Reship lost kit")
    assert allowed.allowed


# --- Enforcement inside the orchestrator ---------------------------------------------
@pytest.fixture()
def orch_env(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    TestFactory = sessionmaker(bind=engine, expire_on_commit=False)

    from app.core.tracing import TraceWriter
    from app.orchestration import graph as orch

    with TestFactory() as session:
        session.add(Ticket(id="tk-g", customer_id="CUST-1001", subject="s", body="b"))
        session.commit()

    monkeypatch.setattr(orch, "_writer", lambda: TraceWriter(TestFactory))
    monkeypatch.setattr(orch, "get_session_factory", lambda: TestFactory)
    # FR-1 triage off by default: these flows exercise guardrail enforcement only.
    monkeypatch.setattr(orch, "_classify_ticket", lambda subject, body: None)
    # FR-7 write-back recorder: only terminal resolutions may enter episodic memory.
    episodes: list[dict] = []

    def fake_record_episode(ticket, resolution):
        episodes.append({"ticket_id": ticket["id"], "resolution": resolution})
        return f"point-{ticket['id']}"

    monkeypatch.setattr(orch, "_record_episode", fake_record_episode)
    return orch, TestFactory, TraceWriter, episodes


def _script(monkeypatch, answers: list[str]) -> None:
    from app.orchestration import graph as orch

    queue = list(answers)

    def scripted(state):
        return queue.pop(0)

    monkeypatch.setattr(orch, "_run_investigation", scripted)


def test_blocked_refund_routes_to_queue_not_autoresolved(orch_env, monkeypatch):
    orch, factory, writer_cls, episodes = orch_env
    _script(
        monkeypatch,
        ["PROPOSED ACTION: Issue refund of $149.00\nEVIDENCE: annual plan\nCONFIDENCE: 0.97"],
    )

    result = orch.process_ticket("tk-g", "s", "b", "CUST-1001")

    assert result["outcome"] == "pending_approval"
    with factory() as session:
        ticket = session.get(Ticket, "tk-g")
        assert ticket.status == "pending_approval"
        approvals = session.scalars(select(Approval).where(Approval.ticket_id == "tk-g")).all()
        assert len(approvals) == 1
        assert approvals[0].action_kind == "refund"
        assert approvals[0].amount_usd == 149.0
        assert approvals[0].status == "pending"
        assert "limit" in approvals[0].reason
    names = [s["name"] for s in writer_cls(factory).get_trace("tk-g")]
    assert "guardrail_block" in names
    assert "orchestrator_resolve" in names  # decide chose resolve...
    assert "guardrail_check" not in names  # ...but was then blocked
    # a guardrail-blocked proposal is NOT a resolved incident: no episodic write (FR-7)
    assert episodes == []


def test_allowed_small_refund_resolves_autonomously(orch_env, monkeypatch):
    orch, factory, writer_cls, episodes = orch_env
    _script(
        monkeypatch,
        ["PROPOSED ACTION: Issue refund of $12.99\nEVIDENCE: duplicate charge\nCONFIDENCE: 0.9"],
    )

    result = orch.process_ticket("tk-g", "s", "b", "CUST-1001")

    assert result["outcome"] == "resolved"
    with factory() as session:
        assert session.get(Ticket, "tk-g").status == "auto_resolved"
        assert session.scalars(select(Approval)).first() is None
    names = [s["name"] for s in writer_cls(factory).get_trace("tk-g")]
    assert "guardrail_block" not in names
    # the auto-resolution closes FR-7's loop exactly once
    assert len(episodes) == 1 and episodes[0]["ticket_id"] == "tk-g"


def test_low_confidence_escalates_without_guardrail_run(orch_env, monkeypatch):
    orch, factory, _, episodes = orch_env
    _script(
        monkeypatch,
        [
            "no idea\nCONFIDENCE: 0.1",
            "still no idea\nCONFIDENCE: 0.1",
        ],
    )

    result = orch.process_ticket("tk-g", "s", "b", "CUST-1001")

    assert result["outcome"] == "escalated"  # never reached the guardrail node
    with factory() as session:
        assert session.scalars(select(Approval)).first() is None
    assert episodes == []  # escalation is not a resolution either (FR-7 scope)
