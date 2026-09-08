"""Dashboard summary aggregation tests (FR-16) — hermetic SQLite via dependency override.

StaticPool in-memory SQLite stands in for Postgres through the `get_session`
override; the eval-report path is repointed at temp files so Module 8's real
artifact never influences these runs.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import dashboard as dashboard_routes
from app.db.base import Base
from app.db.models import Approval, Ticket, TraceStep
from app.db.session import get_session
from app.main import app

client = TestClient(app)


@pytest.fixture()
def db_factory(monkeypatch):
    """Fresh in-memory DB wired into the app; returns its sessionmaker."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_session():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = override_get_session
    # Keep the eval-report seam away from any real backend/eval/report.json.
    monkeypatch.setattr(dashboard_routes, "EVAL_REPORT_PATH", Path("no") / "such" / "report.json")
    yield factory
    app.dependency_overrides.pop(get_session, None)


def _add_ticket(factory, *, status: str = "open", created_at: datetime | None = None) -> str:
    ticket_id = uuid.uuid4().hex
    with factory() as session:
        session.add(
            Ticket(
                id=ticket_id,
                customer_id="CUST-1001",
                subject="Order never arrived",
                body="Tracking says delivered but nothing came.",
                status=status,
                created_at=created_at,
            )
        )
        session.commit()
    return ticket_id


def _add_approval(factory, *, status: str = "pending", created_at: datetime | None = None,
                  ticket_id: str | None = None) -> str:
    approval_id = uuid.uuid4().hex
    with factory() as session:
        session.add(
            Approval(
                id=approval_id,
                ticket_id=ticket_id or uuid.uuid4().hex,
                action_kind="refund",
                amount_usd=149.0,
                reason="Refund exceeds the $50 approval limit",
                status=status,
                created_at=created_at,
            )
        )
        session.commit()
    return approval_id


def _add_step(factory, ticket_id: str, *, step_type: str, output_payload: dict | None = None,
              input_payload: dict | None = None) -> None:
    with factory() as session:
        max_seq = session.scalar(
            select(func.coalesce(func.max(TraceStep.seq), 0)).where(
                TraceStep.ticket_id == ticket_id
            )
        )
        session.add(
            TraceStep(
                id=uuid.uuid4().hex,
                ticket_id=ticket_id,
                seq=int(max_seq or 0) + 1,
                step_type=step_type,
                name="test_step",
                input_payload=input_payload,
                output_payload=output_payload,
            )
        )
        session.commit()


def test_empty_db_returns_clean_zeros(db_factory):
    data = client.get("/api/dashboard/summary").json()

    assert data["volume"]["total_tickets"] == 0
    assert data["volume"]["by_status"] == {
        "auto_resolved": 0,
        "resolved": 0,
        "pending_approval": 0,
        "escalated": 0,
        "rejected": 0,
        "open": 0,
    }
    assert data["volume"]["auto_resolution_rate_pct"] == 0.0

    daily = data["volume"]["daily"]
    assert len(daily) == 14
    today = datetime.now(UTC).date()
    assert daily[-1]["date"] == today.isoformat()
    assert daily[0]["date"] == (today - timedelta(days=13)).isoformat()
    assert all(point["total"] == 0 for point in daily)

    assert data["cost"] == {
        "has_usage_data": False,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert data["quality"]["available"] is False
    assert data["quality"]["report"] is None
    assert data["queue"] == {"pending_count": 0, "oldest_pending_age_minutes": None}
    assert datetime.fromisoformat(data["generated_at"]) is not None


def test_summary_counts_statuses_rates_series_cost_and_queue(db_factory):
    factory = db_factory
    now = datetime.now(UTC)
    today = now.replace(hour=10, minute=0, second=0, microsecond=0)
    three_days_ago = today - timedelta(days=3)
    month_ago = today - timedelta(days=30)

    seeded = [
        ("auto_resolved", today),
        ("auto_resolved", today),
        ("auto_resolved", three_days_ago),
        ("auto_resolved", month_ago),  # outside the 14-day window, still in totals
        ("resolved", today),
        ("resolved", three_days_ago),
        ("pending_approval", three_days_ago),
        ("escalated", today),
        ("rejected", three_days_ago),
        ("open", three_days_ago),
    ]
    ids = {status: _add_ticket(factory, status=status, created_at=at) for status, at in seeded}

    # Token usage on llm_call steps only: direct fields + a nested usage dict.
    _add_step(
        factory, ids["auto_resolved"],
        step_type="llm_call",
        output_payload={"content": "...", "prompt_tokens": 120, "completion_tokens": 80,
                        "total_tokens": 200},
    )
    _add_step(
        factory, ids["resolved"],
        step_type="llm_call",
        output_payload={"content": "...", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    )
    # Non-LLM steps must not inflate the cost totals.
    _add_step(factory, ids["open"], step_type="tool_call", output_payload={"prompt_tokens": 999})

    pending_created = now - timedelta(minutes=90)
    _add_approval(factory, status="pending", created_at=pending_created, ticket_id=ids["open"])
    _add_approval(factory, status="approved", created_at=pending_created, ticket_id=ids["resolved"])

    data = client.get("/api/dashboard/summary").json()

    volume = data["volume"]
    assert volume["total_tickets"] == 10
    assert volume["by_status"] == {
        "auto_resolved": 4,
        "resolved": 2,
        "pending_approval": 1,
        "escalated": 1,
        "rejected": 1,
        "open": 1,
    }
    assert volume["auto_resolution_rate_pct"] == 40.0

    daily = volume["daily"]
    assert len(daily) == 14
    assert daily[0]["date"] == (now.date() - timedelta(days=13)).isoformat()
    assert daily[-1]["date"] == now.date().isoformat()
    assert sum(point["total"] for point in daily) == 9  # the 30-day-old ticket stays out
    assert daily[-1]["total"] == 4  # 2 auto_resolved + resolved + escalated today
    assert daily[-1]["auto_resolved"] == 2
    assert daily[-1]["resolved"] == 1
    assert daily[-1]["escalated"] == 1
    day_minus_3 = daily[10]
    assert day_minus_3["date"] == (now.date() - timedelta(days=3)).isoformat()
    assert day_minus_3 == {
        "date": day_minus_3["date"],
        "total": 5,
        "auto_resolved": 1,
        "resolved": 1,
        "pending_approval": 1,
        "escalated": 0,
        "rejected": 1,
        "in_progress": 1,
    }

    assert data["cost"] == {
        "has_usage_data": True,
        "llm_calls": 2,
        "prompt_tokens": 130,
        "completion_tokens": 85,
        "total_tokens": 215,
    }

    queue = data["queue"]
    assert queue["pending_count"] == 1
    assert 89 <= queue["oldest_pending_age_minutes"] <= 92


def test_quality_report_served_when_present(db_factory, monkeypatch, tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"resolution_correctness_pct": 93.3, "tickets": 40}),
                           encoding="utf-8")
    monkeypatch.setattr(dashboard_routes, "EVAL_REPORT_PATH", report_path)

    data = client.get("/api/dashboard/summary").json()

    assert data["quality"]["available"] is True
    assert data["quality"]["error"] is None
    assert data["quality"]["report"] == {"resolution_correctness_pct": 93.3, "tickets": 40}


def test_quality_report_unreadable_degrades_to_available_false(db_factory, monkeypatch, tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(dashboard_routes, "EVAL_REPORT_PATH", report_path)

    data = client.get("/api/dashboard/summary").json()

    assert data["quality"]["available"] is False
    assert data["quality"]["report"] is None
    assert "unreadable" in data["quality"]["error"]
