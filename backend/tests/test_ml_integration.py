"""API-path integration for M6 ML triage (FR-1) riding the real FastAPI routes.

Hermetic throughout: SQLite (StaticPool) stands in for Postgres via the `get_session`
dependency override, the orchestrator's DB factory is repointed at the same engine, the
investigator seam is scripted, and the classifier seam returns canned predictions — no
network and no trained-model artifacts required.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.core.tracing import TraceWriter
from app.db.base import Base
from app.db.session import get_session
from app.main import app
from app.orchestration import graph as orch

client = TestClient(app)


@pytest.fixture()
def api_env(monkeypatch, tmp_path):
    # File-backed SQLite, not in-memory StaticPool: the job worker persists triage
    # predictions on its own thread while the request path polls, so the engine needs
    # real multi-connection locking — a single shared StaticPool connection lets the
    # two threads interleave transactions and intermittently lose writes.
    engine = create_engine(
        f"sqlite:///{tmp_path / 'api.db'}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA busy_timeout=15000")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    Base.metadata.create_all(engine)
    TestFactory = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_session():
        session = TestFactory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = override_get_session
    # Graph nodes resolve the DB through the module-level factory; repoint it too.
    monkeypatch.setattr(orch, "get_session_factory", lambda: TestFactory)
    # Episodic write-back (FR-7) would reach Qdrant/Ollama live services; record calls
    # instead so terminal resolutions still run their full path.
    episodes: list[tuple[str, str]] = []

    def fake_record_episode(ticket, resolution):
        episodes.append((ticket["id"], resolution))
        return f"point-{ticket['id']}"

    monkeypatch.setattr(orch, "_record_episode", fake_record_episode)

    yield TestFactory, episodes

    app.dependency_overrides.pop(get_session, None)


def _wait_for_completion(ticket_id: str, timeout_s: float = 10.0) -> dict:
    """Poll the async investigation job until it completes (worker thread does the work)."""
    job: dict = {}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = client.get(f"/api/tickets/{ticket_id}/investigate").json()
        if job["status"] == "completed":
            return job
        time.sleep(0.02)
    pytest.fail(f"investigation did not complete within {timeout_s}s: {job}")


def test_triaged_ticket_serves_classification_and_ordered_trace(api_env, monkeypatch):
    factory, _ = api_env

    created = client.post(
        "/api/tickets",
        json={
            "customer_id": "CUST-1001",
            "subject": "replacement kit never arrived",
            "body": "Tracking says delivered but nothing came. Please send a new one.",
        },
    )
    assert created.status_code == 201
    ticket_id = created.json()["id"]

    answer = (
        "PROPOSED ACTION: Reship the lost kit\nEVIDENCE: carrier confirmed loss\nCONFIDENCE: 0.93"
    )
    monkeypatch.setattr(orch, "_run_investigation", lambda state: answer)
    monkeypatch.setattr(
        orch,
        "_classify_ticket",
        lambda subject, body: SimpleNamespace(
            category="request",
            severity="medium",
            category_confidence=0.91,
            severity_confidence=0.55,
        ),
    )

    submitted = client.post(f"/api/tickets/{ticket_id}/investigate")
    assert submitted.status_code == 202
    assert submitted.json()["status"] in ("queued", "running")

    job = _wait_for_completion(ticket_id)
    assert job["outcome"] == "resolved"

    served = client.get(f"/api/tickets/{ticket_id}").json()
    assert served["status"] == "auto_resolved"
    assert served["category"] == "request"
    assert served["severity"] == "medium"

    steps = TraceWriter(factory).get_trace(ticket_id)
    assert steps[0]["step_type"] == "ml_prediction"
    assert steps[0]["name"] == "ticket_classification"
    assert steps[0]["output_payload"]["category"] == "request"
    assert steps[0]["output_payload"]["severity"] == "medium"
    assert steps[0]["reasoning"] == "XGBoost triage from subject+body (FR-1)"
    # seq strictly ordered, classification before every downstream decision step
    assert [s["seq"] for s in steps] == sorted(s["seq"] for s in steps)
    assert steps[0]["seq"] < min(s["seq"] for s in steps if s["step_type"] == "decision")


def test_ticket_without_model_serves_null_category(api_env, monkeypatch):
    """No trained model: the API serves category=null cleanly and the flow is unchanged."""
    created = client.post(
        "/api/tickets",
        json={
            "customer_id": "CUST-1001",
            "subject": "login broken",
            "body": "Cannot sign in since yesterday's update.",
        },
    )
    assert created.status_code == 201
    assert created.json()["category"] is None  # absent before any triage has run
    ticket_id = created.json()["id"]

    # info-kind proposal (guardrail-allowed) keeps this scenario focused on triage
    answer = (
        "PROPOSED ACTION: Explain the password reset steps to the customer\n"
        "EVIDENCE: e\nCONFIDENCE: 0.85"
    )
    monkeypatch.setattr(orch, "_run_investigation", lambda state: answer)
    monkeypatch.setattr(orch, "_classify_ticket", lambda subject, body: None)

    submitted = client.post(f"/api/tickets/{ticket_id}/investigate")
    assert submitted.status_code == 202

    job = _wait_for_completion(ticket_id)
    assert job["outcome"] == "resolved"

    served = client.get(f"/api/tickets/{ticket_id}").json()
    assert served["status"] == "auto_resolved"
    assert served["category"] is None
    assert served["severity"] is None
