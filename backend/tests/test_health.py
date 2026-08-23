"""Smoke test: the app boots and the health endpoint responds.

Requires the dev deps installed (`pip install -e ".[dev]"`). During the scaffold session
these are not installed; this test runs green once they are.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok() -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
