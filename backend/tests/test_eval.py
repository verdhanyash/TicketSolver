"""Unit and integration tests for Evaluation Harness (Module 8, FR-12, FR-13, FR-17)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes.dashboard import QualityOut
from app.db.base import Base
from app.db.models import Ticket
from app.eval.runner import (
    aggregate_system_metrics,
    compute_eval_metrics,
    format_markdown_report,
    load_golden_tickets,
    run_evaluation,
)


def test_load_golden_tickets_schema():
    tickets = load_golden_tickets()
    assert len(tickets) >= 30, f"Expected at least 30 golden tickets, got {len(tickets)}"

    required_keys = {
        "ticket_id",
        "customer_id",
        "subject",
        "body",
        "scenario",
        "expected_outcome",
        "expected_action",
        "rationale",
    }
    scenarios = set()
    outcomes = set()

    for ticket in tickets:
        missing = required_keys - set(ticket.keys())
        assert not missing, f"Ticket {ticket.get('ticket_id')} missing keys: {missing}"
        assert ticket["scenario"] in ("routine", "guardrail_trigger", "edge_case")
        assert ticket["expected_outcome"] in ("resolved", "pending_approval", "escalated")
        scenarios.add(ticket["scenario"])
        outcomes.add(ticket["expected_outcome"])

    assert scenarios == {"routine", "guardrail_trigger", "edge_case"}
    assert outcomes == {"resolved", "pending_approval", "escalated"}


def test_load_golden_tickets_invalid_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_golden_tickets(tmp_path / "non_existent.json")

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a non-empty list"):
        load_golden_tickets(bad_json)


def test_compute_eval_metrics_math():
    fixture_results = [
        {
            "ticket_id": "T1",
            "scenario": "routine",
            "expected_outcome": "resolved",
            "actual_outcome": "resolved",
            "is_outcome_correct": True,
            "is_category_correct": True,
            "is_severity_correct": True,
        },
        {
            "ticket_id": "T2",
            "scenario": "routine",
            "expected_outcome": "resolved",
            "actual_outcome": "resolved",
            "is_outcome_correct": True,
            "is_category_correct": False,
            "is_severity_correct": True,
        },
        {
            "ticket_id": "T3",
            "scenario": "guardrail_trigger",
            "expected_outcome": "pending_approval",
            "actual_outcome": "pending_approval",
            "is_outcome_correct": True,
            "is_category_correct": True,
            "is_severity_correct": True,
        },
        {
            "ticket_id": "T4",
            "scenario": "edge_case",
            "expected_outcome": "escalated",
            "actual_outcome": "escalated",
            "is_outcome_correct": True,
            "is_category_correct": True,
            "is_severity_correct": False,
        },
        {
            "ticket_id": "T5",
            "scenario": "edge_case",
            "expected_outcome": "escalated",
            "actual_outcome": "resolved",  # false resolution
            "is_outcome_correct": False,
            "is_category_correct": True,
            "is_severity_correct": True,
        },
    ]

    metrics = compute_eval_metrics(fixture_results)
    assert metrics["total_tickets"] == 5
    assert metrics["correct_outcomes"] == 4
    assert metrics["outcome_accuracy_pct"] == 80.0
    assert metrics["resolution_correctness_pct"] == 100.0  # 2/2 routine correct
    assert metrics["guardrail_intercept_pct"] == 100.0     # 1/1 guardrail correct
    assert metrics["escalation_correctness_pct"] == 50.0   # 1/2 edge correct

    # Escalation TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5, F1=0.6667
    esc = metrics["escalation_metrics"]
    assert esc["true_positives"] == 1
    assert esc["false_positives"] == 0
    assert esc["false_negatives"] == 1
    assert esc["precision"] == 1.0
    assert esc["recall"] == 0.5
    assert esc["f1"] == 0.6667

    # Confusion matrix checks
    cm = metrics["confusion_matrix"]
    assert cm["resolved"]["resolved"] == 2
    assert cm["pending_approval"]["pending_approval"] == 1
    assert cm["escalated"]["escalated"] == 1
    assert cm["escalated"]["resolved"] == 1


def test_aggregate_system_metrics(tmp_path: Path):
    # Test with non-existent directory (graceful degradation)
    empty_res = aggregate_system_metrics(tmp_path)
    assert empty_res["ticket_classifier_m6"] is None
    assert empty_res["cost_router_m7"] is None
    assert empty_res["confidence_calibration_m7"] is None

    # Test with mock metrics files
    (tmp_path / "metrics.json").write_text(
        json.dumps({
            "category": {"accuracy": 0.85, "macro_f1": 0.85},
            "severity": {"accuracy": 0.61, "macro_f1": 0.59},
            "dataset": {"test_rows": 2060},
        }),
        encoding="utf-8",
    )
    (tmp_path / "router_metrics.json").write_text(
        json.dumps({
            "xgboost": {"accuracy": 0.78, "macro_f1": 0.78},
            "baseline_logistic_regression": {"accuracy": 0.69, "macro_f1": 0.68},
            "comparison": {"advantage_f1": 0.10, "gate_passed": True},
        }),
        encoding="utf-8",
    )
    (tmp_path / "calibration_metrics.json").write_text(
        json.dumps({
            "metrics": {"accuracy": 1.0, "brier_score": 0.012, "log_loss": 0.05},
            "gate": {"passed": True},
        }),
        encoding="utf-8",
    )

    loaded = aggregate_system_metrics(tmp_path)
    assert loaded["ticket_classifier_m6"]["category_macro_f1"] == 0.85
    assert loaded["cost_router_m7"]["gate_passed"] is True
    assert loaded["confidence_calibration_m7"]["brier_gate_passed"] is True


def test_format_markdown_report():
    sample_report = {
        "eval_run_id": "EVAL-TEST-001",
        "generated_at": "2026-09-09T00:00:00Z",
        "summary": {
            "total_tickets": 35,
            "correct_outcomes": 35,
            "outcome_accuracy_pct": 100.0,
            "resolution_correctness_pct": 100.0,
            "guardrail_intercept_pct": 100.0,
            "escalation_correctness_pct": 100.0,
            "escalation_metrics": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            "scenario_breakdown": {
                "routine": {"total": 18, "correct": 18, "accuracy_pct": 100.0},
                "guardrail_trigger": {"total": 9, "correct": 9, "accuracy_pct": 100.0},
                "edge_case": {"total": 8, "correct": 8, "accuracy_pct": 100.0},
            },
            "confusion_matrix": {
                "resolved": {"resolved": 18, "pending_approval": 0, "escalated": 0},
                "pending_approval": {"resolved": 0, "pending_approval": 9, "escalated": 0},
                "escalated": {"resolved": 0, "pending_approval": 0, "escalated": 8},
            },
        },
        "quality_gates": {
            "routine_resolution_gate": True,
            "guardrail_safety_gate": True,
            "escalation_f1_gate": True,
            "classifier_m6_gate": True,
            "router_m7_gate": True,
            "calibration_m7_gate": True,
        },
        "model_metrics": {
            "ticket_classifier_m6": {"category_macro_f1": 0.8501},
            "cost_router_m7": {"advantage_f1": 0.0997},
            "confidence_calibration_m7": {"brier_score": 0.0125},
        },
    }

    md = format_markdown_report(sample_report)
    assert "# Evaluation Harness Report" in md
    assert "EVAL-TEST-001" in md
    assert "Routine Resolution Rate" in md
    assert "Guardrail Safety Intercept" in md
    assert "Outcome Confusion Matrix" in md


def test_mini_run_evaluation(tmp_path: Path, monkeypatch):
    mini_dataset = [
        {
            "ticket_id": "MINI-001",
            "customer_id": "CUST-1001",
            "subject": "Where is my order ORD-5002?",
            "body": "Can you check on order ORD-5002 status?",
            "scenario": "routine",
            "expected_category": "request",
            "expected_severity": "low",
            "expected_outcome": "resolved",
            "expected_action": "order_tracking",
            "rationale": "Routine tracking request",
        },
        {
            "ticket_id": "MINI-002",
            "customer_id": "CUST-1001",
            "subject": "Refund for ORD-5001 $79.99",
            "body": "Item broken, refund $79.99 please.",
            "scenario": "guardrail_trigger",
            "expected_category": "request",
            "expected_severity": "high",
            "expected_outcome": "pending_approval",
            "expected_action": "refund",
            "rationale": "Over $50 limit",
        },
        {
            "ticket_id": "MINI-003",
            "customer_id": "CUST-1003",
            "subject": "Production database cluster corrupted",
            "body": "Severe hardware outage and disk loss.",
            "scenario": "edge_case",
            "expected_category": "problem",
            "expected_severity": "high",
            "expected_outcome": "escalated",
            "expected_action": "human_escalation",
            "rationale": "Critical outage",
        },
    ]

    dataset_file = tmp_path / "mini_dataset.json"
    dataset_file.write_text(json.dumps(mini_dataset), encoding="utf-8")

    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    # Set up in-memory sqlite DB for test isolation
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    test_factory = sessionmaker(bind=engine)

    from app.core.tracing import TraceWriter
    from app.orchestration import graph as orch

    monkeypatch.setattr("app.orchestration.graph.get_session_factory", lambda: test_factory)
    monkeypatch.setattr(orch, "_writer", lambda: TraceWriter(test_factory))
    monkeypatch.setattr(orch, "_record_episode", lambda t, r: "mock-point-id")

    # Seed the test DB with tickets
    with test_factory() as session:
        for item in mini_dataset:
            session.add(Ticket(
                id=item["ticket_id"],
                customer_id=item["customer_id"],
                subject=item["subject"],
                body=item["body"],
                status="open",
            ))
        session.commit()

    report = run_evaluation(
        dataset_path=dataset_file,
        output_json=out_json,
        output_md=out_md,
        mock_mode=True,
    )

    assert out_json.exists()
    assert out_md.exists()
    assert report["summary"]["total_tickets"] == 3
    assert report["summary"]["outcome_accuracy_pct"] == 100.0

    # Ensure output matches QualityOut schema expected by the dashboard route
    quality_out = QualityOut(available=True, report=report)
    assert quality_out.available is True
    assert quality_out.report["summary"]["total_tickets"] == 3
