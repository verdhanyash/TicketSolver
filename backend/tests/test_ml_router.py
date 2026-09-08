"""Unit tests for the ML complexity router (FR-14, Module 7) — hermetic, no network."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.config import settings
from app.ml.router import (
    FEATURE_NAMES,
    UNKNOWN_LABEL_INDEX,
    bootstrap_complexity_label,
    build_router_features,
    load_router,
    select_model_tier,
    train_and_save,
)


def test_build_router_features_shape_and_types():
    features = build_router_features(
        subject="Urgent server crash!",
        body="Our system is completely down, please help immediately! What should we do?",
        category="incident",
        severity="high",
    )
    assert len(features) == len(FEATURE_NAMES)
    assert all(isinstance(f, float) for f in features)

    # Question mark count in body is 1
    q_idx = FEATURE_NAMES.index("question_marks")
    assert features[q_idx] == 1.0

    # Urgency keyword count: "urgent", "crash", "down", "immediately" -> >= 4
    urg_idx = FEATURE_NAMES.index("urgency_keyword_count")
    assert features[urg_idx] >= 4.0


def test_build_router_features_unknown_labels():
    features = build_router_features("sub", "body", category="unknown_cat", severity="unknown_sev")
    cat_idx = FEATURE_NAMES.index("category_index")
    sev_idx = FEATURE_NAMES.index("severity_index")
    assert features[cat_idx] == UNKNOWN_LABEL_INDEX
    assert features[sev_idx] == UNKNOWN_LABEL_INDEX


def test_build_router_features_type_validation():
    with pytest.raises(TypeError, match="subject must be a string"):
        build_router_features(123, "body")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="body must be a string"):
        build_router_features("sub", None)  # type: ignore[arg-type]


def test_select_model_tier():
    use_strong, model, rationale = select_model_tier(
        0, cheap_name="cheap-llm", strong_name="strong-llm"
    )
    assert not use_strong
    assert model == "cheap-llm"
    assert "cheap" in rationale

    use_strong, model, rationale = select_model_tier(
        1, cheap_name="cheap-llm", strong_name="strong-llm"
    )
    assert use_strong
    assert model == "strong-llm"
    assert "strong" in rationale


def test_bootstrap_complexity_label_rules():
    thresholds = {"total_word_threshold": 50, "body_word_threshold": 40}

    # Rule 1: high severity is always complex
    assert (
        bootstrap_complexity_label(
            severity="high", category="request", subject="hi", body="help", thresholds=thresholds
        )
        == 1
    )

    # Rule 2: >= 2 question marks is complex
    assert (
        bootstrap_complexity_label(
            severity="low", category="request", subject="hi?", body="help?", thresholds=thresholds
        )
        == 1
    )

    # Rule 3: long body in problem/incident is complex
    long_body = "word " * 45
    assert (
        bootstrap_complexity_label(
            severity="low",
            category="problem",
            subject="bug",
            body=long_body,
            thresholds=thresholds,
        )
        == 1
    )

    # Simple ticket: low severity request with short text
    assert (
        bootstrap_complexity_label(
            severity="low",
            category="request",
            subject="info",
            body="simple question",
            thresholds=thresholds,
        )
        == 0
    )


def test_router_train_load_predict_roundtrip(tmp_path: Path):
    train_data = pd.DataFrame(
        {
            "subject": [
                "urgent outage crash",
                "quick question",
                "broken server error down",
                "billing info please",
                "critical database failure",
                "password reset",
            ]
            * 10,
            "body": [
                "Everything is down immediately! Can you help? Is there a backup?",
                "How do I update my name?",
                "Error 500 across all nodes! Urgent!",
                "Where can I find my invoice?",
                "Database corrupt and failing! Need emergency assist!",
                "Forgot password.",
            ]
            * 10,
            "category": ["incident", "request", "incident", "request", "problem", "request"] * 10,
            "severity": ["high", "low", "high", "low", "high", "low"] * 10,
        }
    )

    metrics = train_and_save(train_data, None, tmp_path, seed=42, n_estimators=20, max_depth=3)
    assert "xgboost" in metrics
    assert "logistic_regression" in metrics

    # Load bundle from temp dir
    router = load_router(tmp_path)
    label, conf = router.predict_complexity("urgent outage", "server is down!")
    assert label in (0, 1)
    assert 0.0 <= conf <= 1.0

    decision = router.route("urgent outage", "server is down!")
    assert decision.complexity_label in (0, 1)
    assert decision.selected_model in (settings.nim_model_cheap, settings.nim_model_strong)


def test_load_router_missing_bundle(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No router bundle"):
        load_router(tmp_path)
