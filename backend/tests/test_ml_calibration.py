"""Unit tests for the confidence calibration model (FR-15, Module 7) — hermetic, no network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.calibration import (
    CONFIDENCE_BAR,
    FEATURE_NAMES,
    THRESHOLD_CEILING,
    THRESHOLD_FLOOR,
    bootstrap_decision_label,
    build_calibration_features,
    calibrated_threshold,
    load_calibration,
    reliability_bins,
    train_and_save,
)


def test_build_calibration_features():
    features = build_calibration_features(
        attempt=1,
        proposal="Issue refund of $20.00.",
        confidence=0.85,
        tool_calls=2,
        tool_errored=False,
    )
    assert len(features) == len(FEATURE_NAMES)
    assert features == [1.0, float(len("Issue refund of $20.00.")), 0.85, 2.0, 0.0]

    # Type check validation
    with pytest.raises(TypeError, match="proposal must be a string"):
        build_calibration_features(attempt=1, proposal=None, confidence=0.5)  # type: ignore[arg-type]


def test_bootstrap_decision_label():
    # Attempt 1, no error, high confidence -> 1
    assert bootstrap_decision_label(attempt=1, confidence=CONFIDENCE_BAR, tool_errored=False) == 1
    assert bootstrap_decision_label(attempt=1, confidence=0.95, tool_errored=False) == 1

    # Low confidence -> 0
    assert bootstrap_decision_label(attempt=1, confidence=0.79, tool_errored=False) == 0

    # Retry attempt > 1 -> 0
    assert bootstrap_decision_label(attempt=2, confidence=0.95, tool_errored=False) == 0

    # Tool errored -> 0
    assert bootstrap_decision_label(attempt=1, confidence=0.95, tool_errored=True) == 0


def test_calibrated_threshold_mapping_and_clamping():
    base = 0.60

    # P(correct) = 0.50 -> base threshold unchanged
    assert calibrated_threshold(0.50, base) == 0.60

    # P(correct) = 0.90 -> threshold lowered (higher trust in agent)
    # 0.60 + 0.5 * (0.5 - 0.9) = 0.60 - 0.20 = 0.40
    assert calibrated_threshold(0.90, base) == 0.40

    # P(correct) = 0.10 -> threshold raised (low trust in agent)
    # 0.60 + 0.5 * (0.5 - 0.1) = 0.60 + 0.20 = 0.80
    assert calibrated_threshold(0.10, base) == 0.80

    # Clamping at floor
    assert calibrated_threshold(1.0, 0.35) >= THRESHOLD_FLOOR
    assert calibrated_threshold(1.0, 0.10) == THRESHOLD_FLOOR

    # Clamping at ceiling
    assert calibrated_threshold(0.0, 0.85) <= THRESHOLD_CEILING
    assert calibrated_threshold(0.0, 0.95) == THRESHOLD_CEILING


def test_reliability_bins():
    y_true = np.array([0, 0, 1, 1])
    p_pred = np.array([0.2, 0.4, 0.8, 0.95])
    bins = reliability_bins(y_true, p_pred)
    assert len(bins) > 0
    assert all("range" in b and "n" in b for b in bins)


def test_calibration_train_load_predict_roundtrip(tmp_path: Path):
    df = pd.DataFrame(
        {
            "attempt": [1, 1, 2, 1, 2, 1] * 20,
            "proposal": [
                "Issue refund of $20.00",
                "Reset user password",
                "Reship delayed kit",
                "Send manual guide",
                "Escalate to billing",
                "Close resolved ticket",
            ]
            * 20,
            "confidence": [0.9, 0.85, 0.5, 0.95, 0.4, 0.8] * 20,
            "tool_calls": [1, 0, 2, 1, 3, 1] * 20,
            "tool_errored": [0, 0, 1, 0, 1, 0] * 20,
        }
    )

    metrics = train_and_save(df, None, tmp_path, seed=42)
    assert "accuracy" in metrics

    calibrator = load_calibration(tmp_path)
    p = calibrator.predict_probability(
        attempt=1, proposal="Issue refund", confidence=0.9, tool_calls=1, tool_errored=False
    )
    assert 0.0 <= p <= 1.0


def test_load_calibration_missing_bundle(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No calibration bundle"):
        load_calibration(tmp_path)
