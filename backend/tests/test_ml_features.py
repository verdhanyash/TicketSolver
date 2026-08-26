"""Tests for ML feature construction (FR-1) — pure functions, no model training, no I/O.

Normalization is checked against every valid label plus case/whitespace variants;
the TF-IDF pipeline is exercised on a tiny corpus for shape, determinism, and
fit/transform consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.features import (
    VALID_CATEGORIES,
    VALID_SEVERITIES,
    build_feature_pipeline,
    combine_text,
    normalize_category,
    normalize_severity,
)


# --- normalize_category -------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Incident", "incident"),
        ("REQUEST", "request"),
        ("  Problem  ", "problem"),
        ("Change", "change"),
    ],
)
def test_normalize_category_maps_all_four_labels(raw: str, expected: str) -> None:
    assert normalize_category(raw) == expected


def test_normalize_category_rejects_unknown_and_lists_valid_labels() -> None:
    with pytest.raises(ValueError) as excinfo:
        normalize_category("Password reset")
    message = str(excinfo.value)
    assert "Password reset" in message
    for label in VALID_CATEGORIES:
        assert label in message


def test_normalize_category_is_idempotent() -> None:
    once = normalize_category(" Incident ")
    assert normalize_category(once) == once == "incident"


# --- normalize_severity ---------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("High", "high"),
        (" MEDIUM ", "medium"),
        ("low", "low"),
    ],
)
def test_normalize_severity_maps_all_three_labels(raw: str, expected: str) -> None:
    assert normalize_severity(raw) == expected


def test_normalize_severity_rejects_unknown_and_lists_valid_labels() -> None:
    with pytest.raises(ValueError) as excinfo:
        normalize_severity("Urgent")
    message = str(excinfo.value)
    assert "Urgent" in message
    for label in VALID_SEVERITIES:
        assert label in message


def test_normalize_severity_is_idempotent() -> None:
    once = normalize_severity(" High ")
    assert normalize_severity(once) == once == "high"


# --- combine_text ----------------------------------------------------------------------
def test_combine_text_is_deterministic_subject_first() -> None:
    text = combine_text("Login broken", "Cannot sign in since Tuesday")
    assert text == "Login broken\nCannot sign in since Tuesday"
    assert combine_text("Login broken", "Cannot sign in since Tuesday") == text
    # order matters: swapping inputs changes the output
    assert combine_text("Cannot sign in", "Login broken") != text


def test_combine_text_strips_outer_whitespace_only() -> None:
    assert combine_text("  Subject here ", "\n Body line\n ") == "Subject here\nBody line"


# --- build_feature_pipeline ------------------------------------------------------------
TINY_CORPUS = [
    "gateway timeout server error",
    "gateway server down again",
    "refund charged twice money",
    "refund money back charged",
    "server gateway error timeout",
    "charged twice refund please",
    "timeout gateway server error",
    "refund charged money back",
]


def test_pipeline_fit_transform_shape_and_determinism() -> None:
    pipeline = build_feature_pipeline()
    matrix_1 = pipeline.fit_transform(TINY_CORPUS)
    matrix_2 = pipeline.fit_transform(list(TINY_CORPUS))
    assert matrix_1.shape[0] == len(TINY_CORPUS)
    assert 0 < matrix_1.shape[1] <= 20_000
    assert np.array_equal(matrix_1.toarray(), matrix_2.toarray())


def test_pipeline_produces_bigrams_and_transform_matches_fit() -> None:
    pipeline = build_feature_pipeline()
    fitted = pipeline.fit_transform(TINY_CORPUS)
    names = list(pipeline.named_steps["tfidf"].get_feature_names_out())
    assert any(" " in name for name in names), "expected bigram features"
    transformed = pipeline.transform(TINY_CORPUS[:2]).toarray()
    assert np.array_equal(transformed, fitted.toarray()[:2])
