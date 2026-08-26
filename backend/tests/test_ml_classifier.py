"""Tests for the ticket classifier (FR-1) — train/save/load/predict on synthetic data.

Hermetic and fast: a small templated corpus (~60 rows, the real label vocabularies
with disjoint keyword sets per category) trains a 20-tree model in well under a
second; no raw CSV and no network. Exercises bundle structure, the load/predict
roundtrip through the same serving path production uses, determinism, and error
handling.
"""

from __future__ import annotations

import pytest

from app.ml.classifier import (
    CATEGORY_LABELS,
    SEVERITY_LABELS,
    load_classifier,
    train_and_save,
)

# Disjoint keyword vocabularies per category so even a tiny model separates them.
# Every real label value gets one, keeping per-class metrics fully populated.
CLASS_VOCAB = {
    "incident": "outage crash down error failure alert",
    "problem": "bug defect chronic fault glitch reoccurring",
    "request": "license access install setup enable howto",
    "change": "upgrade migration rename policy configuration rollout",
}
FILLER = ("please", "help", "thanks", "asap", "today", "again")


def _synthetic_frame():
    """60 interleaved rows (4 categories x 3 severities), deterministic template text."""
    import pandas as pd

    rows = []
    for i in range(60):
        category = CATEGORY_LABELS[i % len(CATEGORY_LABELS)]
        severity = SEVERITY_LABELS[i % len(SEVERITY_LABELS)]
        keywords = CLASS_VOCAB[category]
        subject = f"{keywords.split()[i % 3]} {FILLER[i % len(FILLER)]}"
        body = f"{keywords} {FILLER[(i + 1) % len(FILLER)]} ticket number {i}"
        rows.append({"subject": subject, "body": body, "category": category, "severity": severity})
    # Interleaved build means any contiguous slice keeps all classes represented.
    frame = pd.DataFrame(rows)
    train = frame.iloc[:45].reset_index(drop=True)
    val = frame.iloc[45:].reset_index(drop=True)
    return train, val


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    models_dir = tmp_path_factory.mktemp("models")
    metrics = train_and_save(
        *_synthetic_frame(), models_dir, seed=7, n_estimators=20, max_depth=4
    )
    return models_dir, metrics


# --- training + bundle structure ------------------------------------------------------
def test_metrics_dict_structure(trained):
    _, metrics = trained
    assert {"category", "severity", "dataset", "seed"} <= set(metrics)
    assert metrics["seed"] == 7
    for target, labels in (("category", CATEGORY_LABELS), ("severity", SEVERITY_LABELS)):
        report = metrics[target]
        assert {"accuracy", "macro_f1", "per_class"} <= set(report)
        assert set(report["per_class"]) == set(labels)
        for stats in report["per_class"].values():
            assert {"precision", "recall", "f1", "support"} <= set(stats)
            assert 0.0 <= stats["precision"] <= 1.0
    assert metrics["dataset"]["train_rows"] == 45
    assert metrics["dataset"]["val_rows"] == 15


def test_tiny_model_separates_synthetic_categories(trained):
    """Sanity check: disjoint vocabularies should be learned near-perfectly."""
    _, metrics = trained
    assert metrics["category"]["macro_f1"] > 0.8


def test_train_and_save_rejects_missing_columns(tmp_path):
    import pandas as pd

    with pytest.raises(ValueError, match="required columns"):
        train_and_save(pd.DataFrame({"subject": ["a"]}), None, tmp_path)


# --- load / predict roundtrip -----------------------------------------------------------
def test_load_and_predict_roundtrip(trained):
    models_dir, _ = trained
    classifier = load_classifier(models_dir)
    prediction = classifier.predict("refund my purchase", "item arrived broken want money back")
    assert prediction.category in CATEGORY_LABELS
    assert prediction.severity in SEVERITY_LABELS
    assert 0.0 < prediction.category_confidence <= 1.0
    assert 0.0 < prediction.severity_confidence <= 1.0


def test_predict_is_deterministic_for_identical_input(trained):
    classifier = load_classifier(trained[0])
    first = classifier.predict("server error on login", "the app crashes every startup attempt")
    second = classifier.predict("server error on login", "the app crashes every startup attempt")
    assert first == second


def test_predict_rejects_non_string_inputs(trained):
    classifier = load_classifier(trained[0])
    with pytest.raises(TypeError, match="subject"):
        classifier.predict(None, "body")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="body"):
        classifier.predict("subject", 123)  # type: ignore[arg-type]


def test_load_classifier_missing_bundle_message(tmp_path):
    empty_dir = tmp_path / "nothing-here"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        load_classifier(empty_dir)
    message = str(excinfo.value)
    assert str(empty_dir) in message or str(empty_dir.resolve()) in message
    assert "ticket_classifier.joblib" in message
    assert "train_classifier.py" in message


def test_load_classifier_memoizes_per_directory(trained):
    # The lru_cache sits on the deserialized bundle: repeat calls share one copy of
    # the pipeline and both models instead of re-reading the joblib file.
    classifier_1 = load_classifier(trained[0])
    classifier_2 = load_classifier(trained[0])
    assert classifier_1.pipeline is classifier_2.pipeline
    assert classifier_1.category_model is classifier_2.category_model
    assert classifier_1.severity_model is classifier_2.severity_model
    assert classifier_2.predict("invoice question", "billing statement looks wrong").category == (
        classifier_1.predict("invoice question", "billing statement looks wrong").category
    )
