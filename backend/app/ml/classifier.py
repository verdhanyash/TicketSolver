"""Ticket category/severity classifier (FR-1) — TF-IDF features + two XGBoost heads.

Training happens offline via scripts/train_classifier.py, which calls `train_and_save`;
serving goes through `load_classifier`, which reads one joblib bundle containing the
fitted feature pipeline plus both models — inference therefore always uses exactly the
preprocessing the models were trained against.

Labels are encoded as integer indices into CATEGORY_LABELS / SEVERITY_LABELS before
fitting, so `predict_proba` column order is explicit and stable across xgboost
versions. Reported confidence is the max class probability; FR-15 calibration arrives
in a later module and will layer on top of these raw probabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.ml.features import (
    VALID_CATEGORIES,
    VALID_SEVERITIES,
    build_feature_pipeline,
    combine_text,
    normalize_category,
    normalize_severity,
)

logger = logging.getLogger(__name__)

CATEGORY_LABELS = VALID_CATEGORIES  # alphabetical; index == model class id
SEVERITY_LABELS = VALID_SEVERITIES  # ordinal: low < medium < high < critical

BUNDLE_NAME = "ticket_classifier.joblib"
BUNDLE_VERSION = 1
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"  # backend/models


@dataclass(frozen=True)
class TicketPrediction:
    """Argmax label pair with raw max-probability confidences (4 decimal places)."""

    category: str
    severity: str
    category_confidence: float
    severity_confidence: float


# --- Inference -------------------------------------------------------------------
class TicketClassifier:
    """Inference wrapper around a trained bundle (see load_classifier)."""

    def __init__(self, bundle: dict) -> None:
        missing = {"pipeline", "category_model", "severity_model"} - set(bundle)
        if missing:
            raise ValueError(f"invalid classifier bundle; missing keys: {sorted(missing)}")
        self.pipeline: Pipeline = bundle["pipeline"]
        self.category_model = bundle["category_model"]
        self.severity_model = bundle["severity_model"]
        self.metadata: dict = dict(bundle.get("metadata", {}))
        for name, model, labels in (
            ("category_model", self.category_model, CATEGORY_LABELS),
            ("severity_model", self.severity_model, SEVERITY_LABELS),
        ):
            classes = int(getattr(model, "n_classes_", len(labels)))
            if classes != len(labels):
                raise ValueError(
                    f"{name} predicts {classes} classes but {len(labels)} labels are expected"
                )

    def predict(self, subject: str, body: str) -> TicketPrediction:
        """Classify one ticket; deterministic for identical inputs."""
        if not isinstance(subject, str):
            raise TypeError(f"subject must be a string, got {type(subject).__name__}")
        if not isinstance(body, str):
            raise TypeError(f"body must be a string, got {type(body).__name__}")
        features = self.pipeline.transform([combine_text(subject, body)])
        cat_proba = np.asarray(self.category_model.predict_proba(features)[0])
        sev_proba = np.asarray(self.severity_model.predict_proba(features)[0])
        cat_idx = int(np.argmax(cat_proba))
        sev_idx = int(np.argmax(sev_proba))
        return TicketPrediction(
            category=CATEGORY_LABELS[cat_idx],
            severity=SEVERITY_LABELS[sev_idx],
            category_confidence=round(float(cat_proba[cat_idx]), 4),
            severity_confidence=round(float(sev_proba[sev_idx]), 4),
        )


# --- Loading ----------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_bundle(models_dir: str) -> dict:
    path = Path(models_dir) / BUNDLE_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No classifier bundle at {path} (expected file '{BUNDLE_NAME}'). "
            "Train one first: python scripts/train_classifier.py"
        )
    bundle = joblib.load(path)
    version = bundle.get("version")
    if version is not None and version != BUNDLE_VERSION:
        raise ValueError(f"unsupported classifier bundle version {version!r} in {path}")
    logger.info("loaded classifier bundle %s (trained_at=%s)",
                path, bundle.get("metadata", {}).get("trained_at", "unknown"))
    return bundle


def clear_classifier_cache() -> None:
    """Drop memoized bundles (used after retraining into an already-loaded directory)."""
    _load_bundle.cache_clear()


def load_classifier(models_dir: str | Path | None = None) -> TicketClassifier:
    """Load the trained bundle from `models_dir` (default backend/models).

    Memoized per resolved directory: repeat calls reuse the same deserialized objects.
    Raises FileNotFoundError naming the expected path when no bundle exists.
    """
    directory = Path(models_dir).resolve() if models_dir is not None else DEFAULT_MODELS_DIR
    return TicketClassifier(_load_bundle(str(directory)))


# --- Training ---------------------------------------------------------------------
def _encode(frame: pd.DataFrame, column: str, labels: tuple[str, ...]) -> np.ndarray:
    return np.array([labels.index(v) for v in frame[column]], dtype=np.int64)


def _texts(frame: pd.DataFrame) -> list[str]:
    for column in ("subject", "body"):
        nulls = int(frame[column].isna().sum())
        if nulls:
            raise ValueError(f"frame has {nulls} null value(s) in required column '{column}'")
    return [
        combine_text(str(subject), str(body))
        for subject, body in zip(frame["subject"], frame["body"], strict=True)
    ]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...]) -> dict:
    indices = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=indices, average="macro",
                                         zero_division=0)), 4),
        "per_class": per_class,
    }


def train_and_save(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    models_dir: str | Path,
    *,
    seed: int = 42,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.2,
) -> dict:
    """Fit both XGBoost heads on `train_df` and save one joblib bundle.

    The feature pipeline is fitted on the training split only (no leakage); `val_df`,
    when given, is used for the returned metrics so they reflect unseen rows. Returns
    {"category": {...}, "severity": {...}, "dataset": {...}, "seed": ...} where each
    target holds accuracy, macro_f1, and per-class precision/recall/f1/support.
    """
    frames: list[tuple[str, pd.DataFrame]] = [("train_df", train_df)]
    if val_df is not None:
        frames.append(("val_df", val_df))
    required = {"subject", "body", "category", "severity"}
    for name, frame in frames:
        absent = required - set(frame.columns)
        if absent:
            raise ValueError(f"{name} is missing required columns: {sorted(absent)}")

    def prepared(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["category"] = [normalize_category(v) for v in out["category"]]
        out["severity"] = [normalize_severity(v) for v in out["severity"]]
        return out

    train = prepared(train_df)
    val = prepared(val_df) if val_df is not None else None

    pipeline = build_feature_pipeline()
    x_train = pipeline.fit_transform(_texts(train))
    y_cat_train = _encode(train, "category", CATEGORY_LABELS)
    y_sev_train = _encode(train, "severity", SEVERITY_LABELS)

    def new_model() -> XGBClassifier:
        return XGBClassifier(
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )

    logger.info("training category head on %d rows (%d features)", len(train), x_train.shape[1])
    category_model = new_model()
    category_model.fit(x_train, y_cat_train)
    logger.info("training severity head on %d rows", len(train))
    severity_model = new_model()
    severity_model.fit(x_train, y_sev_train)

    # Metrics on held-out validation when available; training rows otherwise.
    eval_frame = val if val is not None else train
    evaluated_on = "validation" if val is not None else "train"
    x_eval = pipeline.transform(_texts(eval_frame))
    metrics = {
        "category": _metrics(
            _encode(eval_frame, "category", CATEGORY_LABELS), category_model.predict(x_eval),
            CATEGORY_LABELS,
        ),
        "severity": _metrics(
            _encode(eval_frame, "severity", SEVERITY_LABELS), severity_model.predict(x_eval),
            SEVERITY_LABELS,
        ),
        "dataset": {
            "train_rows": len(train),
            "val_rows": len(val) if val is not None else 0,
            "evaluated_on": evaluated_on,
        },
        "seed": seed,
    }

    metadata = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "train_rows": metrics["dataset"]["train_rows"],
        "val_rows": metrics["dataset"]["val_rows"],
        "evaluated_on": evaluated_on,
        "category_labels": list(CATEGORY_LABELS),
        "severity_labels": list(SEVERITY_LABELS),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
    }
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / BUNDLE_NAME
    joblib.dump(
        {
            "version": BUNDLE_VERSION,
            "pipeline": pipeline,
            "category_model": category_model,
            "severity_model": severity_model,
            "metadata": metadata,
        },
        bundle_path,
    )
    clear_classifier_cache()  # any previously memoized load of this dir is now stale
    logger.info("saved classifier bundle to %s", bundle_path)
    return metrics
