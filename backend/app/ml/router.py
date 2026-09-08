"""Complexity router (FR-14): pick the cheap vs. strong NIM tier per ticket.

Training happens offline via scripts/train_router.py, which calls `train_and_save`;
serving goes through `load_router`, which reads one joblib bundle containing both
heads — an XGBoost complexity classifier and a logistic-regression baseline (SRS 6.3)
kept in the bundle so the comparison stays reproducible.

Features are plain engineered numbers (lengths, punctuation, keyword counts, plus the
triage-predicted category/severity as vocabulary indices) built WITHOUT the TF-IDF
vectorizer: `build_router_features` is shared verbatim by training and serving so the
two paths cannot drift. The predicted labels are inputs, not oracle signals — the M6
severity head sits at macro-F1 ~0.59 (see Docs/CLASSIFIER_METRICS.md), which is exactly
why severity is one feature among several here.

Label caveat (documented, deliberate): training labels are BOOTSTRAP heuristics defined
by `bootstrap_complexity_label` (high predicted severity / long ticket / repeated
questions / problem-or-incident with a long body). They are deterministic functions of
the features, so held-out F1 near 1.0 is expected and is weak evidence by itself; the
informative part of the evaluation is the XGBoost-vs-logistic comparison and the class
balance. Once Module 8 accumulates real outcome labels from the eval harness, the same
pipeline retrains on those and the router becomes genuinely learned.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.config import settings
from app.ml.features import (
    CATEGORY_LABELS,
    SEVERITY_LABELS,
    combine_text,
    normalize_category,
    normalize_severity,
)

logger = logging.getLogger(__name__)

BUNDLE_NAME = "ticket_router.joblib"
BUNDLE_VERSION = 1
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"  # backend/models

# Human-readable names for the two complexity classes; index == model class id.
COMPLEXITY_LABELS = ("simple", "complex")

# Categories whose tickets tend to need deeper investigation when they run long.
COMPLEX_CATEGORIES = ("problem", "incident")

# Keyword lexicons counted into the feature vector (substring-safe word matches).
URGENCY_KEYWORDS = (
    "urgent", "urgently", "asap", "immediately", "critical", "emergency",
    "outage", "down", "broken", "failure", "failed", "blocker", "crash",
)
POLITENESS_KEYWORDS = ("please", "thanks", "thank you", "appreciate", "kindly", "sorry")

# Feature order IS the model input order; export for docs and future importance charts.
FEATURE_NAMES = (
    "subject_chars",
    "subject_words",
    "body_chars",
    "body_words",
    "question_marks",
    "urgency_keyword_count",
    "politeness_keyword_count",
    "subject_upper_ratio",
    "digit_count",
    "category_index",
    "severity_index",
)

UNKNOWN_LABEL_INDEX = -1.0  # feature value when no triage prediction is available

LABELING_RULE_DOC = (
    "complexity=1 when ANY hold: ground-truth severity == 'high'; total subject+body "
    "word count above the train-split quantile; >=2 question marks in subject+body; or "
    "category in (problem, incident) with body word count above the train body-word "
    "quantile. complexity=0 otherwise. Bootstrap heuristic until real outcome labels "
    "accumulate from the eval harness (Module 8)."
)


# --- Pure feature + label builders (shared verbatim by training and serving) --------
def _keyword_count(text: str, keywords: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(kw)}\b", lowered)) for kw in keywords)


def _label_index(vocabulary: tuple[str, ...], value: str) -> float:
    return float(vocabulary.index(value)) if value in vocabulary else UNKNOWN_LABEL_INDEX


def build_router_features(
    subject: str, body: str, category: str = "", severity: str = ""
) -> list[float]:
    """Engineered numeric features for one ticket, in FEATURE_NAMES order.

    `category` / `severity` are the TRIAGE-PREDICTED labels at serving time (what the
    FR-1 classifier wrote onto the ticket) and normalized dataset labels during
    training. Unknown or missing labels encode as UNKNOWN_LABEL_INDEX rather than
    raising: routing must survive a skipped triage.
    """
    if not isinstance(subject, str):
        raise TypeError(f"subject must be a string, got {type(subject).__name__}")
    if not isinstance(body, str):
        raise TypeError(f"body must be a string, got {type(body).__name__}")
    clean_subject = subject.strip()
    clean_body = body.strip()
    combined = f"{clean_subject}\n{clean_body}"
    alpha_subject = [c for c in clean_subject if c.isalpha()]
    upper_ratio = (
        sum(c.isupper() for c in alpha_subject) / len(alpha_subject) if alpha_subject else 0.0
    )
    return [
        float(len(clean_subject)),
        float(len(clean_subject.split())),
        float(len(clean_body)),
        float(len(clean_body.split())),
        float(combined.count("?")),
        float(_keyword_count(combined, URGENCY_KEYWORDS)),
        float(_keyword_count(combined, POLITENESS_KEYWORDS)),
        round(upper_ratio, 6),
        float(sum(c.isdigit() for c in combined)),
        _label_index(CATEGORY_LABELS, category),
        _label_index(SEVERITY_LABELS, severity),
    ]


def bootstrap_complexity_thresholds(frame: pd.DataFrame, *, word_quantile: float = 0.75) -> dict:
    """Fit the word-count cut-offs used by the bootstrap labeling rule.

    Computed on the TRAIN split only and frozen into the bundle metadata, so label
    regeneration and documentation always see the same constants.
    """
    if frame.empty:
        raise ValueError("cannot fit complexity thresholds on an empty frame")
    total_words = [
        len(str(s).split()) + len(str(b).split())
        for s, b in zip(frame["subject"], frame["body"], strict=True)
    ]
    body_words = [len(str(b).split()) for b in frame["body"]]
    return {
        "total_word_threshold": float(np.quantile(total_words, word_quantile)),
        "body_word_threshold": float(np.quantile(body_words, word_quantile)),
        "word_quantile": float(word_quantile),
    }


def bootstrap_complexity_label(
    *, severity: str, category: str, subject: str, body: str, thresholds: dict
) -> int:
    """Bootstrap complexity target; see LABELING_RULE_DOC for the exact rule.

    Deliberately transparent: today the router learns to reproduce this heuristic, and
    the function doubles as executable documentation of what "complex" currently means.
    """
    total_words = len(str(subject).strip().split()) + len(str(body).strip().split())
    body_words = len(str(body).strip().split())
    question_marks = f"{str(subject).strip()}\n{str(body).strip()}".count("?")
    if severity == "high":
        return 1
    if total_words > thresholds["total_word_threshold"]:
        return 1
    if question_marks >= 2:
        return 1
    if category in COMPLEX_CATEGORIES and body_words > thresholds["body_word_threshold"]:
        return 1
    return 0


def select_model_tier(
    complexity_label: int, *, cheap_name: str, strong_name: str
) -> tuple[bool, str | None, str]:
    """Map a complexity label to (use_strong, selected_model, rationale).

    Missing tier names degrade toward strong — the conservative direction for quality —
    and the rationale always says which tier was unconfigured so the trace explains any
    fallback. With neither name configured nothing can be selected (None) but the
    decision still records that the strong tier was wanted.
    """
    if complexity_label == 1:
        if strong_name:
            return True, strong_name, f"complexity=1 routes to the strong tier ({strong_name})"
        if cheap_name:
            return True, None, (
                "complexity=1 but NIM_MODEL_STRONG is unconfigured; no strong-tier model "
                "name available"
            )
        return True, None, (
            "complexity=1 but both NIM_MODEL_STRONG and NIM_MODEL_CHEAP are unconfigured; "
            "defaulting to the strong tier"
        )
    if cheap_name:
        return False, cheap_name, f"complexity=0 routes to the cheap tier ({cheap_name})"
    if strong_name:
        return True, strong_name, (
            "complexity=0 but NIM_MODEL_CHEAP is unconfigured; defaulting to the strong tier"
        )
    return True, None, (
        "complexity=0 but both NIM tiers are unconfigured; defaulting to the strong tier"
    )


@dataclass(frozen=True)
class RouterDecision:
    """One FR-14 serving decision: tier choice plus the evidence behind it."""

    use_strong: bool
    complexity_label: int
    confidence: float
    rationale: str
    selected_model: str | None


# --- Inference --------------------------------------------------------------------
class TicketRouter:
    """Inference wrapper around a trained bundle (see load_router)."""

    def __init__(self, bundle: dict) -> None:
        missing = {"xgb_model", "lr_model"} - set(bundle)
        if missing:
            raise ValueError(f"invalid router bundle; missing keys: {sorted(missing)}")
        self.xgb_model = bundle["xgb_model"]
        self.lr_model = bundle["lr_model"]
        self.metadata: dict = dict(bundle.get("metadata", {}))

    def predict_complexity(
        self, subject: str, body: str, category: str = "", severity: str = ""
    ) -> tuple[int, float]:
        """XGBoost head: (complexity label, max class probability)."""
        features = np.asarray([build_router_features(subject, body, category, severity)])
        proba = np.asarray(self.xgb_model.predict_proba(features)[0])
        classes = [int(c) for c in self.xgb_model.classes_]
        winner = classes[int(np.argmax(proba))]
        return winner, round(float(proba.max()), 4)

    def route(
        self, subject: str, body: str, category: str = "", severity: str = ""
    ) -> RouterDecision:
        """Classify complexity and map it to the configured NIM tier.

        Model names are read from settings at call time so configuration changes apply
        without retraining; empty names fall back per select_model_tier.
        """
        label, confidence = self.predict_complexity(subject, body, category, severity)
        use_strong, selected_model, rationale = select_model_tier(
            label,
            cheap_name=settings.nim_model_cheap,
            strong_name=settings.nim_model_strong,
        )
        return RouterDecision(
            use_strong=use_strong,
            complexity_label=label,
            confidence=confidence,
            rationale=rationale,
            selected_model=selected_model,
        )


# --- Loading ----------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_bundle(models_dir: str) -> dict:
    path = Path(models_dir) / BUNDLE_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No router bundle at {path} (expected file '{BUNDLE_NAME}'). "
            "Train one first: python scripts/train_router.py"
        )
    bundle = joblib.load(path)
    version = bundle.get("version")
    if version is not None and version != BUNDLE_VERSION:
        raise ValueError(f"unsupported router bundle version {version!r} in {path}")
    logger.info("loaded router bundle %s (trained_at=%s)",
                path, bundle.get("metadata", {}).get("trained_at", "unknown"))
    return bundle


def clear_router_cache() -> None:
    """Drop memoized bundles (used after retraining into an already-loaded directory)."""
    _load_bundle.cache_clear()


def load_router(models_dir: str | Path | None = None) -> TicketRouter:
    """Load the trained bundle from `models_dir` (default backend/models).

    Memoized per resolved directory: repeat calls reuse the same deserialized objects.
    Raises FileNotFoundError naming the expected path when no bundle exists.
    """
    directory = Path(models_dir).resolve() if models_dir is not None else DEFAULT_MODELS_DIR
    return TicketRouter(_load_bundle(str(directory)))


# --- Training ----------------------------------------------------------------------
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    indices = [0, 1]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    per_class = {
        COMPLEXITY_LABELS[label]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, label in enumerate(indices)
    }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=indices, average="macro",
                                         zero_division=0)), 4),
        "per_class": per_class,
    }


def _feature_inputs(frame: pd.DataFrame, classifier) -> tuple[list[str], list[str]]:
    """Per-row (category, severity) feature inputs.

    With a classifier bundle these are its PREDICTIONS — matching serving, where the
    router consumes whatever FR-1 triage wrote onto the ticket. Batched transform +
    predict produces exactly the labels per-row predict() would (same fitted objects,
    argmax either way); it is only faster. Without a classifier (tests, tiny corpora)
    the row's own labels stand in.
    """
    if classifier is None:
        return list(frame["category"]), list(frame["severity"])
    texts = [
        combine_text(str(subject), str(body))
        for subject, body in zip(frame["subject"], frame["body"], strict=True)
    ]
    transformed = classifier.pipeline.transform(texts)
    categories = [CATEGORY_LABELS[i] for i in classifier.category_model.predict(transformed)]
    severities = [SEVERITY_LABELS[i] for i in classifier.severity_model.predict(transformed)]
    return categories, severities


def train_and_save(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    models_dir: str | Path,
    *,
    classifier=None,
    seed: int = 42,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    word_quantile: float = 0.75,
) -> dict:
    """Fit the complexity head and the LR baseline on `train_df`, save one bundle.

    Labels come from `bootstrap_complexity_label` against GROUND-TRUTH severity/category
    (the rule defines what complexity should be), while the FEATURES carry the
    classifier's predictions like production does — the label noise that mismatch
    introduces is real and kept honestly. Thresholds are fitted on the train split only;
    metrics reflect `val_df` when given. Returns per-model reports plus dataset/threshold
    info.
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
    thresholds = bootstrap_complexity_thresholds(train, word_quantile=word_quantile)

    def design_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        categories, severities = _feature_inputs(frame, classifier)
        matrix = np.asarray(
            [
                build_router_features(str(subject), str(body), category, severity)
                for subject, body, category, severity in zip(
                    frame["subject"], frame["body"], categories, severities, strict=True
                )
            ]
        )
        targets = np.asarray(
            [
                bootstrap_complexity_label(
                    severity=sev,
                    category=cat,
                    subject=str(subject),
                    body=str(body),
                    thresholds=thresholds,
                )
                for sev, cat, subject, body in zip(
                    frame["severity"],
                    frame["category"],
                    frame["subject"],
                    frame["body"],
                    strict=True,
                )
            ],
            dtype=np.int64,
        )
        return matrix, targets

    x_train, y_train = design_matrix(train)
    logger.info(
        "training router heads on %d rows (%d features, %d complex)",
        len(train), x_train.shape[1], int(y_train.sum()),
    )

    xgb_model = XGBClassifier(
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    xgb_model.fit(x_train, y_train)
    lr_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    lr_model.fit(x_train, y_train)

    eval_frame = val if val is not None else train
    evaluated_on = "validation" if val is not None else "train"
    x_eval, y_eval = design_matrix(eval_frame)
    metrics = {
        "xgboost": _metrics(y_eval, xgb_model.predict(x_eval)),
        "logistic_regression": _metrics(y_eval, lr_model.predict(x_eval)),
        "dataset": {
            "train_rows": len(train),
            "val_rows": len(val) if val is not None else 0,
            "evaluated_on": evaluated_on,
            "train_complex_fraction": round(float(y_train.mean()), 4),
        },
        "thresholds": thresholds,
        "seed": seed,
    }

    metadata = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "tree_method": "hist",
        "feature_names": list(FEATURE_NAMES),
        "labeling_rule": LABELING_RULE_DOC,
        "thresholds": thresholds,
        "label_source": "classifier-predictions" if classifier is not None else "row-labels",
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
            "xgb_model": xgb_model,
            "lr_model": lr_model,
            "metadata": metadata,
        },
        bundle_path,
    )
    clear_router_cache()  # any previously memoized load of this dir is now stale
    logger.info("saved router bundle to %s", bundle_path)
    return metrics
