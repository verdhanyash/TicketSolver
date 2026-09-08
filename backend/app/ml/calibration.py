"""Confidence calibration (FR-15): P(decision correct) from agent-run observables.

The investigator's CONFIDENCE line is self-reported; this module learns to map cheap,
decision-time observables — retry count, proposal length, that self-reported value,
tool-call volume, tool failures — to a probability that the current decision is
actually correct. A logistic regression is deliberately enough: five features, demo
scale, and every coefficient stays inspectable.

Label caveat (documented, deliberate): training labels are BOOTSTRAP heuristics from
`bootstrap_decision_label` (first attempt, no tool errors, self-confidence at/above the
bar counts as "correct"). The model therefore reproduces that policy today and becomes
genuinely learned once Module 8 accumulates real resolve-vs-escalate outcomes.

Serving contract: `load_calibration()` mirrors the classifier/router loaders
(memoized joblib bundle, FileNotFoundError naming the remedy). The orchestrator's
`_calibrated_threshold` seam consumes it best-effort — ANY failure means "fall back to
settings.confidence_threshold", never an orchestration break.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

BUNDLE_NAME = "confidence_calibration.joblib"
BUNDLE_VERSION = 1
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"  # backend/models

# Feature order IS the model input order.
FEATURE_NAMES = (
    "attempt",
    "proposal_length",
    "confidence",
    "tool_calls",
    "tool_errored",
)

# Bootstrap labeling rule constants + threshold mapping constants (all documented).
CONFIDENCE_BAR = 0.8  # self-reported confidence at/above this counts as "correct"
THRESHOLD_FLOOR = 0.30  # calibrated bar never drops below this...
THRESHOLD_CEILING = 0.90  # ...and never rises above this
TRUST_SENSITIVITY = 0.5  # how strongly P(correct) moves the base threshold

LABELING_RULE_DOC = (
    f"label=1 ('decision correct') when attempt == 1 AND no tool errors AND "
    f"self-reported confidence >= {CONFIDENCE_BAR:.2f}; label=0 on retries, tool errors, or low "
    "confidence. Bootstrap heuristic until real eval-harness outcomes accumulate "
    "(Module 8)."
)


# --- Pure feature + label builders (shared verbatim by training and serving) --------
def build_calibration_features(
    *,
    attempt: int,
    proposal: str,
    confidence: float,
    tool_calls: int = 0,
    tool_errored: bool | int = False,
) -> list[float]:
    """Numeric features for one agent run, in FEATURE_NAMES order.

    `proposal` is the raw proposal text; its stripped character length becomes the
    length feature so training and serving measure exactly the same thing. Tool-call
    features default to a clean run for callers (like the orchestrator today) that do
    not yet plumb per-run tool telemetry into decision time.
    """
    if not isinstance(proposal, str):
        raise TypeError(f"proposal must be a string, got {type(proposal).__name__}")
    return [
        float(int(attempt)),
        float(len(proposal.strip())),
        float(confidence),
        float(int(tool_calls)),
        float(1 if tool_errored else 0),
    ]


def bootstrap_decision_label(
    *, attempt: int, confidence: float, tool_errored: bool | int = False
) -> int:
    """Bootstrap correctness target; see LABELING_RULE_DOC for the exact rule."""
    correct = int(attempt) == 1 and not tool_errored and float(confidence) >= CONFIDENCE_BAR
    return 1 if correct else 0


def calibrated_threshold(p_correct: float, base_threshold: float) -> float:
    """Map P(decision correct) to the escalation threshold decide_next compares against.

    Monotone and bounded: more trust in the decision lowers the bar, mistrust raises it,
    and the floor/ceiling keep either safety net reachable — a near-certain model can
    not wave everything through, a panicked one can not escalate everything. At
    p_correct = 0.5 the base threshold passes through unchanged.
    """
    p = min(max(float(p_correct), 0.0), 1.0)
    adjusted = float(base_threshold) + TRUST_SENSITIVITY * (0.5 - p)
    return round(min(max(adjusted, THRESHOLD_FLOOR), THRESHOLD_CEILING), 4)


def reliability_bins(
    y_true: np.ndarray | list[int],
    p_predicted: np.ndarray | list[float],
    bin_edges: tuple[float, ...] = (0.0, 0.6, 0.7, 0.8, 0.9, 1.0001),
) -> list[dict]:
    """Reliability-style check: does predicted P(correct) rank actual correctness?

    Rows are grouped by predicted probability into [lo, hi) bins (last bin closed), and
    each bin reports mean predicted probability vs observed correct rate. Empty bins are
    reported with None rates rather than dropped, so the table shows coverage honestly.
    """
    edges = list(bin_edges)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        closing = "]" if i == len(edges) - 2 else ")"
        members = [
            (float(p), int(y))
            for p, y in zip(p_predicted, y_true, strict=True)
            if lo <= float(p) < hi
        ]
        n = len(members)
        rows.append(
            {
                "range": f"[{lo:.2f},{hi:.2f}{closing}",
                "n": n,
                "mean_predicted": round(float(np.mean([p for p, _ in members])), 4) if n else None,
                "observed_rate": round(float(np.mean([y for _, y in members])), 4) if n else None,
            }
        )
    return rows


# --- Inference --------------------------------------------------------------------
class ConfidenceCalibrator:
    """Inference wrapper around a trained bundle (see load_calibration)."""

    def __init__(self, bundle: dict) -> None:
        missing = {"model"} - set(bundle)
        if missing:
            raise ValueError(f"invalid calibration bundle; missing keys: {sorted(missing)}")
        self.model = bundle["model"]
        self.metadata: dict = dict(bundle.get("metadata", {}))

    def predict_probability(
        self,
        *,
        attempt: int,
        proposal: str,
        confidence: float,
        tool_calls: int = 0,
        tool_errored: bool | int = False,
    ) -> float:
        """P(current decision correct) in [0, 1]; deterministic for identical inputs."""
        features = np.asarray(
            [
                build_calibration_features(
                    attempt=attempt,
                    proposal=proposal,
                    confidence=confidence,
                    tool_calls=tool_calls,
                    tool_errored=tool_errored,
                )
            ]
        )
        proba = np.asarray(self.model.predict_proba(features)[0])
        classes = [int(c) for c in self.model.classes_]
        positive = classes.index(1) if 1 in classes else len(classes) - 1
        return round(float(proba[positive]), 4)


# --- Loading ----------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_bundle(models_dir: str) -> dict:
    path = Path(models_dir) / BUNDLE_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No calibration bundle at {path} (expected file '{BUNDLE_NAME}'). "
            "Train one first: python scripts/train_router.py"
        )
    bundle = joblib.load(path)
    version = bundle.get("version")
    if version is not None and version != BUNDLE_VERSION:
        raise ValueError(f"unsupported calibration bundle version {version!r} in {path}")
    logger.info("loaded calibration bundle %s (trained_at=%s)",
                path, bundle.get("metadata", {}).get("trained_at", "unknown"))
    return bundle


def clear_calibration_cache() -> None:
    """Drop memoized bundles (used after retraining into an already-loaded directory)."""
    _load_bundle.cache_clear()


def load_calibration(models_dir: str | Path | None = None) -> ConfidenceCalibrator:
    """Load the trained bundle from `models_dir` (default backend/models).

    Memoized per resolved directory: repeat calls reuse the same deserialized objects.
    Raises FileNotFoundError naming the expected path when no bundle exists.
    """
    directory = Path(models_dir).resolve() if models_dir is not None else DEFAULT_MODELS_DIR
    return ConfidenceCalibrator(_load_bundle(str(directory)))


# --- Training ----------------------------------------------------------------------
def train_and_save(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    models_dir: str | Path,
    *,
    seed: int = 42,
    max_iter: int = 1000,
) -> dict:
    """Fit the logistic calibrator and save one joblib bundle.

    Input frames carry raw observables: columns attempt, proposal, confidence,
    tool_calls, tool_errored. Labels are derived inside via `bootstrap_decision_label`
    so the rule lives in exactly one place. Metrics reflect `val_df` when given and
    include a binned reliability check of predicted vs observed correctness.
    """
    frames: list[tuple[str, pd.DataFrame]] = [("train_df", train_df)]
    if val_df is not None:
        frames.append(("val_df", val_df))
    required = {"attempt", "proposal", "confidence", "tool_calls", "tool_errored"}
    for name, frame in frames:
        absent = required - set(frame.columns)
        if absent:
            raise ValueError(f"{name} is missing required columns: {sorted(absent)}")

    def design_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(
            [
                build_calibration_features(
                    attempt=int(attempt),
                    proposal=str(proposal),
                    confidence=float(confidence),
                    tool_calls=int(tool_calls),
                    tool_errored=bool(tool_errored),
                )
                for attempt, proposal, confidence, tool_calls, tool_errored in zip(
                    frame["attempt"],
                    frame["proposal"],
                    frame["confidence"],
                    frame["tool_calls"],
                    frame["tool_errored"],
                    strict=True,
                )
            ]
        )
        targets = np.asarray(
            [
                bootstrap_decision_label(
                    attempt=int(attempt),
                    confidence=float(confidence),
                    tool_errored=bool(tool_errored),
                )
                for attempt, confidence, tool_errored in zip(
                    frame["attempt"], frame["confidence"], frame["tool_errored"], strict=True
                )
            ],
            dtype=np.int64,
        )
        return matrix, targets

    x_train, y_train = design_matrix(train_df)
    logger.info(
        "training calibration head on %d rows (%d correct)",
        len(train_df), int(y_train.sum()),
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=max_iter, random_state=seed)),
        ]
    )
    model.fit(x_train, y_train)

    eval_frame = val_df if val_df is not None else train_df
    evaluated_on = "validation" if val_df is not None else "train"
    x_eval, y_eval = design_matrix(eval_frame)
    p_eval = np.asarray(model.predict_proba(x_eval))[:, list(model.classes_).index(1)]
    metrics = {
        "accuracy": round(float(accuracy_score(y_eval, (p_eval >= 0.5).astype(int))), 4),
        "macro_f1": round(float(f1_score(y_eval, (p_eval >= 0.5).astype(int),
                                         average="macro", zero_division=0)), 4),
        "log_loss": round(float(log_loss(y_eval, p_eval, labels=[0, 1])), 4),
        "reliability_bins": reliability_bins(y_eval, p_eval),
        "dataset": {
            "train_rows": len(train_df),
            "val_rows": len(val_df) if val_df is not None else 0,
            "evaluated_on": evaluated_on,
            "train_correct_fraction": round(float(y_train.mean()), 4),
        },
        "seed": seed,
    }

    metadata = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "max_iter": max_iter,
        "feature_names": list(FEATURE_NAMES),
        "labeling_rule": LABELING_RULE_DOC,
        "threshold_mapping": (
            f"base + {TRUST_SENSITIVITY} * (0.5 - p_correct), "
            f"clamped to [{THRESHOLD_FLOOR}, {THRESHOLD_CEILING}]"
        ),
        "train_rows": metrics["dataset"]["train_rows"],
        "val_rows": metrics["dataset"]["val_rows"],
        "evaluated_on": evaluated_on,
        "sklearn_version": sklearn.__version__,
    }
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / BUNDLE_NAME
    joblib.dump({"version": BUNDLE_VERSION, "model": model, "metadata": metadata}, bundle_path)
    clear_calibration_cache()  # any previously memoized load of this dir is now stale
    logger.info("saved calibration bundle to %s", bundle_path)
    return metrics
