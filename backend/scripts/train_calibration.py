"""Train the confidence calibration model (FR-15) and report reliability bins.

Generates bootstrap observable training samples covering realistic agent execution
parameters (attempt, proposal length, self-reported confidence, tool calls, tool errors),
fits the logistic calibrator via app.ml.calibration.train_and_save into backend/models/,
re-loads the bundle through load_calibration, evaluates on held-out test rows,
prints reliability bins and metrics, and writes backend/models/calibration_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from app.ml.calibration import (
    BUNDLE_NAME,
    bootstrap_decision_label,
    load_calibration,
    reliability_bins,
    train_and_save,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BACKEND_DIR / "models"


def generate_bootstrap_dataset(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic agent execution observables."""
    rng = np.random.default_rng(seed)

    attempts = rng.choice([1, 2], size=n_samples, p=[0.75, 0.25])
    confidences = rng.beta(a=5, b=2, size=n_samples)  # skewed towards higher confidences
    tool_calls = rng.poisson(lam=1.5, size=n_samples)
    tool_errored = rng.choice([0, 1], size=n_samples, p=[0.90, 0.10])

    # Generate sample proposals of varied length
    proposals = []
    base_texts = [
        "Refund processed for $25.00 back to original payment method.",
        "Your account MFA status has been reset. Please log in using the temporary link sent to your email.",
        "Package tracking status is updated: shipment was delayed in transit and is expected tomorrow.",
        "We have reviewed your request and updated your subscription plan to Pro Tier.",
        "A replacement item has been dispatched under expedited shipping at no extra cost.",
    ]
    for _ in range(n_samples):
        base = rng.choice(base_texts)
        repeat = rng.integers(1, 4)
        proposals.append(" ".join([base] * repeat))

    df = pd.DataFrame(
        {
            "attempt": attempts,
            "proposal": proposals,
            "confidence": np.round(confidences, 3),
            "tool_calls": tool_calls,
            "tool_errored": tool_errored,
        }
    )
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=3000)
    args = parser.parse_args()

    print(f"generating {args.samples} bootstrap agent execution samples...")
    full_df = generate_bootstrap_dataset(args.samples, seed=args.seed)

    # 70/15/15 split
    n_train = int(len(full_df) * 0.70)
    n_val = int(len(full_df) * 0.15)
    train_df = full_df.iloc[:n_train]
    val_df = full_df.iloc[n_train : n_train + n_val]
    test_df = full_df.iloc[n_train + n_val :]

    print(f"train={len(train_df)} · val={len(val_df)} · test={len(test_df)}")
    print(f"\ntraining calibration model into {MODELS_DIR / BUNDLE_NAME} ...")

    val_metrics = train_and_save(train_df, val_df, MODELS_DIR, seed=args.seed)

    calibrator = load_calibration(MODELS_DIR)

    # Evaluate on held-out test split
    start = time.perf_counter()
    p_preds = []
    y_test = []
    for _, row in test_df.iterrows():
        p = calibrator.predict_probability(
            attempt=int(row["attempt"]),
            proposal=str(row["proposal"]),
            confidence=float(row["confidence"]),
            tool_calls=int(row["tool_calls"]),
            tool_errored=bool(row["tool_errored"]),
        )
        y = bootstrap_decision_label(
            attempt=int(row["attempt"]),
            confidence=float(row["confidence"]),
            tool_errored=bool(row["tool_errored"]),
        )
        p_preds.append(p)
        y_test.append(y)

    elapsed = time.perf_counter() - start
    y_test_arr = np.asarray(y_test, dtype=int)
    p_preds_arr = np.asarray(p_preds, dtype=float)
    y_preds_binary = (p_preds_arr >= 0.5).astype(int)

    acc = round(float(accuracy_score(y_test_arr, y_preds_binary)), 4)
    loss = round(float(log_loss(y_test_arr, p_preds_arr, labels=[0, 1])), 4)
    brier = round(float(brier_score_loss(y_test_arr, p_preds_arr)), 4)
    bins = reliability_bins(y_test_arr, p_preds_arr)

    print(f"\nCalibration Test Results (n={len(test_df)}):")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Log-Loss:    {loss:.4f}")
    print(f"  Brier Score: {brier:.4f}")

    print("\nReliability Bins (Predicted Probability vs Observed Correct Rate):")
    print(f"  {'Range':<15}{'Count':>8}{'Mean Predicted':>18}{'Observed Rate':>18}")
    print("  " + "-" * 62)
    for b in bins:
        mean_p = f"{b['mean_predicted']:.4f}" if b["mean_predicted"] is not None else "N/A"
        obs = f"{b['observed_rate']:.4f}" if b["observed_rate"] is not None else "N/A"
        print(f"  {b['range']:<15}{b['n']:>8}{mean_p:>18}{obs:>18}")

    print(f"\nevaluated {len(test_df)} rows in {elapsed:.2f}s ({elapsed / len(test_df) * 1000:.2f} ms/run)")

    record = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_file": BUNDLE_NAME,
        "config": {
            "seed": args.seed,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        },
        "validation_metrics": val_metrics,
        "test_metrics": {
            "accuracy": acc,
            "log_loss": loss,
            "brier_score": brier,
            "reliability_bins": bins,
        },
        "gates": {
            "brier_score": {
                "threshold": 0.15,
                "value": brier,
                "verdict": "PASS" if brier <= 0.15 else "FAIL",
            },
            "accuracy": {
                "threshold": 0.80,
                "value": acc,
                "verdict": "PASS" if acc >= 0.80 else "FAIL",
            },
        },
    }

    out_metrics = MODELS_DIR / "calibration_metrics.json"
    out_metrics.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote calibration metrics report to {out_metrics}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
