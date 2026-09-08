"""Train the complexity router (FR-14) and evaluate against the baseline (SRS §6.3).

Loads the processed splits from backend/data/processed/{train,val,test}.csv,
trains the XGBoost complexity classifier and the Logistic Regression baseline via
app.ml.router.train_and_save into backend/models/, re-loads the bundle through the
production serving path, evaluates on the TEST split, prints comparison tables,
and writes backend/models/router_metrics.json.
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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from app.ml.classifier import load_classifier
from app.ml.router import (
    BUNDLE_NAME,
    COMPLEXITY_LABELS,
    bootstrap_complexity_label,
    build_router_features,
    load_router,
    train_and_save,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
MODELS_DIR = BACKEND_DIR / "models"

GATE_MACRO_F1 = 0.70  # complexity gate on held-out test


def print_comparison_table(xgb_metrics: dict, lr_metrics: dict, rows: int) -> None:
    print(f"\nComplexity Router Comparison on Test Split (n={rows}):")
    print(f"  {'Model':<22}{'Accuracy':>12}{'Macro-F1':>12}{'Simple F1':>12}{'Complex F1':>12}")
    print("  " + "-" * 70)
    for name, m in [("XGBoost", xgb_metrics), ("Logistic Regression", lr_metrics)]:
        acc = m["accuracy"]
        f1 = m["macro_f1"]
        s_f1 = m["per_class"]["simple"]["f1"]
        c_f1 = m["per_class"]["complex"]["f1"]
        print(f"  {name:<22}{acc:>12.4f}{f1:>12.4f}{s_f1:>12.4f}{c_f1:>12.4f}")


def evaluate_router_on_test(
    router_bundle: dict, test_df: pd.DataFrame, thresholds: dict, classifier=None
) -> tuple[dict, dict, float]:
    """Evaluate both heads in the bundle on the held-out test set."""
    xgb_model = router_bundle["xgb_model"]
    lr_model = router_bundle["lr_model"]

    start = time.perf_counter()
    categories = list(test_df["category"])
    severities = list(test_df["severity"])
    if classifier is not None:
        try:
            preds = [
                classifier.predict(str(s), str(b))
                for s, b in zip(test_df["subject"], test_df["body"], strict=True)
            ]
            categories = [p.category for p in preds]
            severities = [p.severity for p in preds]
        except Exception:  # noqa: BLE001, S110
            pass

    x_test = np.asarray(
        [
            build_router_features(str(s), str(b), cat, sev)
            for s, b, cat, sev in zip(
                test_df["subject"], test_df["body"], categories, severities, strict=True
            )
        ]
    )
    y_test = np.asarray(
        [
            bootstrap_complexity_label(
                severity=sev,
                category=cat,
                subject=str(s),
                body=str(b),
                thresholds=thresholds,
            )
            for sev, cat, s, b in zip(
                test_df["severity"],
                test_df["category"],
                test_df["subject"],
                test_df["body"],
                strict=True,
            )
        ],
        dtype=np.int64,
    )

    xgb_preds = xgb_model.predict(x_test)
    lr_preds = lr_model.predict(x_test)
    elapsed = time.perf_counter() - start

    def metrics(preds: np.ndarray) -> dict:
        indices = [0, 1]
        prec, rec, f1s, supp = precision_recall_fscore_support(
            y_test, preds, labels=indices, zero_division=0
        )
        return {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "macro_f1": round(float(f1_score(y_test, preds, labels=indices, average="macro", zero_division=0)), 4),
            "per_class": {
                COMPLEXITY_LABELS[i]: {
                    "precision": round(float(prec[i]), 4),
                    "recall": round(float(rec[i]), 4),
                    "f1": round(float(f1s[i]), 4),
                    "support": int(supp[i]),
                }
                for i in indices
            },
        }

    return metrics(xgb_preds), metrics(lr_preds), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()

    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"

    for path in (train_path, val_path, test_path):
        if not path.is_file():
            print(f"error: required split missing: {path}\nRun scripts/prepare_data.py first.", file=sys.stderr)
            return 1

    print("loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    print(f"train={len(train_df)} · val={len(val_df)} · test={len(test_df)}")

    classifier = None
    try:
        classifier = load_classifier(MODELS_DIR)
        print("loaded production classifier for triage feature emulation")
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not load classifier ({exc}); using row labels as fallback")

    print(f"\ntraining router heads into {MODELS_DIR / BUNDLE_NAME} ...")
    val_metrics = train_and_save(
        train_df,
        val_df,
        MODELS_DIR,
        classifier=classifier,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    thresholds = val_metrics["thresholds"]

    # Load via production loader and test bundle directly
    router = load_router(MODELS_DIR)
    # Access internal bundle for evaluation of both heads
    bundle = router._bundle if hasattr(router, "_bundle") else router.__dict__.get("_bundle")
    if bundle is None:
        import joblib
        bundle = joblib.load(MODELS_DIR / BUNDLE_NAME)

    xgb_test_metrics, lr_test_metrics, elapsed = evaluate_router_on_test(
        bundle, test_df, thresholds=thresholds, classifier=classifier
    )

    print_comparison_table(xgb_test_metrics, lr_test_metrics, len(test_df))
    print(f"evaluated {len(test_df)} test rows in {elapsed:.2f}s ({elapsed / len(test_df) * 1000:.2f} ms/ticket)")

    gate_val = xgb_test_metrics["macro_f1"]
    gate_pass = gate_val >= GATE_MACRO_F1
    print(f"\nGATE complexity macro-F1 >= {GATE_MACRO_F1:.2f}: {gate_val:.4f} {'PASS' if gate_pass else 'FAIL'}")

    comparison_pass = xgb_test_metrics["macro_f1"] >= lr_test_metrics["macro_f1"]
    print(f"COMPARISON XGBoost >= Logistic Regression: {xgb_test_metrics['macro_f1']:.4f} vs {lr_test_metrics['macro_f1']:.4f} {'PASS' if comparison_pass else 'FAIL'}")

    record = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_file": BUNDLE_NAME,
        "config": {
            "seed": args.seed,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        },
        "thresholds": thresholds,
        "validation_metrics": val_metrics,
        "test_metrics": {
            "xgboost": xgb_test_metrics,
            "logistic_regression": lr_test_metrics,
        },
        "gates": {
            "complexity_macro_f1": {
                "threshold": GATE_MACRO_F1,
                "value": gate_val,
                "verdict": "PASS" if gate_pass else "FAIL",
            },
            "xgboost_vs_baseline": {
                "xgb_f1": xgb_test_metrics["macro_f1"],
                "lr_f1": lr_test_metrics["macro_f1"],
                "verdict": "PASS" if comparison_pass else "FAIL",
            },
        },
    }

    out_metrics = MODELS_DIR / "router_metrics.json"
    out_metrics.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote metrics report to {out_metrics}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
