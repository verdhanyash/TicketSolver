"""Train the ticket classifier (FR-1) and report held-out metrics honestly (FR-17).

Loads the processed splits written by scripts/prepare_data.py, trains both XGBoost
heads via app.ml.classifier.train_and_save into backend/models/, re-loads the bundle
through the production serving path, then evaluates on the TEST split — rows never
seen during training or tuning — printing per-class tables and gate verdicts, and
writing backend/models/metrics.json.

Gates are documented thresholds, not hidden failures: the script exits 0 even when a
gate fails because honest numbers beat green checkboxes.

- category macro-F1 >= 0.60: ticket type is well-signalable from subject/body text.
- severity macro-F1 >= 0.40: three priority labels put the random-guess chance level
  at ~0.33; the bar sits above it to demonstrate real signal without pretending a
  separability the labels do not contain (priority correlates only partially with text).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from app.ml.classifier import (
    BUNDLE_NAME,
    CATEGORY_LABELS,
    SEVERITY_LABELS,
    load_classifier,
    train_and_save,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
MODELS_DIR = BACKEND_DIR / "models"

GATES = {"category": 0.60, "severity": 0.40}  # macro-F1 thresholds, rationale in docstring


def print_report(name: str, report: dict, rows: int) -> None:
    per_class = report["per_class"]
    width = max(len(label) for label in per_class) + 2
    print(f"\n{name} (test rows={rows})")
    print(f"  {'label':<{width}}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>9}")
    for label, stats in per_class.items():
        print(
            f"  {label:<{width}}{stats['precision']:>10.4f}{stats['recall']:>10.4f}"
            f"{stats['f1']:>10.4f}{stats['support']:>9d}"
        )
    print(f"  {'macro-F1':<{width}}{report['macro_f1']:>31.4f}")
    print(f"  {'accuracy':<{width}}{report['accuracy']:>31.4f}")


def evaluate_on_test(classifier, test_df: pd.DataFrame) -> tuple[dict, dict, float]:
    """Predict every test row through the production predict() path; return metrics."""
    start = time.perf_counter()
    categories: list[str] = []
    severities: list[str] = []
    for subject, body in zip(test_df["subject"], test_df["body"], strict=True):
        prediction = classifier.predict(str(subject), str(body))
        categories.append(prediction.category)
        severities.append(prediction.severity)
    elapsed = time.perf_counter() - start

    def metrics(column: str, predictions: list[str], labels: tuple[str, ...]) -> dict:
        truth = [labels.index(v) for v in test_df[column]]
        predicted = [labels.index(v) for v in predictions]
        indices = list(range(len(labels)))
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predicted, labels=indices, zero_division=0
        )
        return {
            "accuracy": round(float(accuracy_score(truth, predicted)), 4),
            "macro_f1": round(float(f1_score(truth, predicted, labels=indices, average="macro",
                                             zero_division=0)), 4),
            "per_class": {
                label: {
                    "precision": round(float(precision[i]), 4),
                    "recall": round(float(recall[i]), 4),
                    "f1": round(float(f1[i]), 4),
                    "support": int(support[i]),
                }
                for i, label in enumerate(labels)
            },
        }

    return (
        metrics("category", categories, CATEGORY_LABELS),
        metrics("severity", severities, SEVERITY_LABELS),
        elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=PROCESSED_DIR / "train.csv")
    parser.add_argument("--val", type=Path, default=PROCESSED_DIR / "val.csv")
    parser.add_argument("--test", type=Path, default=PROCESSED_DIR / "test.csv")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    missing = [str(p) for p in (args.train, args.val, args.test) if not p.is_file()]
    if missing:
        print(f"missing split file(s): {', '.join(missing)}", file=sys.stderr)
        print("run scripts/prepare_data.py first", file=sys.stderr)
        sys.exit(1)

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    test_df = pd.read_csv(args.test)
    print(f"splits: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    print("\ntraining (validation-split metrics below; test stays untouched until after save)")
    val_metrics = train_and_save(train_df, val_df, args.models_dir, seed=args.seed)
    for target in ("category", "severity"):
        print(f"  {target}: val macro-F1={val_metrics[target]['macro_f1']:.4f} "
              f"accuracy={val_metrics[target]['accuracy']:.4f}")

    # Evaluate exactly what production would serve: the bundle re-loaded from disk.
    classifier = load_classifier(args.models_dir)
    category_report, severity_report, elapsed = evaluate_on_test(classifier, test_df)

    reports = {"category": category_report, "severity": severity_report}
    for target, report in reports.items():
        print_report(target, report, len(test_df))
    print(f"\n(test inference: {len(test_df)} tickets in {elapsed:.2f}s "
          f"via {BUNDLE_NAME})")

    verdicts = {}
    for target, threshold in GATES.items():
        value = reports[target]["macro_f1"]
        verdict = "PASS" if value >= threshold else "FAIL"
        verdicts[target] = verdict
        print(f"GATE {target} macro-F1 >= {threshold:.2f}: {value:.4f} {verdict}")

    metrics_path = args.models_dir / "metrics.json"
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_file": BUNDLE_NAME,
        "config": classifier.metadata,
        "dataset": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        },
        "validation_metrics": val_metrics,
        "test_metrics": reports,
        "gates": {
            f"{target}_macro_f1": {
                "threshold": threshold,
                "value": reports[target]["macro_f1"],
                "verdict": verdicts[target],
            }
            for target, threshold in GATES.items()
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {metrics_path}")


if __name__ == "__main__":
    main()
