"""CLI runner for TicketSolver evaluation harness (FR-12, FR-13, FR-17).

Usage:
  python scripts/run_eval.py
  python scripts/run_eval.py --dataset eval/golden_tickets.json --mock
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add backend directory to sys.path so app imports work when executed directly
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.eval.runner import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="TicketSolver Evaluation Harness Runner")
    parser.add_argument("--dataset", type=str, help="Path to golden tickets JSON", default=None)
    parser.add_argument("--output-json", type=str, help="Path to output JSON report", default=None)
    parser.add_argument("--output-md", type=str, help="Path to output Markdown report", default=None)
    parser.add_argument("--mock", action="store_true", help="Force deterministic investigator simulation", default=None)
    args = parser.parse_args()

    print("=" * 72)
    print("           TICKETSOLVER MULTI-AGENT EVALUATION HARNESS            ")
    print("=" * 72)

    report = run_evaluation(
        dataset_path=args.dataset,
        output_json=args.output_json,
        output_md=args.output_md,
        mock_mode=args.mock,
    )

    summary = report["summary"]
    gates = report["quality_gates"]
    confusion = summary["confusion_matrix"]
    models = report.get("model_metrics", {})

    print("\n" + "=" * 72)
    print("                        EVALUATION SUMMARY                        ")
    print("=" * 72)
    print(f"Run ID:             {report['eval_run_id']}")
    print(f"Total Tickets:      {summary['total_tickets']}")
    print(f"Overall Accuracy:   {summary['outcome_accuracy_pct']}% ({summary['correct_outcomes']}/{summary['total_tickets']})")
    print(f"Resolution Rate:    {summary['resolution_correctness_pct']}% (Routine)")
    print(f"Guardrail Rate:     {summary['guardrail_intercept_pct']}% (Safety Intercepts)")
    print(f"Escalation F1:      {summary['escalation_metrics']['f1']}")

    print("\n--- Outcome Confusion Matrix ---")
    header_col = "Expected \\ Actual"
    print(f"{header_col:<20} | {'Resolved':<10} | {'Approval':<10} | {'Escalated':<10}")
    print("-" * 58)
    for exp in ["resolved", "pending_approval", "escalated"]:
        print(
            f"{exp:<20} | "
            f"{confusion[exp]['resolved']:<10} | "
            f"{confusion[exp]['pending_approval']:<10} | "
            f"{confusion[exp]['escalated']:<10}"
        )

    print("\n--- ML Model Metrics (FR-17) ---")
    m6 = models.get("ticket_classifier_m6") or {}
    print(f"M6 Classifier Macro-F1:   Category: {m6.get('category_macro_f1', 'N/A')}, Severity: {m6.get('severity_macro_f1', 'N/A')}")
    m7_r = models.get("cost_router_m7") or {}
    print(f"M7 Cost Router:           XGB F1: {m7_r.get('xgboost_macro_f1', 'N/A')} vs Base F1: {m7_r.get('baseline_macro_f1', 'N/A')} (+{m7_r.get('advantage_f1', 0)*100:.2f}%)")
    m7_c = models.get("confidence_calibration_m7") or {}
    print(f"M7 Calibration:           Brier Score: {m7_c.get('brier_score', 'N/A')} (Acc: {m7_c.get('accuracy', 'N/A')})")

    print("\n--- Quality Gates (SRS §10) ---")
    for gate_name, passed in gates.items():
        if gate_name == "all_passed":
            continue
        status = "PASS" if passed else "FAIL"
        symbol = "[OK]" if passed else "[XX]"
        print(f"{symbol} {gate_name:<30}: {status}")

    print("=" * 72)
    overall_status = "ALL QUALITY GATES PASSED" if gates["all_passed"] else "SOME GATES FAILED"
    print(f"OVERALL STATUS: {overall_status}")
    print("=" * 72 + "\n")

    if not gates["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
