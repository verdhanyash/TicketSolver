# ROUTER_METRICS.md — Module 7 Cost Router & Confidence Calibration

**Date:** 2026-08-25  
**Components:**
1. Complexity & Cost Router (FR-14) — XGBoost classifier + Logistic Regression baseline (SRS §6.3)
2. Confidence Calibration (FR-15) — Logistic calibrator mapping execution observables to dynamic escalation thresholds

---

## 1. Complexity & Cost Router (FR-14)

### Overview
Routes incoming tickets to the cheap model tier (`meta/llama-3.1-8b-instruct`) or strong model tier (`meta/llama-3.1-70b-instruct`) based on lexical and syntactic features, punctuation patterns, and predicted category/severity labels from the FR-1 classifier.

- **Bundle Artifact:** `backend/models/ticket_router.joblib`
- **Training Data:** `backend/data/processed/train.csv` ($n = 9,611$ rows)
- **Validation Data:** `backend/data/processed/val.csv` ($n = 2,060$ rows)
- **Held-Out Test Set:** `backend/data/processed/test.csv` ($n = 2,060$ rows)
- **Feature Set:** 11 engineered numeric signals (`subject_chars`, `subject_words`, `body_chars`, `body_words`, `question_marks`, `urgency_keyword_count`, `politeness_keyword_count`, `subject_upper_ratio`, `digit_count`, `category_index`, `severity_index`).

### Test-Split Results ($n = 2,060$)

| Model | Accuracy | Macro-F1 | Simple F1 | Complex F1 | Inference Latency |
|---|---|---|---|---|---|
| **XGBoost Classifier** | **0.7879 (78.79%)** | **0.7857** | **0.7639** | **0.8074** | ~2.21 ms / ticket |
| **Logistic Regression Baseline** | **0.6951 (69.51%)** | **0.6860** | **0.6323** | **0.7396** | ~1.45 ms / ticket |

### Quality Gates

```
GATE complexity macro-F1 >= 0.70: 0.7857 PASS
COMPARISON XGBoost >= Logistic Regression: 0.7857 vs 0.6860 PASS (+9.97% F1 advantage)
```

---

## 2. Confidence Calibration (FR-15)

### Overview
Learns mapping from runtime execution observables (`attempt`, `proposal_length`, `confidence`, `tool_calls`, `tool_errored`) to $P(\text{decision correct})$ to dynamically adjust the orchestrator's escalation threshold between a strict floor ($0.30$) and ceiling ($0.90$).

- **Bundle Artifact:** `backend/models/confidence_calibration.joblib`
- **Model:** Regularized Logistic Regression with StandardScaler
- **Held-Out Test Set:** $n = 450$ agent execution runs

### Test-Split Performance ($n = 450$)

| Metric | Value | Threshold | Status |
|---|---|---|---|
| **Accuracy** | **1.0000** | $\ge 0.80$ | **PASS** |
| **Log-Loss** | **0.0519** | — | — |
| **Brier Score** | **0.0125** | $\le 0.15$ | **PASS** |
| **Inference Latency** | ~0.17 ms / decision | — | — |

### Reliability Bins (Predicted Probability vs Observed Correctness)

| Probability Range | Count | Mean Predicted | Observed Rate |
|---|---|---|---|
| $[0.00, 0.60)$ | 346 | 0.0407 | 0.0202 |
| $[0.60, 0.70)$ | 5 | 0.6621 | 1.0000 |
| $[0.70, 0.80)$ | 9 | 0.7621 | 1.0000 |
| $[0.80, 0.90)$ | 10 | 0.8503 | 1.0000 |
| $[0.90, 1.00]$ | 80 | 0.9855 | 1.0000 |

---

## 3. Orchestrator Integration & Observability

- **Step 1 (Triage):** `ml_prediction/ticket_classification` tags category and severity.
- **Step 2 (Routing):** `ml_prediction/cost_routing` selects `selected_model` and logs complexity rationale.
- **Step 3 (Investigation):** Passes `selected_model` to the investigator agent; `llm_call` trace records the exact NIM model invoked.
- **Step 4 (Decision Calibration):** `_calibrated_threshold` computes dynamic threshold, adjusting the bar based on agent reliability.
