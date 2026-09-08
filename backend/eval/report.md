# Evaluation Harness Report (FR-12, FR-13, FR-17)

**Run ID:** `EVAL-20260908-200146`  
**Generated At:** 2026-09-08T20:01:46.581430+00:00  
**Dataset Size:** 35 tickets  
**Overall Accuracy:** 100.0% (35/35)  

---

## 1. Quality Gates (SRS §10 Success Criteria)

| Gate | Metric | Target | Actual | Verdict |
|---|---|---|---|---|
| **Routine Resolution Rate (§10.3)** | Resolution correctness | ≥ 80.0% | 100.0% | ✅ PASS |
| **Guardrail Safety Intercept (§10.2)** | Intercept correctness | 100.0% | 100.0% | ✅ PASS |
| **Escalation Precision/Recall (FR-13)** | Escalation F1 | ≥ 0.75 | 1.0 | ✅ PASS |
| **M6 Category Classifier Gate** | Category Macro-F1 | ≥ 0.60 | 0.8501 | ✅ PASS |
| **M7 Cost Router Baseline Gate (§6.3)** | XGBoost vs Baseline | Advantage > 0 | +9.97% | ✅ PASS |
| **M7 Confidence Calibration Gate** | Brier Score | ≤ 0.15 | 0.0125 | ✅ PASS |

---

## 2. Scenario Breakdown

| Scenario | Total Tickets | Correct Outcomes | Accuracy |
|---|---|---|---|
| **routine** | 18 | 18 | 100.0% |
| **guardrail_trigger** | 9 | 9 | 100.0% |
| **edge_case** | 8 | 8 | 100.0% |

---

## 3. Outcome Confusion Matrix

| Expected \ Actual | Resolved | Pending Approval (Guardrail) | Escalated |
|---|---|---|---|
| **Resolved** | 18 | 0 | 0 |
| **Pending Approval** | 0 | 9 | 0 |
| **Escalated** | 0 | 0 | 8 |

---

## 4. Multi-Agent ML Model Metrics (FR-17)

### M6 Ticket Classifier (XGBoost Dual Head)
- **Category Accuracy:** 0.8553
- **Category Macro-F1:** 0.8501
- **Severity Macro-F1:** 0.5899

### M7 Cost Router (XGBoost vs Logistic Regression Baseline)
- **XGBoost Complexity F1:** 0.7857
- **Baseline Logistic F1:** 0.686
- **Advantage Margin:** +9.97%

### M7 Confidence Calibration
- **Accuracy:** 1.0
- **Brier Score:** 0.0125 (Threshold: ≤ 0.15)
- **Log Loss:** 0.0519
