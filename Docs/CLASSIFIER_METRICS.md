# CLASSIFIER_METRICS.md — M6 Ticket Classifier, Held-Out Results

**Date:** 2026-08-25 (retrained on the current primary dataset)
**Component:** Ticket category/severity classifier (FR-1) — TF-IDF features + two XGBoost heads
**Verdict up front:** both quality gates **PASS** with margin on the Tobi-Bueck EN subset
(category 0.8501 vs ≥0.60; severity 0.5899 vs ≥0.40). The earlier dataset evaluated for
M6 (Kaggle `suraj520`) failed both gates because its labels are statistically independent
of the text; that full negative finding is preserved in [Appendix A](#appendix-a--rejected-dataset-kaggle-suraj520).

## Dataset

Tobi-Bueck `customer-support-tickets`, EN rows only — see `Docs/DATA.md` for provenance
and the CC BY-NC 4.0 license record.

| Field | Value |
|---|---|
| File | `backend/data/raw/hf_support_tickets_main.csv` (gitignored) — 28,587 rows × 15 cols |
| EN rows kept | 16,338 (`language == "en"`); 12,249 DE rows dropped |
| Nulls dropped | 2,607 rows missing a required column |
| Usable | **13,731 rows**, split 70/15/15 stratified on joint `category\|severity` (seed 42) |

Split sizes and distributions are tabulated in `Docs/DATA.md`; totals:
**train=9,611 · val=2,060 · test=2,060**.

## Commands

```
cd backend
./.venv/Scripts/python.exe scripts/prepare_data.py      # ~10 s
PYTHONPATH=. ./.venv/Scripts/python.exe scripts/train_classifier.py  # ~3 min
```

Config: seed=42 · n_estimators=300 · max_depth=6 · learning_rate=0.2 · tree_method="hist" ·
eval_metric="mlogloss" · scikit-learn 1.8.0 · xgboost 3.4.1. The feature pipeline is fitted
on the training split only; validation is used for post-training reporting; **test is
touched exactly once**, after saving, through the production `load_classifier()` path.

## Test-split results (n=2060)

### Category (ticket type)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| change | 0.9847 | 0.9234 | 0.9531 | 209 |
| incident | 0.7810 | 0.9193 | 0.8445 | 830 |
| problem | 0.7653 | 0.5220 | 0.6207 | 431 |
| request | 0.9798 | 0.9847 | 0.9822 | 590 |
| **macro-F1** | | | **0.8501** | |
| **accuracy** | | | **0.8553** | |

### Severity (ticket priority)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| low | 0.7080 | 0.3756 | 0.4908 | 426 |
| medium | 0.5782 | 0.6969 | 0.6320 | 838 |
| high | 0.6359 | 0.6583 | 0.6469 | 796 |
| **macro-F1** | | | **0.5899** | |
| **accuracy** | | | **0.6155** | |

Validation split agrees with test (category macro-F1 0.8512 / accuracy 0.8578; severity
0.5899 / 0.6199) — the result is stable, not a tuning artifact.

Inference speed: 2,060 tickets in 5.29 s (~2.6 ms/ticket) via `ticket_classifier.joblib`.

## Gates

```
GATE category macro-F1 >= 0.60: 0.8501 PASS
GATE severity macro-F1 >= 0.40: 0.5899 PASS
```

Gate rationale: category chance level is ~0.25 (four classes), so 0.60 demonstrates real
signal. Severity has three classes (random-guess chance ~0.33) and priority correlates
only partially with text, so its bar sits at 0.40 — above chance, below pretense. The
severity gate was recalibrated from 0.34 when the dataset changed: 0.34 was calibrated
against the *old* dataset's four-class ~0.25 chance floor and would no longer demonstrate
signal over three classes.

## Interpretation

- Category is strongly learnable from subject+body text on this corpus — per-class F1
  spans 0.62 (`problem`) to 0.98 (`request`). The weak spot is `problem` recall (0.52):
  "problem" tickets lexically overlap with "incident" reports in this dataset, and the
  model resolves the ambiguity toward the majority class.
- Severity lands well above chance but far from ceiling (macro-F1 0.59): priority is only
  partially encoded in ticket text. Low severity suffers most (recall 0.38) — low-priority
  requests read like medium ones. This matches the dataset-selection pre-check (LR CV
  macro-F1 0.476) and is exactly why FR-14's cost router (M7) treats predicted severity as
  one feature among several rather than an oracle.
- Both heads serve through the production path today: predictions persist onto
  `tickets.category`/`tickets.severity` and trace as `ml_prediction/ticket_classification`
  ahead of routing (see M6 record in `Docs/MODULES.md`).

---

## Appendix A — rejected dataset (Kaggle `suraj520`)

**Date:** 2026-08-25 (run executed 2026-08-24T18:57 UTC)
**Verdict:** both gates **FAIL** — kept as evidence for the dataset switch recorded in
`Docs/DATA.md`. The numbers below are real, reproducible, and diagnostics show they
reflect a property of the source data (labels statistically independent of text), not a
pipeline defect.

### Dataset

| Field | Value |
|---|---|
| Source | Kaggle "Customer Support Ticket Dataset" by `suraj520` |
| File | `backend/data/raw/customer_support_tickets.csv` (gitignored) — 8,469 rows × 17 cols |
| Rows dropped in cleaning | 0 (no nulls in the four required columns) |
| Split | 70/15/15 stratified on joint `category\|severity`, `random_state=42` |

Split label distributions (rows):

| label | train | val | test |
|---|---|---|---|
| billing_inquiry | 1144 | 246 | 244 |
| cancellation_request | 1187 | 254 | 254 |
| product_inquiry | 1149 | 246 | 246 |
| refund_request | 1226 | 262 | 264 |
| technical_issue | 1222 | 262 | 263 |
| **low / medium / high / critical** | 1445 / 1535 / 1458 / 1490 | 310 / 328 / 314 / 318 | 308 / 329 / 313 / 321 |

Totals: **train=5928 · val=1270 · test=1271**.

### Test-split results (n=1271)

Category (Ticket Type):

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| billing_inquiry | 0.1552 | 0.1475 | 0.1513 | 244 |
| cancellation_request | 0.2234 | 0.2559 | 0.2385 | 254 |
| product_inquiry | 0.2018 | 0.1789 | 0.1897 | 246 |
| refund_request | 0.2355 | 0.2311 | 0.2333 | 264 |
| technical_issue | 0.2251 | 0.2319 | 0.2285 | 263 |
| **macro-F1** | | | **0.2082** | |
| **accuracy** | | | **0.2101** | |

Severity (Ticket Priority):

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| low | 0.2399 | 0.2110 | 0.2245 | 308 |
| medium | 0.2253 | 0.2492 | 0.2367 | 329 |
| high | 0.2378 | 0.2173 | 0.2270 | 313 |
| critical | 0.2657 | 0.2897 | 0.2772 | 321 |
| **macro-F1** | | | **0.2414** | |
| **accuracy** | | | **0.2423** | |

For context, the same bundle scored on the validation split:
category macro-F1 0.2070 / accuracy 0.2087; severity macro-F1 0.2570 / accuracy 0.2575.
Validation and test agree — the result is stable, not a tuning artifact.

Gates under the thresholds in force at the time:

```
GATE category macro-F1 >= 0.60: 0.2082 FAIL
GATE severity macro-F1 >= 0.34: 0.2414 FAIL
```

### Interpretation

The working assumption going in was "category strong, severity weak." The measurements
say otherwise for this source data: **both targets sit at chance level** (~0.20 for five
balanced classes, ~0.25 for four balanced classes), and every per-class F1 hovers at its
chance value. Diagnostics run after seeing these numbers:

1. **The training pipeline works.** The saved bundle fits its own training split at
   97.8% (category) and 97.0% (severity) accuracy — encoding, alignment, vectorization,
   and the predict path are all mechanically sound. It memorizes the training rows and
   learns nothing that generalizes: the classic signature of fitting noise.
2. **Subjects carry no category signal.** `Ticket Subject` has only 16 unique values;
   *every one of them* appears under all five categories at roughly uniform rates
   (weighted subject→category purity 0.224 vs 0.20 chance).
3. **Descriptions are random filler w.r.t. the labels.** Bodies are nearly unique
   (5,665 distinct among 5,928 train rows); weighted purity among the 303 bodies that
   repeat is 0.44 on tiny counts (singleton noise). Keyword checks confirm it — e.g.
   bodies containing "refund" are spread across categories at 0.16–0.28 each, and
   "crash" at ~0.20 across all five.
4. **An independent baseline agrees.** Logistic regression on the same TF-IDF features
   scores val macro-F1 0.1896 (category). Two model families landing exactly at chance
   means the information isn't there to find, not that XGBoost missed it.

Conclusion: in this Kaggle dataset, `Ticket Type` and `Ticket Priority` are assigned
statistically independently of `Ticket Subject` + `Ticket Description`. No modeling
change (more trees, deeper analysis, different features) can beat chance on labels that
are random with respect to the text. This matches how the dataset reads qualitatively:
templated boilerplate descriptions with shuffled generic sentences.

## Artifacts

| Artifact | Path |
|---|---|
| Trained bundle (current, Tobi-Bueck-trained) | `backend/models/ticket_classifier.joblib` |
| Held-out metrics (machine-readable) | `backend/models/metrics.json` |
| Processed splits | `backend/data/processed/{train,val,test}.csv` |
| Feature/classifier code | `backend/app/ml/features.py`, `backend/app/ml/classifier.py` |
| Pipeline scripts | `backend/scripts/prepare_data.py`, `backend/scripts/train_classifier.py` |
| Tests | `backend/tests/test_ml_features.py`, `backend/tests/test_ml_classifier.py` |
