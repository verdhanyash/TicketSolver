# DATA.md — Dataset Provenance & Licensing

Per `CLAUDE.md` / SRS §8: **no dataset is committed to the repository until its license
is verified.** This file records provenance, license status, and acquisition steps.

## Primary ticket dataset (M6 training data)

| Field | Value |
|---|---|
| Name | Customer Support Tickets (`Tobi-Bueck/customer-support-tickets`) |
| Source | https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets |
| File | `aa_dataset-tickets-multi-lang-5-2-50-version.csv` (~60 MB, 28,587 rows × 15 columns, EN+DE) |
| Retrieved | 2026-08-25, via Hugging Face's public `resolve/main` endpoint (no account needed) |
| Local copy | `backend/data/raw/hf_support_tickets_main.csv` — **gitignored, never committed** |

### License: CC BY-NC 4.0 (verified)

Confirmed via the Hugging Face API dataset tags
(`['license:cc-by-nc-4.0', 'size_categories:10K<n<100K', ...]`) on 2026-08-25:
https://huggingface.co/api/datasets/Tobi-Bueck/customer-support-tickets.

Implications for this repo:

- **Attribution required** — this file is the attribution record; keep the source table
  above intact wherever the data is used.
- **Non-commercial only** — fine for this portfolio/demo project; must be re-checked
  before any commercial deployment built on artifacts trained from it.
- The raw CSV itself stays out of version control regardless; only derived, trained
  model weights (`backend/models/ticket_classifier.joblib`) are committed.

### Why this dataset

It replaced Kaggle `suraj520/customer-support-ticket-dataset`, whose labels proved
statistically independent of the ticket text (chance-level held-out F1 for two model
families — full evidence in `Docs/CLASSIFIER_METRICS.md`, Appendix A). Before committing
to the switch, a logistic-regression sanity check on this dataset's EN subset showed real
text→label signal (3-fold CV macro-F1: type 0.812 vs ~0.25 chance; priority 0.476 vs
~0.33 chance).

## Schema mapping to application fields

The pipeline keeps only `language == "en"` rows (the classifier serves English tickets;
DE rows are ignored).

| Dataset column | Used as | App field | Normalization |
|---|---|---|---|
| `type` | classification label | `tickets.category` | lowercase: `change`, `incident`, `problem`, `request` |
| `priority` | classification label | `tickets.severity` | lowercase: `low`, `medium`, `high` |
| `subject` | feature text | — | verbatim into the feature pipeline |
| `body` | feature text | — | verbatim into the feature pipeline |
| `language` | row filter | — | keep `"en"` only |

Unused columns (`answer`, `queue`, `version`, `tag_1`…`tag_8`) are left for future work
(e.g., answer-conditioned eval sets).

Cleaning result (2026-08-25): 28,587 raw → 16,338 EN rows kept → 2,607 dropped with
nulls in a required column → **13,731 usable rows**.

Split sizes and label distributions after stratified 70/15/15 (seed 42, joint
category\|severity key):

| label | train | val | test |
|---|---|---|---|
| change | 978 | 210 | 209 |
| incident | 3871 | 830 | 830 |
| problem | 2012 | 431 | 431 |
| request | 2750 | 589 | 590 |
| low | 1982 | 424 | 426 |
| medium | 3917 | 840 | 838 |
| high | 3712 | 796 | 796 |

Totals: **train=9,611 · val=2,060 · test=2,060**.

## Pipeline layout

```
backend/
  data/
    raw/hf_support_tickets_main.csv       # downloaded, gitignored
    processed/{train,val,test}.csv        # produced by scripts/prepare_data.py, gitignored
  models/
    ticket_classifier.joblib              # trained bundle (vectorizer + both XGBoost models)
    metrics.json                          # held-out metrics, committed
  scripts/
    prepare_data.py                       # download (optional) + EN filter + clean + stratified split
    train_classifier.py                   # trains, evaluates, writes artifacts + metrics
```

Reproduce:

```
cd backend
./.venv/Scripts/python.exe scripts/prepare_data.py      # add --download to fetch the CSV
PYTHONPATH=. ./.venv/Scripts/python.exe scripts/train_classifier.py
```

## Synthetic edge cases (SRS §8)

LLM-generated edge-case tickets (NVIDIA NIM) are part of §8 but are **deferred until a
NIM API key is configured** — same standing caveat as the M1–M5 LLM-path validation. The
classifier pipeline accepts an augmented CSV, so synthetic rows can be mixed in later
without code changes.

---

## Appendix: datasets evaluated and rejected

### Kaggle `suraj520/customer-support-ticket-dataset`

- Source: https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
  (8,469 rows × 17 cols; license reported as CC0 by secondary sources, never confirmed
  on-page).
- Evaluated 2026-08-25 as the initial M6 dataset. Held-out results were at chance level
  for both XGBoost and a logistic-regression baseline (diagnostics: 16 unique subjects
  spread uniformly across all classes; bodies statistically independent of labels).
  Full measurements and interpretation: `Docs/CLASSIFIER_METRICS.md`, Appendix A.
- Verdict: **rejected as training data** — labels carry no learnable text→label signal.
  The raw CSV remains in `backend/data/raw/` (gitignored) only as evidence for that
  analysis; no artifact in `backend/models/` is trained on it anymore.
