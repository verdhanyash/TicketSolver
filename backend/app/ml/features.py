"""Feature construction for the ticket classifier (FR-1): labels + text pipeline.

Pure, dependency-light helpers shared verbatim by training (scripts/prepare_data.py,
scripts/train_classifier.py) and serving (app/ml/classifier.py), so label handling and
text assembly cannot drift between the batch and online paths:

- `normalize_category` / `normalize_severity` map raw dataset labels to the exact
  lowercase tokens the orchestrator persists (`tickets.category` / `tickets.severity`);
  anything unrecognized raises ValueError naming the valid labels.
- `combine_text` is the single documented way subject and body become model input.
- `build_feature_pipeline` builds the TF-IDF transform both XGBoost heads consume.

Vectorizer choices (deliberately minimal, fully deterministic):
- word-level unigrams + bigrams — ticket types are strongly lexical ("server down",
  "license key"); bigrams capture those phrases without character-level noise;
- `sublinear_tf` — long description bodies repeat boilerplate terms; log scaling
  stops them from drowning short, signal-dense subjects;
- `min_df=2` — drops one-off typos/tokens so the learned vocabulary generalizes;
- `max_features=20000` — caps bundle size and keeps fits fast at portfolio scale;
- remaining scikit-learn defaults (lowercasing, no stop-word list) stay untouched: a
  curated stop-word list adds configuration surface for marginal gain at this scale.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Canonical label sets, matching the primary training dataset (Tobi-Bueck
# customer-support-tickets, EN subset — see Docs/DATA.md). Categories are sorted
# alphabetically (the order used as the model's integer class encoding); severities
# are in ordinal order low -> high. These live here (not only in classifier.py) so
# normalization and training share exactly one source of truth.
VALID_CATEGORIES = ("change", "incident", "problem", "request")
VALID_SEVERITIES = ("low", "medium", "high")


def normalize_category(raw: str) -> str:
    """Normalize a raw category label to lowercase snake_case (e.g. 'Request').

    Raises ValueError listing the valid labels for anything unrecognized.
    """
    token = str(raw).strip().lower().replace(" ", "_")
    if token not in VALID_CATEGORIES:
        valid = ", ".join(VALID_CATEGORIES)
        raise ValueError(f"unknown category {str(raw)!r}: valid categories are: {valid}")
    return token


def normalize_severity(raw: str) -> str:
    """Normalize a raw severity label to lowercase (e.g. 'High' -> 'high').

    Raises ValueError listing the valid labels for anything unrecognized.
    """
    token = str(raw).strip().lower()
    if token not in VALID_SEVERITIES:
        valid = ", ".join(VALID_SEVERITIES)
        raise ValueError(f"unknown severity {str(raw)!r}: valid severities are: {valid}")
    return token


def combine_text(subject: str, body: str) -> str:
    """Assemble classifier input text: stripped subject, a newline, stripped body.

    The newline is plain whitespace to the TF-IDF tokenizer, so words never glue
    across the subject/body boundary; it is kept for readability of any logged text.
    Subject goes first because it carries the densest signal, and sublinear TF
    weighting keeps very long, repetitive bodies from dominating it.
    """
    return f"{str(subject).strip()}\n{str(body).strip()}"


def build_feature_pipeline() -> Pipeline:
    """Single-step TF-IDF pipeline shared by training and inference.

    Kept as a Pipeline (rather than a bare vectorizer) so the saved bundle and all
    callers go through one fitted object — inference always applies exactly the
    preprocessing the models were trained against. See the module docstring for the
    rationale behind each setting.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    sublinear_tf=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                ),
            )
        ]
    )
