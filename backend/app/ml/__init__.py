"""ML models (SRS §6).

- Ticket category/severity classifier (FR-1) — XGBoost / lightweight text classifier
- Complexity -> cheap/strong LLM cost router (FR-14) — XGBoost
- Confidence calibration for escalation thresholds (FR-15) — logistic / small GBM
- Logistic-regression baseline for comparison (SRS §6.3)

No logic yet — training + inference added in a later session.
"""
