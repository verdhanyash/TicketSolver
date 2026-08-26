"""Simulated account-lookup tool (SRS FR-2).

An in-memory CRM stand-in with a handful of synthetic customers spanning plan tiers
and security postures. All data is fictional; lookups return copies so callers can
never mutate the store.
"""

from __future__ import annotations

# Fictional customers — no real PII. Keys are canonical customer IDs ("CUST-####").
ACCOUNTS: dict[str, dict] = {
    "CUST-1001": {
        "customer_id": "CUST-1001",
        "name": "Amara Okafor",
        "email": "amara.okafor@example.com",
        "plan_tier": "pro",
        "account_age_days": 412,
        "mfa_enabled": True,
        "last_login": "2026-08-21",
    },
    "CUST-1002": {
        "customer_id": "CUST-1002",
        "name": "Devon Blake",
        "email": "devon.blake@example.com",
        "plan_tier": "free",
        "account_age_days": 23,
        "mfa_enabled": False,
        "last_login": "2026-08-19",
    },
    "CUST-1003": {
        "customer_id": "CUST-1003",
        "name": "Priya Raman",
        "email": "priya.raman@example.com",
        "plan_tier": "enterprise",
        "account_age_days": 1287,
        "mfa_enabled": True,
        "last_login": "2026-08-22",
    },
    "CUST-1004": {
        "customer_id": "CUST-1004",
        "name": "Marcus Feld",
        "email": "marcus.feld@example.com",
        "plan_tier": "free",
        "account_age_days": 95,
        "mfa_enabled": False,
        "last_login": "2026-08-14",
    },
}


def lookup_account(customer_id: str) -> dict | None:
    """Return a copy of the account record for `customer_id`, or None if unknown."""
    record = ACCOUNTS.get(customer_id)
    return dict(record) if record is not None else None
