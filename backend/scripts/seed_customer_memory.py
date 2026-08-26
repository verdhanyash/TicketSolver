"""Seed long-term customer memory for the synthetic CRM customers (FR-6).

Upserts one `customer_memory` row per account in `app/tools/accounts.py`, keeping
plan tier / account age consistent with that simulated source of truth and adding a
few plausible support-history notes. Idempotent — safe to rerun.

Usage:
  python scripts/seed_customer_memory.py            # seed
  python scripts/seed_customer_memory.py --verify   # seed + round-trip check
Requires running Postgres (see backend/docker-compose.yml).
"""

from __future__ import annotations

import argparse

from app.memory.long_term import (
    CustomerMemoryRecord,
    get_customer_memory,
    upsert_customer_memory,
)
from app.tools.accounts import ACCOUNTS

# Fictional support history per customer; plan/age mirror ACCOUNTS so memory and the
# simulated CRM never disagree. All content synthetic.
SEED_NOTES: dict[str, list[str]] = {
    "CUST-1001": [
        "Prefers email follow-ups over calls.",
        "Aug 2026: late onboarding-kit delivery acknowledged; goodwill discount applied once.",
    ],
    "CUST-1002": ["New signup; no prior tickets."],
    "CUST-1003": [
        "Enterprise contract owner; escalation path is their named CSM.",
        "Renewal window opens Nov 2026.",
    ],
    "CUST-1004": ["Mar 2026: billing dispute resolved in customer's favor ($8 credit)."],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="re-read rows and assert equality")
    args = parser.parse_args()

    for customer_id, account in ACCOUNTS.items():
        record = CustomerMemoryRecord(
            customer_id=customer_id,
            plan_tier=account["plan_tier"],
            account_age_days=account["account_age_days"],
            notes=SEED_NOTES.get(customer_id, []),
        )
        upsert_customer_memory(record)
        print(f"upserted {customer_id} (plan={record.plan_tier}, age={record.account_age_days}d)")

    if not args.verify:
        return

    failures: list[str] = []
    for customer_id, account in ACCOUNTS.items():
        stored = get_customer_memory(customer_id)
        expected = CustomerMemoryRecord(
            customer_id=customer_id,
            plan_tier=account["plan_tier"],
            account_age_days=account["account_age_days"],
            notes=SEED_NOTES.get(customer_id, []),
        )
        if stored != expected:
            failures.append(f"{customer_id}: {stored!r} != {expected!r}")
    if failures:
        raise SystemExit("ROUND-TRIP MISMATCH:\n" + "\n".join(failures))
    print(f"round-trip OK: {len(ACCOUNTS)} customers read back identical")


if __name__ == "__main__":
    main()
