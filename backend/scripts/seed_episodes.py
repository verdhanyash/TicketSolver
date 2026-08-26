"""Seed episodic memory (FR-7) with synthetic resolved incidents, then spot-check.

Upserts a handful of fictional incidents + plausible resolutions into the Qdrant
`episodes` collection (embedded via local Ollama). Point IDs are deterministic per
ticket id, so re-running replaces rather than duplicates.

Usage:
  python scripts/seed_episodes.py            # seed
  python scripts/seed_episodes.py --verify   # seed + curated retrieval spot-checks
                                             # (>=5 query/expected pairs must all hit)
Requires running Qdrant (:6333) and Ollama (:11434).
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.memory.episodic import (
    EPISODES_COLLECTION,
    count_episodes,
    find_similar_incidents,
    record_episode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logger = logging.getLogger("seed_episodes")

# Fictional resolved incidents — all content synthetic. Bodies are written in the
# voice of a ticket so retrieval behaves like production write-backs.
SEED_EPISODES: list[dict] = [
    {
        "id": "T-SEED-001",
        "customer_id": "CUST-1001",
        "subject": "Onboarding kit never arrived - tracking shows nothing",
        "body": (
            "I ordered the onboarding kit three weeks ago and the carrier still shows no "
            "movement at all. I think the parcel is simply lost. I want my money back."
        ),
        "resolution": (
            "Apologized; confirmed with the carrier that the shipment was lost in transit; "
            "issued a full refund of $89.99 and offered a free reship if she still wants the kit."
        ),
        "category": "shipping_issue",
    },
    {
        "id": "T-SEED-002",
        "customer_id": "CUST-1004",
        "subject": "Charged twice for monthly plan",
        "body": (
            "My card shows two identical charges of $12.99 for this month's subscription but I "
            "only signed up once. Please fix this and return the extra money."
        ),
        "resolution": (
            "Verified the duplicate capture in the billing log; refunded one $12.99 charge "
            "(3-5 business days); confirmed only one active subscription remains."
        ),
        "category": "billing_dispute",
    },
    {
        "id": "T-SEED-003",
        "customer_id": "CUST-1003",
        "subject": "Want refund of annual plan fee",
        "body": (
            "We paid $149 for the annual plan two weeks ago and it doesn't fit our needs after "
            "an internal review. Requesting a full refund of the annual fee."
        ),
        "resolution": (
            "Refund of $149 exceeds the $50 autonomous limit; routed to supervisor approval "
            "per policy; approved next business day and refunded to the original card."
        ),
        "category": "billing_dispute",
    },
    {
        "id": "T-SEED-004",
        "customer_id": "CUST-1002",
        "subject": "Locked out - verification code never arrives",
        "body": (
            "I can't sign in. Every time I try, the 2FA text message never shows up on my "
            "phone. I'm completely locked out of my account."
        ),
        "resolution": (
            "Walked through backup-code sign-in; after identity verification re-enrolled MFA "
            "with a fresh number; confirmed successful login."
        ),
        "category": "account_access",
    },
    {
        "id": "T-SEED-005",
        "customer_id": "CUST-1001",
        "subject": "Kit arrived with broken parts",
        "body": (
            "The hardware kit finally arrived but several components are cracked and one is "
            "missing from the box. I need working equipment."
        ),
        "resolution": (
            "Shipped a replacement kit free of charge with expedited delivery; damaged unit "
            "did not need to be returned; missing component included in replacement."
        ),
        "category": "product_issue",
    },
    {
        "id": "T-SEED-006",
        "customer_id": "CUST-1002",
        "subject": "Cancel subscription and refund unused time",
        "body": (
            "Please cancel my plan entirely. I'd also like the money back for the days I "
            "won't be using since I paid until the end of the month."
        ),
        "resolution": (
            "Cancelled the subscription effective immediately per request; issued a prorated "
            "refund of $6.50 covering unused days; confirmation email sent."
        ),
        "category": "cancellation",
    },
    {
        "id": "T-SEED-007",
        "customer_id": "CUST-1004",
        "subject": "Package stuck in transit for ten days",
        "body": (
            "My order has been sitting in the same sorting facility for over a week and a "
            "half. The estimated delivery date keeps slipping. What is going on?"
        ),
        "resolution": (
            "Escalated with the carrier for a trace; dispatched an expedited reshipment as a "
            "precaution; issued a 10% discount code for the inconvenience."
        ),
        "category": "shipping_issue",
    },
]

# Curated rubric: each query must retrieve the expected episode within top-k=3.
# Queries deliberately use different wording than the seed bodies.
SPOT_CHECKS: list[tuple[str, str]] = [
    ("the tracking number hasn't updated in weeks, I think my parcel is gone", "T-SEED-001"),
    ("my bank statement shows two identical subscription charges this month", "T-SEED-002"),
    ("I paid one hundred forty nine dollars for the yearly plan and want it back", "T-SEED-003"),
    ("can't access my account, the two factor text message never arrives", "T-SEED-004"),
    ("my device came with cracked pieces inside the box", "T-SEED-005"),
    ("please terminate my plan and give money back for time not used", "T-SEED-006"),
    ("shipment delayed almost two weeks sitting at the same facility", "T-SEED-007"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="run curated retrieval spot-checks after seeding")
    parser.add_argument("-k", type=int, default=3, help="top-k for spot-checks (default 3)")
    args = parser.parse_args()

    before = count_episodes()
    for ep in SEED_EPISODES:
        record_episode(ep, ep["resolution"])
    after = count_episodes()
    logger.info("collection=%s before=%d after=%d seeded=%d", EPISODES_COLLECTION, before, after, len(SEED_EPISODES))

    if not args.verify:
        return

    misses = []
    for query, expected_id in SPOT_CHECKS:
        hits = find_similar_incidents(query, k=args.k)
        top_ids = [h["ticket_id"] for h in hits]
        if expected_id in top_ids:
            best = hits[[h["ticket_id"] for h in hits].index(expected_id)]
            logger.info("HIT  %-12s <- %s (score %.3f)", expected_id, query[:60], best["score"])
        else:
            misses.append((query, expected_id, top_ids))
            logger.error("MISS %-12s query=%r got=%s", expected_id, query[:60], top_ids)
    total, hit = len(SPOT_CHECKS), len(SPOT_CHECKS) - len(misses)
    print(f"\nspot-checks: {hit}/{total} retrieved within top-{args.k}")
    if misses:
        sys.exit(1)


if __name__ == "__main__":
    main()
