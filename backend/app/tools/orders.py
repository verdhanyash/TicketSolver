"""Simulated order-status tool (SRS FR-2).

An in-memory order-management stand-in covering the statuses that drive agent
decisions (delivered / in_transit / lost / delayed). Lookups return copies so
callers can never mutate the store.
"""

from __future__ import annotations

# Synthetic orders referencing accounts from app.tools.accounts — fictional items only.
ORDERS: dict[str, dict] = {
    "ORD-5001": {
        "order_id": "ORD-5001",
        "customer_id": "CUST-1001",
        "item": "Wireless keyboard and mouse combo",
        "amount": 79.99,
        "status": "delivered",
        "placed_at": "2026-07-28",
        "delivered_at": "2026-08-02",
    },
    "ORD-5002": {
        "order_id": "ORD-5002",
        "customer_id": "CUST-1002",
        "item": "USB-C charging cable (2 m)",
        "amount": 18.50,
        "status": "in_transit",
        "placed_at": "2026-08-18",
        "delivered_at": None,
    },
    "ORD-5003": {
        "order_id": "ORD-5003",
        "customer_id": "CUST-1003",
        "item": "27-inch 4K monitor",
        "amount": 349.00,
        "status": "lost",
        "placed_at": "2026-08-05",
        "delivered_at": None,
    },
    "ORD-5004": {
        "order_id": "ORD-5004",
        "customer_id": "CUST-1003",
        "item": "Ergonomic laptop stand",
        "amount": 42.25,
        "status": "delayed",
        "placed_at": "2026-08-12",
        "delivered_at": None,
    },
    "ORD-5005": {
        "order_id": "ORD-5005",
        "customer_id": "CUST-1004",
        "item": "Noise-cancelling headphones",
        "amount": 129.00,
        "status": "delivered",
        "placed_at": "2026-06-30",
        "delivered_at": "2026-07-04",
    },
    "ORD-5006": {
        "order_id": "ORD-5006",
        "customer_id": "CUST-1001",
        "item": "Mechanical keycap set",
        "amount": 64.95,
        "status": "delayed",
        "placed_at": "2026-08-15",
        "delivered_at": None,
    },
}


def get_order_status(order_id: str) -> dict | None:
    """Return a copy of the order record for `order_id`, or None if unknown."""
    record = ORDERS.get(order_id)
    return dict(record) if record is not None else None
