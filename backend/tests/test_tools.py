"""Tests for the simulated Investigator tools (SRS FR-2).

Covers the lookup contract (found / not-found / copy semantics) for both the
account and order stores, plus sanity checks on the seed data itself.
"""

from __future__ import annotations

from app.tools import get_order_status, lookup_account
from app.tools.accounts import ACCOUNTS
from app.tools.orders import ORDERS


def test_known_account_found_with_correct_fields() -> None:
    record = lookup_account("CUST-1001")
    assert record is not None
    assert set(record) == {
        "customer_id",
        "name",
        "email",
        "plan_tier",
        "account_age_days",
        "mfa_enabled",
        "last_login",
    }
    assert record["customer_id"] == "CUST-1001"
    assert isinstance(record["account_age_days"], int)
    assert isinstance(record["mfa_enabled"], bool)
    assert record["email"].endswith("@example.com")


def test_unknown_account_returns_none() -> None:
    assert lookup_account("CUST-9999") is None


def test_account_result_is_a_copy() -> None:
    record = lookup_account("CUST-1002")
    assert record is not None
    record["plan_tier"] = "enterprise"
    record["name"] = "tampered"
    assert ACCOUNTS["CUST-1002"]["plan_tier"] == "free"
    assert ACCOUNTS["CUST-1002"]["name"] == "Devon Blake"


def test_known_order_found_with_correct_fields() -> None:
    record = get_order_status("ORD-5003")
    assert record is not None
    assert set(record) == {
        "order_id",
        "customer_id",
        "item",
        "amount",
        "status",
        "placed_at",
        "delivered_at",
    }
    assert record["order_id"] == "ORD-5003"
    assert record["customer_id"] in ACCOUNTS
    assert isinstance(record["amount"], float)
    assert record["status"] in {"delivered", "in_transit", "lost", "delayed"}
    assert record["delivered_at"] is None


def test_unknown_order_returns_none() -> None:
    assert get_order_status("ORD-9999") is None


def test_order_result_is_a_copy() -> None:
    record = get_order_status("ORD-5005")
    assert record is not None
    record["status"] = "lost"
    record["amount"] = 0.01
    assert ORDERS["ORD-5005"]["status"] == "delivered"
    assert ORDERS["ORD-5005"]["amount"] == 129.00


def test_seed_data_has_expected_shape() -> None:
    # Seed guarantees used by downstream logic (refund threshold, escalation paths).
    amounts = [order["amount"] for order in ORDERS.values()]
    assert any(amount > 50.00 for amount in amounts)
    assert any(amount < 50.00 for amount in amounts)

    statuses = {order["status"] for order in ORDERS.values()}
    assert {"lost", "delayed"} <= statuses

    assert len(ACCOUNTS) == 4
    assert len(ORDERS) == 6
    assert all(order["customer_id"] in ACCOUNTS for order in ORDERS.values())
