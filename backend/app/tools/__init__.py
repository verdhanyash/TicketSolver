"""Simulated external-system tools for the Investigator agent (SRS FR-2).

Deterministic, in-memory stand-ins for real account/order systems so agent behaviour
is reproducible in tests and evals without touching live data or PII.
"""

from app.tools.accounts import lookup_account
from app.tools.orders import get_order_status

__all__ = ["get_order_status", "lookup_account"]
