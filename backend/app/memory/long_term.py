"""Long-term customer memory (FR-6).

Per-customer facts (plan tier, account age, ordered notes) persisted in Postgres and
loaded into the agent's context at session start so resolutions respect customer
history. `format_customer_memory` renders a record into the prompt block; empty or
missing memory renders as an empty string so callers can omit it gracefully.

Session-factory injection: every public function takes an optional `session_factory`;
when omitted, the app-wide factory is resolved lazily so importing this module never
opens a connection (and tests can pass a SQLite-backed sessionmaker).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session, sessionmaker

from app.db.models import CustomerMemory
from app.db.session import get_session_factory


@dataclass
class CustomerMemoryRecord:
    """One customer's accumulated facts. `notes` keeps insertion order."""

    customer_id: str
    plan_tier: str | None = None
    account_age_days: int | None = None
    notes: list[str] = field(default_factory=list)


def _to_record(row: CustomerMemory) -> CustomerMemoryRecord:
    return CustomerMemoryRecord(
        customer_id=row.customer_id,
        plan_tier=row.plan_tier,
        account_age_days=row.account_age_days,
        notes=list(row.notes or []),
    )


def get_customer_memory(
    customer_id: str, *, session_factory: sessionmaker[Session] | None = None
) -> CustomerMemoryRecord | None:
    """Return the customer's facts, or None when nothing is stored."""
    factory = session_factory or get_session_factory()
    with factory() as session:
        row = session.get(CustomerMemory, customer_id)
        return _to_record(row) if row is not None else None


def upsert_customer_memory(
    record: CustomerMemoryRecord, *, session_factory: sessionmaker[Session] | None = None
) -> None:
    """Insert or update the customer's facts (idempotent on customer_id)."""
    factory = session_factory or get_session_factory()
    with factory() as session:
        row = session.get(CustomerMemory, record.customer_id)
        if row is None:
            row = CustomerMemory(customer_id=record.customer_id)
            session.add(row)
        row.plan_tier = record.plan_tier
        row.account_age_days = record.account_age_days
        row.notes = list(record.notes)
        session.commit()


def format_customer_memory(record: CustomerMemoryRecord | None) -> str:
    """Render memory rows into the agent-prompt context block.

    Empty/None memory → "" (graceful omission); populated → facts in fixed order
    (plan tier, account age, then stored note order).
    """
    if record is None:
        return ""
    lines: list[str] = []
    if record.plan_tier:
        lines.append(f"- Plan tier: {record.plan_tier}")
    if record.account_age_days is not None:
        lines.append(f"- Account age: {record.account_age_days} days")
    for i, note in enumerate(record.notes or [], start=1):
        lines.append(f"- Note {i}: {note}")
    if not lines:
        return ""
    header = f"KNOWN CUSTOMER FACTS ({record.customer_id}, from long-term memory):"
    return "\n".join([header, *lines])
