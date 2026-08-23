"""SQLAlchemy ORM models (PostgreSQL).

Declarative base only for now. Ticket, CustomerMemory, Trace, and Approval tables (FR-6,
FR-9, FR-10) are defined in a later session — schema changes require sign-off (see rules.md).
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Models to come:  Ticket, CustomerMemory, Trace, Approval.
