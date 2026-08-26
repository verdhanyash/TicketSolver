"""Create all tables (dev convenience; Alembic migrations take over later)."""

from __future__ import annotations

from app.db.models import Base
from app.db.session import get_engine


def main() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("tables:", ", ".join(sorted(Base.metadata.tables)))


if __name__ == "__main__":
    main()
