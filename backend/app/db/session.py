"""SQLAlchemy engine and session factory (PostgreSQL).

The engine is created lazily so the app can be imported without a live DB (e.g. in tests
or during scaffolding). `get_session_factory()` returns a `sessionmaker`; construct
sessions from it (`session = get_session_factory()()`), or hand it to helpers like
`TraceWriter` that expect the factory itself.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not set (see .env.example).")
    return create_engine(settings.database_url, pool_pre_ping=True, future=True)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return the shared sessionmaker (construct Sessions by calling it)."""
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a scoped session."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
