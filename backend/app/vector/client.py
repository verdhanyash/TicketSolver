"""Qdrant client factory (FR-7 episodic memory + KB RAG).

Lazily constructs a client from settings so the app imports without a running Qdrant.
"""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient

from app.config import settings


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )
