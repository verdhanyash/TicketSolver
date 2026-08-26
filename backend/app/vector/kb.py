"""Knowledge-base retrieval over Qdrant, embedded via local Ollama.

Embeddings come from Ollama serving nomic-embed-text over HTTP (user decision,
2026-08-23) — not NIM-hosted and not sentence-transformers. Every embedding call
(ingestion and query time) is timed and logged so latency stays observable.
"""

from __future__ import annotations

import logging
import time
import uuid

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings

logger = logging.getLogger(__name__)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts via Ollama's /api/embed endpoint, logging elapsed time."""
    start = time.perf_counter()
    resp = httpx.post(
        f"{settings.ollama_base_url}/api/embed",
        json={"model": settings.ollama_embed_model, "input": texts},
        timeout=120.0,
    )
    resp.raise_for_status()
    embeddings = resp.json()["embeddings"]
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "embed call: model=%s texts=%d dim=%d latency_ms=%d",
        settings.ollama_embed_model,
        len(texts),
        len(embeddings[0]) if embeddings else -1,
        elapsed_ms,
    )
    if embeddings and len(embeddings[0]) != settings.embedding_dim:
        raise RuntimeError(
            f"Embedding dim {len(embeddings[0])} != configured {settings.embedding_dim}; "
            "update EMBEDDING_DIM and recreate the Qdrant collection."
        )
    return embeddings


def ensure_collection(client: QdrantClient) -> None:
    """Create the KB collection at the verified dimension if absent."""
    if client.collection_exists(settings.qdrant_collection):
        return
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
    )
    logger.info("created Qdrant collection %s (dim=%d)", settings.qdrant_collection, settings.embedding_dim)


def point_id_for(source: str, section: str) -> str:
    """Deterministic ID so re-ingesting a doc replaces its chunks cleanly."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ticketsolver:{source}:{section}"))


def upsert_chunks(client: QdrantClient, chunks: list[dict]) -> int:
    """Embed and upsert {id?, text, source, section} chunks; returns count written."""
    if not chunks:
        return 0
    vectors = embed_texts([c["text"] for c in chunks])
    points = [
        PointStruct(
            id=c.get("id") or point_id_for(c["source"], c["section"]),
            vector=vector,
            payload={"text": c["text"], "source": c["source"], "section": c["section"]},
        )
        for c, vector in zip(chunks, vectors, strict=True)
    ]
    start = time.perf_counter()
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info(
        "qdrant.upsert collection=%s points=%d latency_ms=%d",
        settings.qdrant_collection,
        len(points),
        int((time.perf_counter() - start) * 1000),
    )
    return len(points)


def search_kb(query: str, top_k: int = 4) -> list[dict]:
    """Retrieve top-k KB chunks for a query. Returns [{text, source, section, score}]."""
    start = time.perf_counter()
    [vector] = embed_texts([query])
    from app.vector.client import get_qdrant  # local import avoids engine work on import

    client = get_qdrant()
    ensure_collection(client)
    result = client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    hits = [
        {
            "text": p.payload.get("text", ""),
            "source": p.payload.get("source", ""),
            "section": p.payload.get("section", ""),
            "score": p.score,
        }
        for p in result.points
    ]
    logger.info(
        "kb.search collection=%s top_k=%d hits=%d total_latency_ms=%d",
        settings.qdrant_collection,
        top_k,
        len(hits),
        int((time.perf_counter() - start) * 1000),
    )
    return hits
