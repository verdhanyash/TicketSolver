"""Episodic memory (FR-7): past resolved incidents as vectors in Qdrant.

Before the agent proposes anything, similar past cases are retrieved and surfaced in
its context; after a resolution, the incident is written back so future tickets can
learn from it. Episodes live in their own Qdrant collection (`episodes`) — distinct
from the KB chunks in M1 — embedded with the same Ollama-served `nomic-embed-text`
(dim 768) used for KB RAG.

Point IDs are deterministic per ticket (`uuid5` of the ticket id), so re-recording a
ticket's episode replaces it instead of duplicating. All payload (de)serialization goes
through pure helpers so callers never touch raw dicts, and tests can round-trip them
without a vector store.
"""

from __future__ import annotations

import logging
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.vector.client import get_qdrant
from app.vector.kb import embed_texts

logger = logging.getLogger(__name__)

EPISODES_COLLECTION = "episodes"
_MAX_EMBED_CHARS = 4000  # subject+body excerpt ceiling; payloads stay well under this


# --- Pure serialization helpers -------------------------------------------------
def episode_to_payload(
    *,
    ticket_id: str,
    customer_id: str | None,
    subject: str,
    body: str,
    resolution: str,
    category: str | None = None,
) -> dict:
    """Build the Qdrant payload stored alongside an episode's vector."""
    return {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "subject": subject,
        "body": body[:2000],
        "resolution": resolution[:2000],
        "category": category,
    }


def episode_from_payload(payload: dict) -> dict:
    """Normalize a stored payload back to a clean dict (drops unknown keys)."""
    return {
        "ticket_id": payload.get("ticket_id", ""),
        "customer_id": payload.get("customer_id"),
        "subject": payload.get("subject", ""),
        "body": payload.get("body", ""),
        "resolution": payload.get("resolution", ""),
        "category": payload.get("category"),
    }


def format_similar_incidents(cases: list[dict]) -> str:
    """Render retrieval hits into the agent-prompt block; "" when nothing found."""
    if not cases:
        return ""
    lines = ["SIMILAR PAST INCIDENTS (resolved cases from episodic memory):"]
    for i, case in enumerate(cases, start=1):
        lines.append(
            f"{i}. \"{case['subject']}\" -> Resolution: {case['resolution']} "
            f"(similarity {case['score']:.2f})"
        )
    return "\n".join(lines)


def episode_point_id(ticket_id: str) -> str:
    """Deterministic per ticket: same ticket re-recorded replaces its episode."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ticketsolver:episode:{ticket_id}"))


# --- Store operations -----------------------------------------------------------
def _ticket_field(ticket, name: str, default=None):
    """Read a field off a Ticket ORM row or a plain dict."""
    if isinstance(ticket, dict):
        return ticket.get(name, default)
    return getattr(ticket, name, default)


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(EPISODES_COLLECTION):
        return
    client.create_collection(
        collection_name=EPISODES_COLLECTION,
        vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
    )
    logger.info("created Qdrant collection %s (dim=%d)", EPISODES_COLLECTION, settings.embedding_dim)


def record_episode(ticket, resolution: str, *, client: QdrantClient | None = None) -> str:
    """Embed and store one resolved incident; returns its point id.

    `ticket` may be a Ticket ORM row or any mapping with the usual fields
    (id/customer_id/subject/body/category).
    """
    ticket_id = str(_ticket_field(ticket, "id") or "")
    customer_id = _ticket_field(ticket, "customer_id")
    subject = str(_ticket_field(ticket, "subject") or "")
    body = str(_ticket_field(ticket, "body") or "")
    category = _ticket_field(ticket, "category")

    client = client or get_qdrant()
    ensure_collection(client)
    [vector] = embed_texts([f"{subject}\n{body}"[:_MAX_EMBED_CHARS]])
    point_id = episode_point_id(ticket_id)
    start = time.perf_counter()
    client.upsert(
        collection_name=EPISODES_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload=episode_to_payload(
                    ticket_id=ticket_id,
                    customer_id=str(customer_id) if customer_id else None,
                    subject=subject,
                    body=body,
                    resolution=resolution,
                    category=str(category) if category else None,
                ),
            )
        ],
    )
    logger.info(
        "episodic.record collection=%s ticket=%s latency_ms=%d",
        EPISODES_COLLECTION,
        ticket_id,
        int((time.perf_counter() - start) * 1000),
    )
    return point_id


def find_similar_incidents(
    text: str,
    k: int = 3,
    *,
    exclude_ticket_id: str | None = None,
    client: QdrantClient | None = None,
) -> list[dict]:
    """Return up to k episodes similar to `text`, as [{...payload fields..., score}].

    `exclude_ticket_id` keeps a ticket's own earlier episode out of its context when
    a ticket is re-investigated.
    """
    client = client or get_qdrant()
    ensure_collection(client)
    start = time.perf_counter()
    [vector] = embed_texts([text[:_MAX_EMBED_CHARS]])
    result = client.query_points(
        collection_name=EPISODES_COLLECTION,
        query=vector,
        limit=k + (1 if exclude_ticket_id else 0),  # over-fetch to survive exclusion
        with_payload=True,
    )
    hits = []
    for p in result.points:
        if exclude_ticket_id and p.payload.get("ticket_id") == exclude_ticket_id:
            continue
        hits.append({**episode_from_payload(p.payload), "score": float(p.score)})
        if len(hits) >= k:
            break
    logger.info(
        "episodic.search collection=%s top_k=%d hits=%d total_latency_ms=%d",
        EPISODES_COLLECTION,
        k,
        len(hits),
        int((time.perf_counter() - start) * 1000),
    )
    return hits


def count_episodes(*, client: QdrantClient | None = None) -> int:
    """Number of stored episodes (write-back verification / seeding checks)."""
    client = client or get_qdrant()
    ensure_collection(client)
    return client.count(collection_name=EPISODES_COLLECTION, exact=True).count
