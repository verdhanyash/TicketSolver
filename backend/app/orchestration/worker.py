"""Background investigation jobs (FR-4 async processing, NFR Scalability).

Submission enqueues work onto a small thread pool; the orchestrator runs off the
request path and completion is broadcast over the dashboard WebSocket (feeds M9/M10).
Jobs are tracked in-memory per process — demo scale; a durable queue would arrive
with real multi-instance hosting.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"


@dataclass
class Job:
    """State of one background investigation."""

    ticket_id: str
    status: str = QUEUED
    outcome: str | None = None  # resolved | escalated once finished
    proposal: str | None = None
    attempts: int | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ticketsolver-job")


def enqueue_investigation(ticket_id: str, subject: str, body: str, customer_id: str) -> Job:
    """Queue one ticket for orchestrated processing (idempotent while in flight).

    Returns a snapshot of the job state — the worker mutates the stored object
    concurrently, so callers always get an immutable view.
    """
    with _JOBS_LOCK:
        existing = _JOBS.get(ticket_id)
        if existing is not None and existing.status in (QUEUED, RUNNING):
            return Job(**vars(existing))
        job = Job(ticket_id=ticket_id)
        _JOBS[ticket_id] = job
    _executor.submit(_process, ticket_id, subject, body, customer_id)
    logger.info("job queued ticket=%s", ticket_id)
    return Job(**vars(job))


def get_job(ticket_id: str) -> Job | None:
    """Snapshot of one job's state, or None if never enqueued."""
    with _JOBS_LOCK:
        job = _JOBS.get(ticket_id)
        return Job(**vars(job)) if job is not None else None


def _process(ticket_id: str, subject: str, body: str, customer_id: str) -> None:
    from app.api.websocket import broadcast_from_thread
    from app.orchestration.graph import process_ticket

    with _JOBS_LOCK:
        _JOBS[ticket_id].status = RUNNING
    try:
        result = process_ticket(ticket_id, subject, body, customer_id)
    except Exception as exc:  # a crashed run must still be visible, not silent
        logger.exception("investigation job failed ticket=%s", ticket_id)
        with _JOBS_LOCK:
            job = _JOBS[ticket_id]
            job.status = FAILED
            job.error = str(exc)
        broadcast_from_thread(
            {"type": "ticket_failed", "ticket_id": ticket_id, "error": str(exc)}
        )
        return

    with _JOBS_LOCK:
        job = _JOBS[ticket_id]
        job.status = COMPLETED
        job.outcome = result["outcome"]
        job.proposal = result["proposal"]
        job.attempts = result["attempts"]
    logger.info(
        "job %s ticket=%s attempts=%s", result["outcome"], ticket_id, result["attempts"]
    )
    broadcast_from_thread(
        {
            "type": "ticket_processed",
            "ticket_id": ticket_id,
            "outcome": result["outcome"],
            "attempts": result["attempts"],
        }
    )
