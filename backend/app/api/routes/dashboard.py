"""Dashboard aggregation endpoint (FR-16): one payload feeding all four panels.

Combines ticket outcomes, LLM token usage from traces, eval-harness scores (when
Module 8 has produced a report), and the approval-queue snapshot into a single JSON
document so the frontend needs one request per refresh. Aggregation is plain Python
over full selects — portfolio scale. Every section tolerates an empty database
(zeros / empty lists / `available: false`) so the endpoint never 500s on fresh data.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Approval, Ticket, TraceStep
from app.db.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]

# Module 8 drops its latest run here; until then the quality section reports
# available=false and the panel shows its "not run yet" state.
EVAL_REPORT_PATH = Path(__file__).resolve().parents[3] / "eval" / "report.json"

SERIES_DAYS = 14

# Outcome buckets the volume panel always shows (zeros included so the shape is stable).
STATUS_KEYS = ("auto_resolved", "resolved", "pending_approval", "escalated", "rejected", "open")
# Buckets that get their own chart segment; anything else (e.g. `open`) folds into
# `in_progress`.
DAILY_OUTCOME_KEYS = ("auto_resolved", "resolved", "pending_approval", "escalated", "rejected")

# Token usage rides inside llm_call trace payloads when callers persist it
# (nim_client always has it on the response; persistence is best-effort). Look for
# direct fields or a nested usage dict; absence degrades to graceful zeros.
_USAGE_DICT_KEYS = ("usage", "token_usage", "tokens")
_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


# --- Response models -----------------------------------------------------------------
class DailyPoint(BaseModel):
    date: str  # YYYY-MM-DD
    total: int
    auto_resolved: int
    resolved: int
    pending_approval: int
    escalated: int
    rejected: int
    in_progress: int


class VolumeOut(BaseModel):
    total_tickets: int
    by_status: dict[str, int]
    daily: list[DailyPoint]
    auto_resolution_rate_pct: float


class CostOut(BaseModel):
    has_usage_data: bool
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class QualityOut(BaseModel):
    available: bool
    report: dict[str, Any] | None = None
    error: str | None = None  # set when a report exists but could not be served


class QueueOut(BaseModel):
    pending_count: int
    oldest_pending_age_minutes: int | None


class DashboardSummaryOut(BaseModel):
    generated_at: datetime
    volume: VolumeOut
    cost: CostOut
    quality: QualityOut
    queue: QueueOut


# --- Helpers -------------------------------------------------------------------------
def _as_utc(value: datetime | None) -> datetime | None:
    """Normalize DB datetimes to aware UTC (SQLite hands back naive values)."""
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _extract_tokens(payload: Any) -> dict[str, int]:
    """Best-effort token counts from one trace payload (direct or nested usage dict)."""
    if not isinstance(payload, dict):
        return {}
    sources: list[dict] = [payload]
    sources.extend(
        nested for key in _USAGE_DICT_KEYS if isinstance((nested := payload.get(key)), dict)
    )
    counts: dict[str, int] = {}
    for source in sources:
        for field in _TOKEN_FIELDS:
            if field in counts:
                continue
            value = source.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                counts[field] = int(value)
    return counts


def _volume_section(tickets: list[Ticket]) -> VolumeOut:
    by_status: dict[str, int] = dict.fromkeys(STATUS_KEYS, 0)
    auto_resolved = 0
    for ticket in tickets:
        status = ticket.status or "open"
        by_status[status] = by_status.get(status, 0) + 1
        if status == "auto_resolved":
            auto_resolved += 1

    today = datetime.now(UTC).date()
    days = [today - timedelta(days=offset) for offset in range(SERIES_DAYS - 1, -1, -1)]
    series: dict[str, dict] = {
        day.isoformat(): {
            "date": day.isoformat(),
            "total": 0,
            **dict.fromkeys(DAILY_OUTCOME_KEYS, 0),
            "in_progress": 0,
        }
        for day in days
    }
    for ticket in tickets:
        created = _as_utc(ticket.created_at)
        point = series.get(created.date().isoformat()) if created else None
        if point is None:  # outside the window (or no timestamp) — totals still count it
            continue
        point["total"] += 1
        point[ticket.status if ticket.status in DAILY_OUTCOME_KEYS else "in_progress"] += 1

    return VolumeOut(
        total_tickets=len(tickets),
        by_status=by_status,
        daily=[series[day.isoformat()] for day in days],
        auto_resolution_rate_pct=(
            round(auto_resolved / len(tickets) * 100, 1) if tickets else 0.0
        ),
    )


def _cost_section(session: Session) -> CostOut:
    steps = session.scalars(select(TraceStep).where(TraceStep.step_type == "llm_call")).all()
    sums: dict[str, int] = dict.fromkeys(_TOKEN_FIELDS, 0)
    for step in steps:
        # output_payload wins over input_payload when both record a field.
        counts = {**_extract_tokens(step.input_payload), **_extract_tokens(step.output_payload)}
        # Per-call total: explicit when recorded, otherwise derived from its parts —
        # so mixed explicit/implicit rows neither double-count nor undercount.
        counts.setdefault(
            "total_tokens",
            counts.get("prompt_tokens", 0) + counts.get("completion_tokens", 0),
        )
        for field in _TOKEN_FIELDS:
            sums[field] += counts.get(field, 0)
    return CostOut(
        has_usage_data=any(sums[field] > 0 for field in _TOKEN_FIELDS),
        llm_calls=len(steps),
        **sums,
    )


def _quality_section() -> QualityOut:
    if not EVAL_REPORT_PATH.exists():
        return QualityOut(available=False)
    try:
        report = json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("eval report at %s unreadable: %s", EVAL_REPORT_PATH, exc)
        return QualityOut(available=False, error=f"Report file unreadable: {exc}")
    if not isinstance(report, dict):
        return QualityOut(available=False, error="Report file did not contain an object")
    return QualityOut(available=True, report=report)


def _queue_section(session: Session, now: datetime) -> QueueOut:
    pending = session.scalars(select(Approval).where(Approval.status == "pending")).all()
    created_times = [t for t in (_as_utc(a.created_at) for a in pending) if t is not None]
    oldest_minutes = (
        max(0, int((now - min(created_times)).total_seconds() // 60)) if created_times else None
    )
    return QueueOut(pending_count=len(pending), oldest_pending_age_minutes=oldest_minutes)


@router.get("/summary", response_model=DashboardSummaryOut)
def dashboard_summary(session: SessionDep) -> DashboardSummaryOut:
    now = datetime.now(UTC)
    tickets = list(session.scalars(select(Ticket)).all())
    return DashboardSummaryOut(
        generated_at=now,
        volume=_volume_section(tickets),
        cost=_cost_section(session),
        quality=_quality_section(),
        queue=_queue_section(session, now),
    )
