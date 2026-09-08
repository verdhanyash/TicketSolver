"""Orchestrator (FR-4): a stateful LangGraph wrapping the Investigator agent.

Not a fixed pipeline — after each investigation the orchestrator *decides* the next
move from the outcome:

    investigate ──▶ decide ──▶ guardrail ──▶ resolve          (allowed action)
                      │            └──▶ await_review     (blocked, FR-3)
                      │──▶ retry    (low confidence / empty, attempts remain)
                      └──▶ escalate (out of attempts; terminal safety state)

The guardrail check (FR-3, hard-coded policy) sits between proposal and execution:
nothing autonomous happens before it, and blocked actions land in the HITL queue with
a persisted Approval row. Retry re-enters `investigate` with an explicit nudge.
Short-term session memory (FR-5) rides alongside: stored conversation turns are handed
to every investigation. Escalation is terminal, never a crash. Every transition is
traced (FR-10) and mirrored on the ticket row's `status`.

Confidence is LLM-self-reported for now (the `CONFIDENCE:` line of the investigator's
output contract); calibration replaces this seam in M7. ML classification (FR-1) runs
ahead of the graph, filling the ticket's category/severity before any routing happens.
"""

from __future__ import annotations

import logging
import operator
import re
import time
from typing import Annotated, Any, TypedDict

from langgraph.constants import END
from langgraph.graph import StateGraph

from app.config import settings
from app.db.models import Ticket
from app.db.session import get_session_factory

logger = logging.getLogger(__name__)

RESOLVE = "resolve"
RETRY = "retry"
ESCALATE = "escalate"
PENDING_APPROVAL = "pending_approval"

_RETRY_NUDGE = (
    "[orchestrator] Your previous answer was not confident enough to act on. "
    "Re-examine the evidence (use tools if you have not), then answer again with all "
    "four sections."
)


class OrchestratorState(TypedDict, total=False):
    ticket_id: str
    customer_id: str
    subject: str
    body: str
    attempt: int  # 1-based count of investigations run
    proposal: str
    confidence: float
    outcome: str  # set by terminal-path nodes: resolve | escalate | pending_approval
    approval_id: str  # set when a guardrail block queues the action for review
    session_turns: list[str]  # FR-5: prior conversation, fixed for the run
    nudges: Annotated[list[str], operator.add]  # retry guidance, grows each loop
    selected_model: str  # FR-14: cheap vs strong NIM model tier
    complexity_label: str  # FR-14: simple vs complex


# --- Pure decision logic (unit-test target) --------------------------------------
_CONFIDENCE_RE = re.compile(r"CONFIDENCE\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def extract_confidence(proposal: str, default: float = 1.0) -> float:
    """Parse the `CONFIDENCE:` meta-line of the output contract (clamped to [0,1])."""
    match = _CONFIDENCE_RE.search(proposal or "")
    if not match:
        return default
    return min(max(float(match.group(1)), 0.0), 1.0)


def strip_confidence(raw: str) -> tuple[str, float]:
    """Split a raw answer into (proposal without the meta-line, parsed confidence)."""
    confidence = extract_confidence(raw)
    cleaned = _CONFIDENCE_RE.sub("", raw or "").strip()
    return cleaned, confidence


def decide_next(
    *,
    attempt: int,
    confidence: float,
    proposal: str,
    max_attempts: int | None = None,
    threshold: float | None = None,
) -> str:
    """Transition table: outcome so far → next orchestrator action.

    resolve   — non-empty proposal at/above the confidence threshold
    retry     — weak/empty outcome but attempts remain
    escalate  — weak/empty outcome and out of attempts (terminal safety state)
    """
    max_attempts = settings.orchestrator_max_attempts if max_attempts is None else max_attempts
    threshold = settings.confidence_threshold if threshold is None else threshold
    strong_enough = bool((proposal or "").strip()) and confidence >= threshold
    if strong_enough:
        return RESOLVE
    return RETRY if attempt < max_attempts else ESCALATE


# --- Node actions ------------------------------------------------------------------
def _writer():
    from app.core.tracing import TraceWriter

    return TraceWriter(get_session_factory())


def _run_investigation(state: OrchestratorState) -> str:
    """Investigator entry point. Module-level seam — patch target for tests."""
    from app.agents.investigator import run_investigation

    follow_ups = [*state.get("session_turns", []), *state.get("nudges", [])]
    model = state.get("selected_model") or settings.nim_model_strong
    return run_investigation(
        state["ticket_id"],
        state["subject"],
        state["body"],
        state["customer_id"],
        follow_ups=follow_ups,
        model=model,
    )


def _classify_ticket(subject: str, body: str):
    """FR-1 ML triage entry point. Module-level seam — patch target for tests.

    Returns a prediction object or None when the model is unavailable; never raises.
    """
    try:
        from app.ml.classifier import load_classifier

        return load_classifier().predict(subject, body)
    except Exception:
        logger.warning("ML classifier unavailable; skipping triage", exc_info=True)
        return None


def _record_episode(ticket: dict, resolution: str) -> str:
    """Episodic write-back (FR-7). Module-level seam — patch target for tests.

    Called only from the terminal resolution paths: auto-resolve here, human-approved
    decisions in the approval endpoint (FR-9). A proposal alone is never recorded.
    """
    from app.memory.episodic import record_episode

    return record_episode(ticket, resolution)


def _trace_decision(ticket_id: str, name: str, reasoning: str, payload: dict) -> None:
    try:
        _writer().log_step(
            ticket_id=ticket_id,
            step_type="decision",
            name=name,
            input_payload=payload,
            reasoning=reasoning,
        )
    except Exception:
        logger.exception("failed to trace orchestrator decision %s", name)


def _set_ticket_status(
    ticket_id: str, status: str, resolution: str | None, *, skip_if_decided: bool = False
) -> None:
    try:
        with get_session_factory()() as session:
            ticket = session.get(Ticket, ticket_id)
            if ticket is not None:
                # A reviewer may decide while the run is still winding down; never
                # clobber an outcome that was already recorded (FR-8/9 integrity).
                if skip_if_decided and ticket.status in ("resolved", "rejected", "auto_resolved"):
                    return
                ticket.status = status
                if resolution is not None:
                    ticket.proposed_resolution = resolution
                session.commit()
    except Exception:
        logger.exception("failed to set ticket %s status=%s", ticket_id, status)


def _persist_prediction(ticket_id: str, prediction) -> None:
    """Store FR-1 predictions on the ticket row. Own session; touches ONLY the
    category/severity columns — status/proposed_resolution belong to the graph."""
    with get_session_factory()() as session:
        ticket = session.get(Ticket, ticket_id)
        if ticket is not None:
            ticket.category = prediction.category
            ticket.severity = prediction.severity
            session.commit()


def _triage_ticket(ticket_id: str, subject: str, body: str) -> None:
    """Run FR-1 ML triage ahead of the routing table: predict, persist, and trace.

    Best-effort end to end — an untrained model or any storage failure is logged and
    swallowed; triage must never break orchestration (same guard as resolve_node's
    episodic write-back).
    """
    try:
        started = time.perf_counter()
        prediction = _classify_ticket(subject, body)
        if prediction is None:  # model unavailable: silent skip keeps pre-M6 behavior
            return
        _persist_prediction(ticket_id, prediction)
        _writer().log_step(
            ticket_id=ticket_id,
            step_type="ml_prediction",
            name="ticket_classification",
            output_payload={
                "category": prediction.category,
                "severity": prediction.severity,
                "category_confidence": round(prediction.category_confidence, 3),
                "severity_confidence": round(prediction.severity_confidence, 3),
            },
            reasoning="XGBoost triage from subject+body (FR-1)",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception:
        logger.exception("ML triage failed for ticket %s; orchestration continues", ticket_id)


def _route_ticket(ticket_id: str, subject: str, body: str) -> tuple[str, str]:
    """Run FR-14 complexity routing: pick cheap vs strong NIM tier, trace it.

    Best-effort: missing model or errors fall back to settings.nim_model_strong.
    """
    if not settings.router_enabled:
        return settings.nim_model_strong, "complex"
    category = ""
    severity = ""
    try:
        with get_session_factory()() as session:
            t = session.get(Ticket, ticket_id)
            if t is not None:
                category = t.category or ""
                severity = t.severity or ""
    except Exception:  # noqa: BLE001, S110
        pass

    try:
        from app.ml.router import load_router

        started = time.perf_counter()
        decision = load_router().route(subject, body, category, severity)
        latency_ms = int((time.perf_counter() - started) * 1000)
        _writer().log_step(
            ticket_id=ticket_id,
            step_type="ml_prediction",
            name="cost_routing",
            output_payload={
                "complexity_label": decision.complexity_label,
                "confidence": round(decision.confidence, 4),
                "use_strong": decision.use_strong,
                "selected_model": decision.selected_model,
                "rationale": decision.rationale,
            },
            reasoning=f"Complexity router (FR-14): {decision.complexity_label} -> {decision.selected_model} ({decision.rationale})",
            model=decision.selected_model,
            latency_ms=latency_ms,
        )
        return decision.selected_model, decision.complexity_label
    except Exception:
        logger.warning("Complexity router unavailable for ticket %s; using strong model", ticket_id, exc_info=True)
        return settings.nim_model_strong, "complex"


def _calibrated_threshold(
    *,
    attempt: int,
    proposal: str,
    confidence: float,
    base_threshold: float,
) -> tuple[float, float | None]:
    """FR-15 confidence calibration: map run observables to calibrated threshold.

    Returns (effective_threshold, p_correct_or_none).
    """
    if not settings.calibration_enabled:
        return base_threshold, None
    try:
        from app.ml.calibration import calibrated_threshold, load_calibration

        calibrator = load_calibration()
        p_correct = calibrator.predict_probability(
            attempt=attempt,
            proposal=proposal,
            confidence=confidence,
        )
        threshold = calibrated_threshold(p_correct, base_threshold)
        return threshold, p_correct
    except Exception:
        logger.warning("Confidence calibration unavailable; using base threshold", exc_info=True)
        return base_threshold, None


def investigate_node(state: OrchestratorState) -> dict:
    attempt = state["attempt"] + 1
    raw = _run_investigation(state)
    proposal, confidence = strip_confidence(raw)
    return {"attempt": attempt, "proposal": proposal, "confidence": confidence}


def decide_node(state: OrchestratorState) -> dict:
    """Pure decision point: pick the next action from the outcome and trace it."""
    base_thresh = settings.confidence_threshold
    threshold, p_correct = _calibrated_threshold(
        attempt=state["attempt"],
        proposal=state["proposal"],
        confidence=state["confidence"],
        base_threshold=base_thresh,
    )
    action = decide_next(
        attempt=state["attempt"],
        confidence=state["confidence"],
        proposal=state["proposal"],
        threshold=threshold,
    )
    payload = {
        "attempt": state["attempt"],
        "confidence": round(state["confidence"], 3),
        "threshold": threshold,
        "calibrated_threshold": threshold,
        "base_threshold": base_thresh,
    }
    if p_correct is not None:
        payload["p_correct"] = p_correct
    if action == RESOLVE:
        _trace_decision(
            state["ticket_id"],
            "orchestrator_resolve",
            f"resolved autonomously on attempt {state['attempt']} "
            f"(confidence {state['confidence']:.2f} >= threshold {threshold:.2f})",
            payload,
        )
        return {"outcome": RESOLVE}
    if action == ESCALATE:
        reason = (
            "no usable proposal"
            if not state["proposal"].strip()
            else f"confidence below threshold {threshold:.2f}"
        )
        _trace_decision(
            state["ticket_id"],
            "orchestrator_escalate",
            f"escalating after {state['attempt']} attempt(s): {reason}",
            payload,
        )
        return {"outcome": ESCALATE}
    _trace_decision(
        state["ticket_id"],
        "orchestrator_retry",
        f"attempt {state['attempt']} not actionable "
        f"(confidence {state['confidence']:.2f} < threshold {threshold:.2f}); retrying",
        payload,
    )
    return {}  # routing falls through to the retry node


def retry_node(state: OrchestratorState) -> dict:
    """Loop-back node: queue guidance so the next investigation tries differently."""
    return {"nudges": [_RETRY_NUDGE]}


def resolve_node(state: OrchestratorState) -> dict:
    """Terminal node: mark the ticket auto-resolved with its proposal.

    Closing FR-7's loop happens here, not at investigation time: only *resolved*
    incidents belong in episodic memory, and the guardrail sits between proposal
    and resolution. (Human-approved resolutions are recorded by the approval
    endpoint instead — FR-9.)
    """
    _set_ticket_status(state["ticket_id"], "auto_resolved", state["proposal"])
    try:
        point_id = _record_episode(
            {
                "id": state["ticket_id"],
                "customer_id": state["customer_id"],
                "subject": state["subject"],
                "body": state["body"],
            },
            state["proposal"],
        )
        _writer().log_step(
            ticket_id=state["ticket_id"],
            step_type="memory_writeback",
            name="episode_writeback",
            output_payload={"point_id": point_id},
        )
    except Exception:
        logger.exception("episodic write-back failed; resolution flow unaffected")
    return {"outcome": RESOLVE}


def escalate_node(state: OrchestratorState) -> dict:
    """Terminal safety state: never crash, hand over to human review (M5 queue)."""
    _set_ticket_status(state["ticket_id"], "escalated", state["proposal"] or None)
    return {"outcome": ESCALATE}


# --- Guardrail enforcement point (FR-3): between proposal and execution -----------
def _create_approval(ticket_id: str, action, decision) -> str:
    """Persist one guardrail-blocked action into the HITL queue. Returns approval id."""
    import uuid

    from app.db.models import Approval

    approval_id = uuid.uuid4().hex
    try:
        with get_session_factory()() as session:
            session.add(
                Approval(
                    id=approval_id,
                    ticket_id=ticket_id,
                    action_kind=action.kind,
                    amount_usd=action.amount_usd,
                    reason=decision.blocked_reason(),
                )
            )
            session.commit()
        try:
            from app.api.websocket import broadcast_from_thread

            broadcast_from_thread(
                {
                    "type": "approval_created",
                    "approval_id": approval_id,
                    "ticket_id": ticket_id,
                    "action_kind": action.kind,
                    "amount_usd": action.amount_usd,
                }
            )
        except Exception:
            logger.exception("approval WS broadcast failed")
    except Exception:
        logger.exception("failed to persist approval for ticket %s", ticket_id)
        raise
    return approval_id


def guardrail_node(state: OrchestratorState) -> dict:
    """Check the proposal against hard-coded policy; block routes to HITL review."""
    from app.guardrails.engine import check_proposal

    action, decision = check_proposal(state["proposal"])
    payload = {
        "action_kind": action.kind,
        "amount_usd": action.amount_usd,
        "rule": decision.rule,
        "allowed": decision.allowed,
    }
    if decision.allowed:
        _trace_decision(
            state["ticket_id"],
            "guardrail_check",
            f"allowed: {action.kind}"
            + (f" ${action.amount_usd:.2f}" if action.amount_usd is not None else ""),
            payload,
        )
        return {}
    _trace_decision(
        state["ticket_id"],
        "guardrail_block",
        decision.blocked_reason(),
        payload,
    )
    approval_id = _create_approval(state["ticket_id"], action, decision)
    return {"outcome": PENDING_APPROVAL, "approval_id": approval_id}


def await_review_node(state: OrchestratorState) -> dict:
    """Terminal node: ticket sits in the HITL queue until a reviewer decides."""
    _set_ticket_status(state["ticket_id"], "pending_approval", state["proposal"], skip_if_decided=True)
    return {"outcome": PENDING_APPROVAL}


def _route_after_decide(state: OrchestratorState) -> str:
    """decide's chosen action → graph node. Resolves pass the guardrail first (FR-3)."""
    action = state.get("outcome") or RETRY
    return "guardrail" if action == RESOLVE else action


def _route_after_guardrail(state: OrchestratorState) -> str:
    return "await_review" if state.get("outcome") == PENDING_APPROVAL else RESOLVE


def build_orchestrator():
    graph = StateGraph(OrchestratorState)
    graph.add_node("investigate", investigate_node)
    graph.add_node("decide", decide_node)
    graph.add_node("retry", retry_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("resolve", resolve_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("await_review", await_review_node)
    graph.set_entry_point("investigate")
    graph.add_edge("investigate", "decide")
    graph.add_conditional_edges(
        "decide", _route_after_decide, [RETRY, "guardrail", ESCALATE]
    )
    graph.add_edge(RETRY, "investigate")
    # FR-3: allowed proposals flow through resolve; blocked ones queue for review.
    graph.add_conditional_edges("guardrail", _route_after_guardrail, [RESOLVE, "await_review"])
    graph.add_edge("await_review", END)
    graph.add_edge(RESOLVE, END)
    graph.add_edge(ESCALATE, END)
    return graph.compile()


compiled_orchestrator = build_orchestrator()


def process_ticket(
    ticket_id: str,
    subject: str,
    body: str,
    customer_id: str,
    *,
    session_turns: list[str] | None = None,
) -> dict[str, Any]:
    """Entry point: run one ticket through the orchestrator.

    `session_turns` carries FR-5 short-term memory (earlier conversation turns).
    ML triage (FR-1) runs first and is best-effort: a missing model changes nothing.
    Returns {"outcome": "resolved"|"escalated"|"pending_approval",
             "proposal": str, "attempts": int}.
    """
    _triage_ticket(ticket_id, subject, body)
    selected_model, complexity_label = _route_ticket(ticket_id, subject, body)
    result = compiled_orchestrator.invoke(
        {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "subject": subject,
            "body": body,
            "attempt": 0,
            "proposal": "",
            "confidence": 0.0,
            "outcome": "",
            "session_turns": list(session_turns or []),
            "nudges": [],
            "selected_model": selected_model,
            "complexity_label": complexity_label,
        }
    )
    # Public vocabulary is past tense ("resolved"/"escalated"); graph actions are not.
    outcome = {
        "resolve": "resolved",
        "escalate": "escalated",
        PENDING_APPROVAL: PENDING_APPROVAL,
    }.get(result.get("outcome", ""), "escalated")
    return {
        "outcome": outcome,
        "proposal": result.get("proposal", ""),
        "attempts": result.get("attempt", 0),
    }


# --- FR-5 short-term session context --------------------------------------------
# Conversation-scoped turns per open ticket, in-memory (demo scale); FR-5 scopes
# short-term memory to in-context/session storage, unlike long-term facts (M2).
SESSION_CONTEXTS: dict[str, list[str]] = {}


def record_session_turn(ticket_id: str, speaker: str, text: str) -> list[str]:
    """Append one conversation turn to the ticket's short-term context."""
    SESSION_CONTEXTS.setdefault(ticket_id, []).append(f"{speaker}: {text}")
    return get_session_context(ticket_id)


def get_session_context(ticket_id: str) -> list[str]:
    return list(SESSION_CONTEXTS.get(ticket_id, []))


def follow_up(
    ticket_id: str, customer_message: str, subject: str, body: str, customer_id: str
) -> dict[str, Any]:
    """Handle a follow-up message on an existing ticket with full prior context."""
    record_session_turn(ticket_id, "customer", customer_message)
    result = process_ticket(
        ticket_id, subject, body, customer_id, session_turns=get_session_context(ticket_id)
    )
    record_session_turn(ticket_id, "agent", (result.get("proposal") or "")[:500])
    return result
