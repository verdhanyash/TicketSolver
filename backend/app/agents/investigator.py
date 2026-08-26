"""Investigator/Resolver agent (SRS FR-2) — the first real agent.

A single LangGraph agent role: given a ticket, it gathers evidence via tool calls
(KB search over Qdrant, simulated account/order lookups), then proposes a resolution.
Every step (LLM call, tool call) is traced to Postgres per FR-10.

Graph shape: LangGraph's canonical bind-tools loop. The `investigator` node invokes
the LLM with tools bound; a conditional edge routes to `execute_tools` while the model
requests tools and back again; when it answers plainly, `finalize` persists the proposal.
One agent role, two nodes + conditional edge (the stock ToolNode is replaced by a custom
executor so each tool call gets its own trace row with latency).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from app.config import settings
from app.db.models import Ticket
from app.db.session import get_session_factory
from app.llm.nim_client import chat as nim_chat
from app.memory.episodic import format_similar_incidents
from app.memory.long_term import format_customer_memory
from app.tools import get_order_status, lookup_account
from app.vector.kb import search_kb

logger = logging.getLogger(__name__)

# --- Tools exposed to the model (name -> callable). Patchable in tests. ---------
TOOL_REGISTRY: dict[str, Any] = {
    "kb_search": lambda query: search_kb(query),
    "lookup_account": lambda customer_id: lookup_account(customer_id),
    "get_order_status": lambda order_id: get_order_status(order_id),
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "kb_search",
            "description": "Search company policy/FAQ documents for guidance relevant to a support question.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search phrase, e.g. 'refund window for annual plan'"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_account",
            "description": "Look up a customer account by ID: plan tier, account age, MFA status.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up an order's status (delivered/in_transit/lost/delayed), item, and amount.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are NimbusWare's support investigator. You resolve customer support tickets.

Rules:
- Before proposing anything, gather evidence with tools: kb_search for policy questions,
  lookup_account for the customer's account, get_order_status when an order is mentioned.
- Base claims ONLY on tool results or the ticket text — never invent policy numbers.
- Then produce your final answer in exactly this format:

PROPOSED ACTION: <one concrete action, e.g. "Issue refund of $XX.XX" / "Reship lost kit">
CUSTOMER REPLY: <the response to send to the customer>
EVIDENCE: <bullet list of the facts/tools that justify it>
CONFIDENCE: <a number from 0 to 1 for how confident you are this action is correct>"""


class AgentState(TypedDict):
    ticket_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    proposal: str  # set by finalize_node


# --- Trace writer (module-level so tests can substitute a SQLite-backed one) ------
def _writer():
    from app.core.tracing import TraceWriter

    return TraceWriter(get_session_factory())


def _load_customer_memory(customer_id: str):
    """Fetch long-term customer facts (FR-6). Patch target for unit tests."""
    from app.memory.long_term import get_customer_memory

    return get_customer_memory(customer_id)


def _find_similar_incidents(text: str, ticket_id: str) -> list[dict]:
    """Retrieve similar past incidents (FR-7). Patch target for unit tests."""
    from app.memory.episodic import find_similar_incidents

    return find_similar_incidents(text, exclude_ticket_id=ticket_id)


# Write-back is NOT done here: FR-7 scopes episodic memory to *resolved* incidents,
# and a proposal only becomes resolved after the orchestrator's guardrail/decision
# layer. The orchestrator's resolve node (auto-resolutions) and the approval
# endpoint (human-approved resolutions, FR-9) record episodes.


# --- Nodes -------------------------------------------------------------------
def _call_model(messages: list[AnyMessage]):
    """Single LLM invocation point (patch target for unit tests).

    Runs under the LLM reliability policy (retry/timeout/breaker, NFR Reliability).
    """
    from app.core.resilience import get_llm_policy

    return get_llm_policy().call(
        nim_chat, settings.nim_model_strong, [m.model_dump() for m in messages], TOOL_SCHEMAS
    )


def _execute_tool(fn: Any, **kwargs):
    """Run one tool call under the reliability policy. Patch target for unit tests."""
    from app.core.resilience import get_tool_policy

    return get_tool_policy().call(fn, **kwargs)


def _to_ai_message(response: Any):
    choice = response.choices[0]
    msg = choice.message
    return AIMessage(
        content=msg.content or "",
        tool_calls=[
            {"name": tc.function.name, "args": json.loads(tc.function.arguments), "id": tc.id}
            for tc in (msg.tool_calls or [])
        ],
    )


def investigator_node(state: AgentState) -> dict:
    start = time.perf_counter()
    response = _call_model(state["messages"])
    ai_msg = _to_ai_message(response)
    latency_ms = int((time.perf_counter() - start) * 1000)
    system_head = next((m.content for m in state["messages"] if isinstance(m, SystemMessage)), "")
    try:
        _writer().log_step(
            ticket_id=state["ticket_id"],
            step_type="llm_call",
            name="investigator_llm",
            input_payload={
                "message_count": len(state["messages"]),
                # auditable context: long-term customer facts land here (FR-6)
                "system_prompt": system_head[:2000] or None,
            },
            output_payload={"content": ai_msg.content[:2000], "tool_calls": [tc["name"] for tc in ai_msg.tool_calls]},
            reasoning=(ai_msg.content or "")[:4000] or None,
            model=settings.nim_model_strong,
            latency_ms=latency_ms,
        )
    except Exception:  # tracing must never break resolution flow
        logger.exception("trace write failed (llm_call)")
    return {"messages": [ai_msg]}


def execute_tools_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    outputs: list[ToolMessage] = []
    for tc in last.tool_calls:
        fn = TOOL_REGISTRY.get(tc["name"])
        start = time.perf_counter()
        if fn is None:
            result: dict | str = f"Unknown tool: {tc['name']}"
        else:
            try:
                result = _execute_tool(fn, **tc["args"])
            except Exception as exc:  # surface tool failure to the model
                logger.exception("tool %s failed", tc["name"])
                result = f"Tool error: {exc}"
        latency_ms = int((time.perf_counter() - start) * 1000)
        payload = {"result": result}
        try:
            _writer().log_step(
                ticket_id=state["ticket_id"],
                step_type="tool_call",
                name=tc["name"],
                input_payload=tc["args"],
                output_payload={"result_preview": str(result)[:1500]},
                latency_ms=latency_ms,
            )
        except Exception:
            logger.exception("trace write failed (tool_call %s)", tc["name"])
        outputs.append(ToolMessage(content=json.dumps(payload)[:8000], tool_call_id=tc["id"]))
    return {"messages": outputs}


def finalize_node(state: AgentState) -> dict:
    final = next(
        (
            m
            for m in reversed(state["messages"])
            if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip()
        ),
        None,
    )
    proposal = (final.content if final else "").strip()
    try:
        _writer().log_step(
            ticket_id=state["ticket_id"],
            step_type="decision",
            name="proposed_resolution",
            reasoning=proposal,
        )
        with get_session_factory()() as session:
            ticket = session.get(Ticket, state["ticket_id"])
            if ticket is not None:
                ticket.proposed_resolution = proposal
                session.commit()
    except Exception:
        logger.exception("failed to persist proposed resolution")
    return {"proposal": proposal}


def _route_after_investigator(state: AgentState) -> str:
    last = state["messages"][-1]
    return "execute_tools" if getattr(last, "tool_calls", None) else "finalize"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("investigator", investigator_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("finalize", finalize_node)
    graph.set_entry_point("investigator")
    graph.add_conditional_edges("investigator", _route_after_investigator, ["execute_tools", "finalize"])
    graph.add_edge("execute_tools", "investigator")
    graph.add_edge("finalize", END)
    return graph.compile()


compiled_graph = build_graph()


def run_investigation(
    ticket_id: str,
    subject: str,
    body: str,
    customer_id: str,
    *,
    follow_ups: list[str] | None = None,
) -> str:
    """Entry point: investigate one loaded ticket and return the proposed resolution.

    `follow_ups` carries earlier conversation turns (FR-5 short-term memory) so
    multi-turn follow-up questions keep their context; each is appended as its own
    message after the initial ticket text.
    """
    # Long-term customer memory (FR-6): load facts and prepend them to the system
    # prompt. Memory failure must never break resolution flow — degrade to no facts.
    start = time.perf_counter()
    try:
        record = _load_customer_memory(customer_id)
        memory_block = format_customer_memory(record)
    except Exception:
        logger.exception("failed to load customer memory for %s; continuing without it", customer_id)
        record, memory_block = None, ""
    latency_ms = int((time.perf_counter() - start) * 1000)
    try:
        _writer().log_step(
            ticket_id=ticket_id,
            step_type="memory_retrieval",
            name="customer_memory",
            input_payload={"customer_id": customer_id},
            output_payload={"found": record is not None, "facts_block": memory_block or None},
            latency_ms=latency_ms,
        )
    except Exception:
        logger.exception("trace write failed (memory_retrieval)")

    # Episodic memory (FR-7): surface similar past incidents before proposing.
    start = time.perf_counter()
    try:
        similar = _find_similar_incidents(f"{subject}\n{body}", ticket_id)
        episodes_block = format_similar_incidents(similar)
    except Exception:
        logger.exception("failed to retrieve similar incidents; continuing without them")
        similar, episodes_block = [], ""
    latency_ms = int((time.perf_counter() - start) * 1000)
    try:
        _writer().log_step(
            ticket_id=ticket_id,
            step_type="memory_retrieval",
            name="episodic_memory",
            input_payload={"query": f"{subject}\n{body}"[:500]},
            output_payload={
                "hits": [
                    {"ticket_id": c["ticket_id"], "subject": c["subject"], "score": round(c["score"], 4)}
                    for c in similar
                ]
            },
            latency_ms=latency_ms,
        )
    except Exception:
        logger.exception("trace write failed (memory_retrieval episodic)")

    context_blocks = [b for b in (memory_block, episodes_block) if b]
    system_prompt = (
        f"{SYSTEM_PROMPT}\n\n" + "\n\n".join(context_blocks) if context_blocks else SYSTEM_PROMPT
    )
    initial = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Ticket {ticket_id}\nCustomer: {customer_id}\nSubject: {subject}\n\n{body}"
        ),
        *[HumanMessage(content=f"Follow-up: {turn}") for turn in (follow_ups or [])],
    ]
    result = compiled_graph.invoke({"ticket_id": ticket_id, "messages": initial})
    return result.get("proposal", "")
