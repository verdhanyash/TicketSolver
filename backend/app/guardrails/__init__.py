"""Guardrails (FR-3).

Hard-coded policy checks enforced in application code — NOT delegated to the LLM. Any
action exceeding a threshold (e.g. refund > $50) is blocked and routed to the HITL queue.
No logic yet — added in a later session.
"""
