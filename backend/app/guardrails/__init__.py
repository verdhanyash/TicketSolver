"""Guardrails (FR-3).

Hard-coded policy checks enforced in application code — NOT delegated to the LLM.
`engine.py` parses proposed actions and evaluates the declarative rule set; blocked
actions are routed to the HITL approval queue instead of executing autonomously.
"""
