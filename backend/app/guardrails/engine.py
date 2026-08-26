"""Guardrail engine (FR-3): hard-coded policy checks enforced in application code.

Rules live HERE as code and thresholds as constants — never in prompts, never
delegated to the LLM. The engine parses the investigator's `PROPOSED ACTION:` line
into a structured action, then evaluates the declarative rule set:

- refund at/above $50 → block (human approval required)
- recognized non-refund actions → allow
- refunds without a parseable amount → block (limit can't be verified)
- unrecognized actions → block conservatively (default-deny)

`check_action` returns an allow/block decision; blocking is the caller's cue to route
to the HITL queue (M5) instead of executing autonomously.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Policy constants (FR-3). Changing these changes system behavior by definition.
REFUND_APPROVAL_LIMIT_USD = 50.0

# Action kinds the engine recognizes; anything else is default-deny.
KNOWN_KINDS = {"refund", "reship", "cancel", "credit", "discount", "info", "no_action"}

_REFUND_RE = re.compile(r"\brefund(?:ing|ed)?\b|\bmoney\s+back\b|\breimburse", re.IGNORECASE)
_RESHIP_RE = re.compile(r"\breship|\breplacement\b|\bsend(ing)?\s+(a\s+)?new\s+", re.IGNORECASE)
_CANCEL_RE = re.compile(r"\bcancel", re.IGNORECASE)
_CREDIT_RE = re.compile(r"\bcredit\b.*\baccount\b|\bstore\s+credit\b", re.IGNORECASE)
_DISCOUNT_RE = re.compile(r"\bdiscount\b|\bvoucher\b|\bcoupon\b", re.IGNORECASE)
_INFO_RE = re.compile(
    r"\b(confirm|clarify|explain|provide information|look into|investigate)\b|\bno action\b",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(r"\$\s?([0-9][0-9,]*(?:\.[0-9]{1,2})?)|(?:USD\s?)([0-9][0-9,]*(?:\.[0-9]{1,2})?)")


@dataclass(frozen=True)
class ProposedAction:
    """Structured view of the investigator's proposed action."""

    kind: str  # one of KNOWN_KINDS or "unknown"
    amount_usd: float | None = None


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    reason: str | None = None
    rule: str | None = None

    def blocked_reason(self) -> str:
        return self.reason or "blocked by policy"


def parse_proposed_action(proposal_text: str) -> ProposedAction:
    """Extract (kind, amount) from a free-text PROPOSED ACTION line.

    Refund detection wins over other verbs so "reship AND refund" still hits the
    money-rule; amounts are parsed from anywhere in the text.
    """
    text = proposal_text or ""
    if _REFUND_RE.search(text):
        return ProposedAction(kind="refund", amount_usd=_parse_amount(text))
    if _RESHIP_RE.search(text):
        return ProposedAction(kind="reship")
    if _CANCEL_RE.search(text):
        return ProposedAction(kind="cancel")
    if _CREDIT_RE.search(text):
        return ProposedAction(kind="credit", amount_usd=_parse_amount(text))
    if _DISCOUNT_RE.search(text):
        return ProposedAction(kind="discount", amount_usd=_parse_amount(text))
    if _INFO_RE.search(text):
        return ProposedAction(kind="info")
    return ProposedAction(kind="unknown")


def _parse_amount(text: str) -> float | None:
    match = _AMOUNT_RE.search(text)
    if not match:
        return None
    raw = (match.group(1) or match.group(2)).replace(",", "")
    try:
        return round(float(raw), 2)
    except ValueError:
        return None


def check_action(action: ProposedAction) -> GuardrailDecision:
    """Evaluate the declarative rule set against one structured action."""
    if action.kind == "unknown":
        return GuardrailDecision(
            allowed=False,
            reason=f"unrecognized action type requires human review: {action.kind}",
            rule="default_deny",
        )
    if action.kind == "refund":
        if action.amount_usd is None:
            return GuardrailDecision(
                allowed=False,
                reason="refund amount missing/unparseable; limit cannot be verified",
                rule="refund_limit",
            )
        if action.amount_usd >= REFUND_APPROVAL_LIMIT_USD:
            return GuardrailDecision(
                allowed=False,
                reason=(
                    f"refund of ${action.amount_usd:,.2f} meets/exceeds the "
                    f"${REFUND_APPROVAL_LIMIT_USD:,.2f} autonomous limit"
                ),
                rule="refund_limit",
            )
        return GuardrailDecision(allowed=True, rule="refund_limit")

    # Recognized, currently-unregulated kinds pass through.
    return GuardrailDecision(allowed=True, rule=f"{action.kind}_unregulated")


def check_proposal(proposal_text: str) -> tuple[ProposedAction, GuardrailDecision]:
    """Convenience: parse then check in one call."""
    action = parse_proposed_action(proposal_text)
    return action, check_action(action)
