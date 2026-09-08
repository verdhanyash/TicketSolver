"""Logging setup with PII redaction (NFR: Security).

`configure_logging` sets up the root logger from settings and attaches a
`PiiRedactionFilter` to every root handler, so every application logger inherits
redaction through normal propagation. Setup is idempotent: importing the app twice
(or calling `configure_logging` repeatedly) never duplicates handlers or filters.

Redaction rules (applied to message text and traceback text alike):
- Email addresses -> ``[EMAIL_REDACTED]``
- Card-like numbers (13-19 digits, optionally grouped by spaces/dashes) that pass a
  Luhn checksum -> ``[CARD_REDACTED]``. Non-Luhn digit runs (order ids, timestamps,
  arbitrary identifiers) are left untouched.
- Phone-like numbers, US-ish shapes such as ``(555) 123-4567``, ``555-123-4567``,
  ``+1 555 123 4567`` -> ``[PHONE_REDACTED]``

Ordinary text — order ids like ``T-12345``, dollar amounts like ``$1,234.50``,
dates, versions — passes through byte-identical.
"""

from __future__ import annotations

import logging
import re
import traceback

from app.config import settings

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s :: %(message)s"

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Candidate digit runs of 13-19 digits, possibly grouped by single spaces/dashes.
# The leading/trailing \b keeps runs embedded in longer alphanumeric tokens intact;
# Luhn validation below decides whether a run is really card-like.
_CARD_CANDIDATE_RE = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
_PHONE_RE = re.compile(
    r"(?<![\w+])"
    r"(?:\+?1[ -]?)?"  # optional US country code
    r"(?:\(\d{3}\)[ -]?\d{3}[ -]\d{4}|\d{3}[ -]\d{3}[ -]\d{4})"
    r"(?!\w)"
)

_EMAIL_TOKEN = "[EMAIL_REDACTED]"
_CARD_TOKEN = "[CARD_REDACTED]"
_PHONE_TOKEN = "[PHONE_REDACTED]"


def _luhn_valid(digits: str) -> bool:
    """Standard Luhn checksum: True if `digits` satisfies it."""
    total = 0
    parity = len(digits) % 2  # indices whose digit gets doubled
    for i, ch in enumerate(digits):
        d = ord(ch) - 48
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_pii(text: str) -> str:
    """Return `text` with emails, Luhn-valid card-like numbers, and phone-like
    numbers replaced by fixed tokens. Everything else is left byte-identical."""
    if not isinstance(text, str) or not text:
        return text
    text = _EMAIL_RE.sub(_EMAIL_TOKEN, text)
    text = _CARD_CANDIDATE_RE.sub(_mask_card_candidate, text)
    text = _PHONE_RE.sub(_PHONE_TOKEN, text)
    return text


def _mask_card_candidate(match: re.Match[str]) -> str:
    digits = re.sub(r"[ -]", "", match.group())
    if 13 <= len(digits) <= 19 and _luhn_valid(digits):
        return _CARD_TOKEN
    return match.group()


class PiiRedactionFilter(logging.Filter):
    """Logging filter that scrubs PII out of a record before it is formatted.

    Rewrites ``record.msg`` to the fully interpolated (and redacted) message and
    empties ``record.args`` so every formatter downstream sees clean text.
    Traceback text is sanitized too: when ``exc_info`` is present, the traceback is
    rendered once here, redacted, and cached in ``record.exc_text`` — the standard
    formatter reuses the cache instead of rendering the raw text again.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact_pii(record.getMessage())
            record.args = ()
            if record.exc_info and not record.exc_text:
                record.exc_text = "".join(traceback.format_exception(*record.exc_info))
            if record.exc_text:
                record.exc_text = redact_pii(record.exc_text)
        except Exception:  # noqa: BLE001, S110 -- never let redaction break the caller or the handler
            pass
        return True


_redactor: PiiRedactionFilter | None = None


def _get_redactor() -> PiiRedactionFilter:
    global _redactor
    if _redactor is None:
        _redactor = PiiRedactionFilter()
    return _redactor


def configure_logging() -> None:
    """Configure root logging from settings and wire PII redaction onto every root
    handler, so records from any application logger (which propagate to root) are
    scrubbed before emission.

    Idempotent: mirrors `logging.basicConfig` semantics (first call wins) and only
    ever installs one redaction filter per handler, so repeated calls — double
    import of `app.main`, script re-entry, test re-runs — change nothing.
    """
    root = logging.getLogger()
    if not root.handlers:
        level = getattr(logging, settings.log_level.upper(), logging.INFO)
        logging.basicConfig(level=level, format=_LOG_FORMAT)
    for handler in root.handlers:
        if not any(isinstance(f, PiiRedactionFilter) for f in handler.filters):
            handler.addFilter(_get_redactor())
