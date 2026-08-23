"""Logging setup with a PII-redaction hook (NFR: Security).

`configure_logging` sets up the root logger from settings. `redact_pii` is a stub that
later sessions will expand to strip emails, card numbers, etc. before anything is logged.
"""

from __future__ import annotations

import logging

from app.config import settings


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def redact_pii(text: str) -> str:
    """Placeholder — return text with PII removed. To be implemented (NFR: Security)."""
    return text
