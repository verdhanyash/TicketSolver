"""Tests for PII log redaction (NFR: Security, Module 13) — pure, no network.

Covers the pattern rules directly via `redact_pii` and the logging wiring via a
handler-level `PiiRedactionFilter`, mirroring how `configure_logging` attaches the
filter to root handlers so every child logger inherits it.
"""

from __future__ import annotations

import io
import logging

import pytest

from app.config import settings
from app.core.logging import PiiRedactionFilter, configure_logging, redact_pii


# --- pattern rules: each class of PII is masked -------------------------------------
@pytest.mark.parametrize(
    ("raw", "token"),
    [
        # emails
        ("contact jane.doe@example.com today", "[EMAIL_REDACTED]"),
        ("user+tag@sub.example.co.uk replied", "[EMAIL_REDACTED]"),
        # card-like numbers, Luhn-valid, in every grouping style
        ("card 4111111111111111 declined", "[CARD_REDACTED]"),
        ("card 4111 1111 1111 1111 declined", "[CARD_REDACTED]"),
        ("card 4111-1111-1111-1111 declined", "[CARD_REDACTED]"),
        ("amex 378282246310005 charged", "[CARD_REDACTED]"),
        ("visa 4222222222222 authorized", "[CARD_REDACTED]"),
        # phone-like numbers, US-ish shapes
        ("call (555) 123-4567 now", "[PHONE_REDACTED]"),
        ("call 555-123-4567 now", "[PHONE_REDACTED]"),
        ("call +1 555 123 4567 now", "[PHONE_REDACTED]"),
    ],
)
def test_each_pattern_is_masked(raw: str, token: str):
    redacted = redact_pii(raw)
    assert token in redacted


@pytest.mark.parametrize(
    ("raw", "forbidden"),
    [
        ("contact jane.doe@example.com today", "jane.doe@example.com"),
        ("user+tag@sub.example.co.uk replied", "user+tag@sub.example.co.uk"),
        ("card 4111111111111111 declined", "4111111111111111"),
        ("card 4111 1111 1111 1111 declined", "4111"),
        ("amex 378282246310005 charged", "378282246310005"),
        ("visa 4222222222222 authorized", "4222222222222"),
        ("call (555) 123-4567 now", "4567"),
        ("call 555-123-4567 now", "555-123-4567"),
        ("call +1 555 123 4567 now", "123 4567"),
    ],
)
def test_masked_output_contains_no_original_pii(raw: str, forbidden: str):
    out = redact_pii(raw)
    assert forbidden not in out


# --- ordinary content survives byte-identical ----------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        "order 12345 shipped",
        "total $1,234.50 charged to invoice",
        "ticket T-12345 updated status",
        "invoice dated 2024-01-15 closed",
        "run 1234567890123456 failed checksum",  # 16-digit non-Luhn run stays intact
        "no sensitive content here at all",
        "build version 2026.08.26 finished",
        "",
    ],
)
def test_ordinary_content_survives_untouched(raw: str):
    assert redact_pii(raw) == raw


# --- mixed message: all three rules in one single pass -------------------------------
def test_mixed_message_single_pass_masks_all_three_types():
    raw = (
        "Email jane.doe@acme.io about card 4111 1111 1111 1111, "
        "ref T-12345, total $49.00, call (555) 123-4567"
    )
    expected = (
        "Email [EMAIL_REDACTED] about card [CARD_REDACTED], "
        "ref T-12345, total $49.00, call [PHONE_REDACTED]"
    )
    assert redact_pii(raw) == expected


# --- logging pipeline ------------------------------------------------------------------
class _Capture:
    """Handler-on-logger harness mirroring configure_logging's root-handler wiring."""

    def __init__(self, name: str = "pii-redaction-test") -> None:
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.handler.addFilter(PiiRedactionFilter())
        self.logger = logging.getLogger(name)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    @property
    def output(self) -> str:
        return self.stream.getvalue()

    def close(self) -> None:
        self.logger.removeHandler(self.handler)
        self.handler.close()


@pytest.fixture
def capture() -> _Capture:
    cap = _Capture()
    yield cap
    cap.close()


def test_nested_dict_interpolated_via_args_is_redacted(capture: _Capture):
    payload = {"customer": {"email": "bob.smith@corp.example"}, "card": "4111111111111111"}
    capture.logger.info("payload=%s", payload)
    out = capture.output
    assert "payload=" in out  # args were interpolated into the message
    assert "[EMAIL_REDACTED]" in out
    assert "[CARD_REDACTED]" in out
    assert "bob.smith@corp.example" not in out
    assert "4111111111111111" not in out


def test_exception_text_is_redacted(capture: _Capture):
    try:
        raise ValueError("login failed for bob.smith@corp.example")
    except ValueError:
        capture.logger.exception("handler crashed")
    out = capture.output
    assert "Traceback (most recent call last):" in out  # traceback still present
    assert "ValueError: login failed for" in out  # non-PII traceback detail kept
    assert "[EMAIL_REDACTED]" in out
    assert "bob.smith@corp.example" not in out


@pytest.mark.parametrize(
    ("fmt", "args", "expected_line"),
    [
        ("TicketSolver API started on port %d", (8000,), "TicketSolver API started on port 8000\n"),
        ("ticket=%s amount=%.2f", ("T-99999", 49.5), "ticket=T-99999 amount=49.50\n"),
        ("plain message with no args", (), "plain message with no args\n"),
        # formatted output containing %% must not be re-interpolated by the filter
        ("progress: %d%% done", (99,), "progress: 99% done\n"),
    ],
)
def test_normal_log_lines_are_byte_identical(
    capture: _Capture, fmt: str, args: tuple, expected_line: str
):
    capture.logger.info(fmt, *args)
    assert capture.output == expected_line


# --- configure_logging wiring -----------------------------------------------------------
@pytest.fixture
def root_logging_snapshot():
    """Save/restore root logger state around tests that mutate it."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield root
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_configure_logging_installs_filter_and_is_idempotent(root_logging_snapshot):
    root = root_logging_snapshot
    root.handlers.clear()
    configure_logging()
    configure_logging()
    configure_logging()
    assert len(root.handlers) == 1  # basicConfig ran exactly once
    for handler in root.handlers:
        installed = [f for f in handler.filters if isinstance(f, PiiRedactionFilter)]
        assert len(installed) == 1  # filter attached once, never duplicated


def test_configure_logging_respects_settings_level_and_format(root_logging_snapshot):
    from app.core import logging as core_logging

    root = root_logging_snapshot
    root.handlers.clear()
    configure_logging()
    assert root.level == getattr(logging, settings.log_level.upper(), logging.INFO)
    formatter = root.handlers[0].formatter
    assert formatter is not None
    # same format string configure_logging passes to basicConfig
    assert formatter._fmt == core_logging._LOG_FORMAT


def test_child_logger_records_are_redacted_through_root_handlers(root_logging_snapshot):
    root = root_logging_snapshot
    root.handlers.clear()
    configure_logging()
    buf = io.StringIO()
    original_stream = root.handlers[0].stream
    root.handlers[0].stream = buf
    try:
        child = logging.getLogger("app.some.module.redaction-check")
        child.info("contact bob.smith@corp.example about order T-12345")
    finally:
        root.handlers[0].stream = original_stream
    out = buf.getvalue()
    assert "[EMAIL_REDACTED]" in out
    assert "bob.smith@corp.example" not in out
    assert "T-12345" in out  # ordinary identifiers untouched
