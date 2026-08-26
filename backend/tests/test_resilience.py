"""Tests for the reliability layer (NFR Reliability) — pure, no network, no sleeps.

The circuit breaker uses an injected fake clock; Resilience policies use a recorded
sleep stub, so timing behavior is exercised deterministically.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.core.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    Resilience,
    backoff_delay,
)


class FakeClock:
    """Monotonic-ish clock the test can advance manually."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --- backoff calculator -----------------------------------------------------------
def test_backoff_is_exponential_then_capped():
    assert backoff_delay(0, base_s=0.5) == 0.5
    assert backoff_delay(1, base_s=0.5) == 1.0
    assert backoff_delay(2, base_s=0.5) == 2.0
    # cap: no matter the attempt, never above max_s
    assert backoff_delay(10, base_s=0.5, max_s=8.0) == 8.0


def test_backoff_rejects_negative_attempt():
    with pytest.raises(ValueError):
        backoff_delay(-1)


# --- CircuitBreaker state machine ---------------------------------------------------
def test_breaker_opens_after_threshold_consecutive_failures():
    breaker = CircuitBreaker(failure_threshold=3, clock=FakeClock())
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed"
    breaker.allow()  # still accepting calls below threshold
    breaker.record_failure()
    assert breaker.state == "open"
    with pytest.raises(CircuitOpenError):
        breaker.allow()


def test_breaker_success_resets_failure_count():
    breaker = CircuitBreaker(failure_threshold=3, clock=FakeClock())
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()  # streak broken
    breaker.record_failure()
    assert breaker.state == "closed"  # only 1 consecutive failure now


def test_breaker_half_open_probe_then_close_or_reopen():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_s=30.0, clock=clock)
    breaker.record_failure()
    assert breaker.state == "open"

    # before the timeout elapses, all calls are refused
    with pytest.raises(CircuitOpenError):
        breaker.allow()

    clock.advance(30.0)
    assert breaker.state == "half_open"
    breaker.allow()  # exactly one probe granted
    with pytest.raises(CircuitOpenError):
        breaker.allow()  # second call while probe outstanding is refused

    # failed probe re-opens immediately (timeout restarts)
    breaker.record_failure()
    assert breaker.state == "open"
    clock.advance(30.0)

    # successful probe closes and fully resets
    breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed"


def test_breaker_validates_threshold():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)


# --- Resilience facade --------------------------------------------------------------
def test_resilience_returns_result_on_first_success():
    policy = Resilience.off()
    assert policy.call(lambda x: x * 2, 21) == 42


def _raise(msg: str):
    raise RuntimeError(msg)


def test_resilience_retries_until_success_with_backoff():
    sleeps: list[float] = []
    policy = Resilience(
        name="t", max_attempts=3, timeout_s=0, backoff_base_s=0.25, sleep=sleeps.append
    )
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    assert policy.call(flaky) == "ok"
    assert calls["n"] == 3
    assert sleeps == [0.25, 0.5]  # exponential between attempts


def test_resilience_exhausts_attempts_and_raises_original_error():
    policy = Resilience(name="t", max_attempts=2, timeout_s=0, sleep=lambda _s: None)

    def always():
        raise ValueError("terminal")

    with pytest.raises(ValueError, match="terminal"):
        policy.call(always)


def test_resilience_does_not_retry_circuit_open():
    breaker = CircuitBreaker(failure_threshold=1, clock=FakeClock())
    policy = Resilience(
        name="t", max_attempts=3, timeout_s=0, sleep=lambda _s: None, breaker=breaker
    )
    with pytest.raises(RuntimeError):
        policy.call(_raise, "x")  # one failure opens it
    assert breaker.state == "open"

    calls = {"n": 0}

    def counted():
        calls["n"] += 1

    with pytest.raises(CircuitOpenError):
        policy.call(counted)
    assert calls["n"] == 0  # short-circuited without invoking the callable


def test_resilience_times_out_slow_calls():
    policy = Resilience(name="t", max_attempts=1, timeout_s=0.05, sleep=lambda _s: None)

    def slow():
        time.sleep(1.0)

    start = time.perf_counter()
    with pytest.raises(TimeoutError):
        policy.call(slow)
    assert time.perf_counter() - start < 0.7  # stopped waiting, did not hang for 1s


def test_resilience_records_success_and_failure_into_breaker():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=2, clock=clock)
    policy = Resilience(
        name="t", max_attempts=1, timeout_s=0, sleep=lambda _s: None, breaker=breaker
    )

    policy.call(lambda: "fine")
    assert breaker.state == "closed"

    with pytest.raises(RuntimeError):
        policy.call(_raise, "f1")
    with pytest.raises(RuntimeError):
        policy.call(_raise, "f2")
    assert breaker.state == "open"


def test_timed_out_call_finishing_later_does_not_crash_pool():
    """A timed-out call completing later in its worker thread must stay quiet."""
    policy = Resilience(name="t", max_attempts=1, timeout_s=0.02, sleep=lambda _s: None)
    done = threading.Event()

    def late_finisher():
        time.sleep(0.1)
        done.set()

    with pytest.raises(TimeoutError):
        policy.call(late_finisher)
    assert done.wait(timeout=2.0)  # thread completed quietly in the background
