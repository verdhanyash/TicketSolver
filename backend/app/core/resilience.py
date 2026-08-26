"""Reliability primitives (NFR: Reliability) — retry, timeout, circuit breaker.

Small, dependency-free building blocks wrapped around any callable:

- `backoff_delay(attempt)` — exponential delay before retry N (pure, unit-tested).
- `CircuitBreaker` — closed → opens after `failure_threshold` consecutive failures →
  refuses calls until `reset_timeout_s` elapses → allows a single half-open probe;
  probe success closes it, probe failure reopens it. Takes an injectable monotonic
  clock so tests can advance time without sleeping.
- `Resilience` — facade combining breaker + bounded retries + per-attempt timeout for
  sync callables. Timeout runs the call in a worker thread and stops waiting when it
  expires (the underlying thread is abandoned; acceptable at demo scale).

Policies used in production are built from settings (`get_tool_policy` /
`get_llm_policy`); tests use `Resilience.off()` (pass-through, no state).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


class CircuitOpenError(RuntimeError):
    """Raised when a call is attempted while the circuit is open."""


def backoff_delay(attempt: int, base_s: float = 0.5, factor: float = 2.0, max_s: float = 8.0) -> float:
    """Delay before retry number `attempt` (0-based): base * factor**attempt, capped."""
    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    return min(base_s * (factor**attempt), max_s)


class CircuitBreaker:
    """Three-state breaker: closed → open → (after timeout) half-open."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        reset_timeout_s: float = 30.0,
        name: str = "breaker",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.name = name
        self._clock = clock
        self._state = "closed"  # closed | open | half_open
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        # An open breaker whose timeout has elapsed becomes half-open on inspection.
        if self._state == "open" and self._probe_due():
            return "half_open"
        return self._state

    def _probe_due(self) -> bool:
        return (
            self._opened_at is not None
            and (self._clock() - self._opened_at) >= self.reset_timeout_s
        )

    def allow(self) -> None:
        """Raise `CircuitOpenError` unless the next call may proceed."""
        if self._state == "closed":
            return
        if self._state == "open":
            if self._probe_due():
                logger.info("circuit %s half-open: allowing one probe", self.name)
                self._state = "half_open"
                return
            raise CircuitOpenError(f"circuit {self.name} is open")
        # half_open: exactly one probe allowed at a time.
        raise CircuitOpenError(f"circuit {self.name} is half-open; probe already outstanding")

    def record_success(self) -> None:
        self._consecutive_failures = 0
        if self._state != "closed":
            logger.info("circuit %s closed after successful %s", self.name, self._state)
        self._state = "closed"
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._state == "half_open":
            logger.warning("circuit %s re-opens after failed probe", self.name)
            self._trip()
        elif self._consecutive_failures >= self.failure_threshold:
            logger.warning(
                "circuit %s opens after %d consecutive failures",
                self.name,
                self._consecutive_failures,
            )
            self._trip()

    def _trip(self) -> None:
        self._state = "open"
        self._opened_at = self._clock()


class Resilience:
    """Bounded-retry + timeout + optional breaker wrapper for sync callables."""

    def __init__(
        self,
        *,
        name: str,
        max_attempts: int = 3,
        timeout_s: float = 10.0,
        backoff_base_s: float = 0.5,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.name = name
        self.max_attempts = max_attempts
        self.timeout_s = timeout_s
        self.backoff_base_s = backoff_base_s
        self.breaker = breaker
        self._sleep = sleep
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"resil-{name}")

    @classmethod
    def off(cls, name: str = "off") -> Resilience:
        """Pass-through policy: single attempt, no timeout, no breaker (for tests)."""
        return cls(name=name, max_attempts=1, timeout_s=10**6, backoff_base_s=0)

    def call(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Run `fn` under this policy and return its result or raise."""
        last_error: BaseException | None = None
        for attempt in range(self.max_attempts):
            if self.breaker is not None:
                self.breaker.allow()  # raises CircuitOpenError while refusing calls
            try:
                return self._call_once(fn, args, kwargs)
            except CircuitOpenError:
                raise  # never retried — the point of the breaker
            except Exception as exc:  # noqa: BLE001 — any failure is retryable here
                last_error = exc
                if self.breaker is not None:
                    self.breaker.record_failure()
                logger.warning(
                    "%s attempt %d/%d failed: %s", self.name, attempt + 1, self.max_attempts, exc
                )
                if attempt + 1 < self.max_attempts:
                    self._sleep(backoff_delay(attempt, base_s=self.backoff_base_s))
        assert last_error is not None  # loop only exits via return or raise
        raise last_error

    def _call_once(self, fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
        if not self.timeout_s or self.timeout_s >= 10**5:  # .off() fast path: run inline
            result = fn(*args, **kwargs)
            if self.breaker is not None:
                self.breaker.record_success()
            return result
        future = self._executor.submit(fn, *args, **kwargs)
        try:
            result = future.result(timeout=self.timeout_s)
        except TimeoutError:
            future.cancel()
            raise
        if self.breaker is not None:
            self.breaker.record_success()
        return result


# --- Production policies (lazy singletons built from settings) -------------------
_tool_policy: Resilience | None = None
_llm_policy: Resilience | None = None


def get_tool_policy() -> Resilience:
    global _tool_policy
    if _tool_policy is None:
        from app.config import settings

        _tool_policy = Resilience(
            name="tool",
            max_attempts=settings.tool_max_attempts,
            timeout_s=settings.tool_timeout_s,
            breaker=CircuitBreaker(
                name="tool",
                failure_threshold=settings.tool_breaker_failures,
                reset_timeout_s=settings.tool_breaker_reset_s,
            ),
        )
    return _tool_policy


def get_llm_policy() -> Resilience:
    global _llm_policy
    if _llm_policy is None:
        from app.config import settings

        _llm_policy = Resilience(
            name="llm",
            max_attempts=settings.llm_max_attempts,
            timeout_s=settings.llm_timeout_s,
            breaker=CircuitBreaker(
                name="llm",
                failure_threshold=settings.llm_breaker_failures,
                reset_timeout_s=settings.llm_breaker_reset_s,
            ),
        )
    return _llm_policy


def reset_policies() -> None:
    """Drop cached production policies (used by tests to isolate breaker state)."""
    global _tool_policy, _llm_policy
    _tool_policy = None
    _llm_policy = None
