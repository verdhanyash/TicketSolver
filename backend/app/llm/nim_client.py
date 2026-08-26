"""NVIDIA NIM client (OpenAI-compatible).

NIM exposes an OpenAI-compatible API, so we use the `openai` SDK pointed at the NIM base
URL. The ML cost-router (FR-14) later picks between the cheap and strong model per ticket.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_nim_client() -> OpenAI:
    return OpenAI(api_key=settings.nim_api_key, base_url=settings.nim_base_url)


def chat(model: str, messages: list[dict], tools: list[dict] | None = None) -> Any:
    """One chat completion against NIM, timed and logged. Returns the raw response."""
    start = time.perf_counter()
    client = get_nim_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=0.2,
    )
    usage = response.usage
    logger.info(
        "nim.chat model=%s latency_ms=%d prompt_tokens=%s completion_tokens=%s",
        model,
        int((time.perf_counter() - start) * 1000),
        getattr(usage, "prompt_tokens", "?"),
        getattr(usage, "completion_tokens", "?"),
    )
    return response
