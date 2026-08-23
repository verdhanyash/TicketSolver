"""NVIDIA NIM client factory (OpenAI-compatible).

NIM exposes an OpenAI-compatible API, so we use the `openai` SDK pointed at the NIM base
URL. The ML cost-router (FR-14) later picks between the cheap and strong model per ticket.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.config import settings


@lru_cache(maxsize=1)
def get_nim_client() -> OpenAI:
    return OpenAI(api_key=settings.nim_api_key, base_url=settings.nim_base_url)
