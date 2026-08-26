"""Application configuration loaded from environment variables.

Uses pydantic-settings so all config is centralized and validated. No secrets are
hardcoded anywhere else in the codebase (see NFR: Security).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173"

    # NVIDIA NIM (LLM)
    nim_api_key: str = ""
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_model_strong: str = ""
    nim_model_cheap: str = ""
    nim_embedding_model: str = ""

    # PostgreSQL
    database_url: str = ""

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "kb_chunks"

    # Ollama (local embedding server — user decision 2026-08-23: embeddings come
    # from nomic-embed-text via Ollama's HTTP API, NOT NIM-hosted/sentence-transformers)
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    embedding_dim: int = 768  # verified against a live call, not assumed

    # Reliability policies (NFR Reliability; see app/core/resilience.py)
    tool_max_attempts: int = 3
    tool_timeout_s: float = 10.0
    tool_breaker_failures: int = 3
    tool_breaker_reset_s: float = 30.0
    llm_max_attempts: int = 3
    llm_timeout_s: float = 60.0
    llm_breaker_failures: int = 3
    llm_breaker_reset_s: float = 60.0

    # Orchestrator (FR-4): bounded retry-on-low-confidence; LLM self-reported
    # confidence until calibration replaces it (M7)
    orchestrator_max_attempts: int = 2
    confidence_threshold: float = 0.6

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
