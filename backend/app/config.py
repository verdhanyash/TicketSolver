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

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
