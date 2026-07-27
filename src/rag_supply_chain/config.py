"""Central configuration, loaded from environment / .env.

Every component imports `settings` from here rather than reading os.environ
directly, so there is one source of truth for hosts, topics, and model IDs.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Kafka / Redpanda ---
    kafka_bootstrap_servers: str = "localhost:9092"
    topic_documents_raw: str = "documents.raw"
    topic_documents_dlq: str = "documents.dlq"
    topic_chunks: str = "chunks.embed"

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_chunks"

    # --- Postgres ---
    postgres_host: str = "localhost"
    postgres_port: int = 5433  # not 5432 — avoids clashing with a locally-installed Postgres
    postgres_user: str = "rag"
    postgres_password: str = "rag"
    postgres_db: str = "rag_registry"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # --- Chunking ---
    chunk_min_tokens: int = 128
    chunk_max_tokens: int = 512

    # --- Claude ---
    anthropic_api_key: str | None = None
    gen_model: str = "claude-opus-4-8"
    judge_model: str = "claude-haiku-4-5"

    # --- Paths ---
    inbox_dir: str = Field(default="./inbox")

    @property
    def postgres_dsn(self) -> str:
        """SQLAlchemy/psycopg connection string for the metadata registry."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
