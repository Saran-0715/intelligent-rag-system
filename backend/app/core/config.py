"""
Intelligent RAG System — Application Configuration
====================================================
Uses Pydantic Settings to load config from environment variables / .env file.
This is the single source of truth for all configuration values.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    How it works:
    - Reads from .env file first (if it exists)
    - Then from actual environment variables (which override .env)
    - Provides typed, validated defaults
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM Provider ---
    groq_api_key: str = Field(default="", description="Groq API key")
    llm_model: str = Field(default="llama-3.1-8b-instant", description="LLM model name")
    llm_provider: str = Field(default="none", description="LLM provider: groq | none")

    # --- Embedding ---
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence Transformers model name"
    )

    # --- Database ---
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/rag_system.db",
        description="SQLAlchemy database URL"
    )

    # --- Vector Store ---
    chroma_db_path: str = Field(
        default="./data/chroma_db",
        description="Path to Chroma persistent storage"
    )
    chroma_collection_name: str = Field(
        default="rag_documents",
        description="Chroma collection name"
    )

    # --- File Upload ---
    upload_dir: str = Field(default="./data/uploads", description="File upload directory")
    max_upload_size_mb: int = Field(default=50, description="Max upload file size in MB")

    # --- Server ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=True)

    # --- CORS ---
    allowed_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        description="Comma-separated allowed CORS origins"
    )

    # --- Chunking ---
    chunk_size: int = Field(default=512, description="Text chunk size in characters")
    chunk_overlap: int = Field(default=64, description="Overlap between chunks in characters")

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=5, description="Number of chunks to retrieve")

    # --- Generation ---
    max_context_tokens: int = Field(
        default=4000,
        description="Maximum characters of context to send to LLM"
    )
    llm_temperature: float = Field(default=0.1, description="LLM temperature (0=deterministic)")

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse comma-separated origins into a list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_db_path)

    @property
    def llm_enabled(self) -> bool:
        """True when an LLM provider is configured and an API key is present."""
        if self.llm_provider == "none":
            return False
        if self.llm_provider == "groq" and not self.groq_api_key:
            return False
        return True

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    @lru_cache ensures this is only computed once per process.
    """
    return Settings()
