"""
Intelligent RAG System — Pydantic API Schemas
==============================================
These schemas define the shape of data that flows through the API:
- Request bodies (what clients send)
- Response bodies (what the API returns)

WHY SEPARATE FROM DATABASE MODELS?
Database models define how data is STORED. API schemas define how data is
COMMUNICATED. Keeping them separate gives you flexibility — you can expose
different fields in the API without changing the database, and add computed
fields like "is_llm_configured".
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# Document Schemas
# =============================================================================

class DocumentUploadResponse(BaseModel):
    """Returned immediately after a file upload is accepted."""
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    message: str


class ChunkInfo(BaseModel):
    """Brief info about a retrieved document chunk."""
    chunk_id: str
    text: str
    page_number: Optional[int] = None
    chunk_index: int
    relevance_score: Optional[float] = None  # Cosine similarity score


class DocumentResponse(BaseModel):
    """Full document details returned by GET /documents/{id}."""
    id: str
    filename: str
    file_type: str
    file_size: int
    page_count: Optional[int] = None
    status: str
    error_message: Optional[str] = None
    chunk_count: int
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # Allow creation from SQLAlchemy model


class DocumentListResponse(BaseModel):
    """List of documents with summary stats."""
    documents: list[DocumentResponse]
    total: int
    total_chunks: int


# =============================================================================
# Query / Answer Schemas
# =============================================================================

class QueryRequest(BaseModel):
    """Body of a question submitted to the Ask AI endpoint."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The natural-language question to answer",
        examples=["What is the attendance policy?"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of chunks to retrieve (uses system default if omitted)",
    )


class SourceReference(BaseModel):
    """
    A source citation attached to an answer.
    Tells the user WHICH document and WHERE the information came from.
    """
    chunk_id: str
    document_id: str
    filename: str
    page_number: Optional[int] = None
    chunk_index: int
    # The actual passage from the document
    passage: str
    # How relevant this chunk was (0.0–1.0, higher = more relevant)
    relevance_score: Optional[float] = None


class QueryResponse(BaseModel):
    """
    Full response to a question.
    
    Design principle: Every answer includes the sources used so the user
    can verify the information themselves. This is core to grounded RAG.
    """
    query_id: str
    question: str
    answer: str
    sources: list[SourceReference]
    # True if an LLM generated the answer; False = retrieval-only summary
    llm_used: bool
    # True if the LLM said it couldn't find enough info
    answer_grounded: bool
    # Processing times for transparency
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    # LLM model used (None if not used)
    model_used: Optional[str] = None


# =============================================================================
# System / Dashboard Schemas
# =============================================================================

class SystemStatus(BaseModel):
    """
    System health and configuration status.
    Shown on the dashboard and used by the frontend to decide what to display.
    """
    status: str                     # "ok" | "degraded"
    version: str
    llm_provider: str               # "groq" | "none"
    llm_model: str
    llm_enabled: bool               # Whether LLM is actually configured
    embedding_model: str
    total_documents: int
    total_chunks: int
    total_queries: int
    # Configuration warnings for the user
    warnings: list[str]


class DashboardStats(BaseModel):
    """Statistics displayed on the dashboard."""
    total_documents: int
    total_chunks: int
    total_queries: int
    # Last 5 queries with their questions
    recent_queries: list[dict]
    # Document processing status breakdown
    documents_by_status: dict[str, int]


# =============================================================================
# Error Schema
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response body."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
