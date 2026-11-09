"""
Intelligent RAG System — SQLAlchemy Database Models
=====================================================
Defines the database schema using SQLAlchemy ORM.

WHAT IS AN ORM?
An ORM (Object-Relational Mapper) lets you work with database tables as Python
classes. Instead of writing raw SQL, you create Python objects and SQLAlchemy
translates them to SQL automatically.

WHY SQLITE FOR PHASE 1?
SQLite is a file-based database with zero setup — perfect for development.
The same SQLAlchemy models work with PostgreSQL later (Phase 5).
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text, ForeignKey
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Document(Base):
    """
    Represents an uploaded document.
    
    One Document → many DocumentChunks (one-to-many relationship).
    """
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    # e.g. "college_attendance_policy.pdf"
    filename = Column(String, nullable=False)
    # Original file path on disk
    file_path = Column(String, nullable=False)
    # "pdf" or "txt"
    file_type = Column(String, nullable=False)
    # File size in bytes
    file_size = Column(Integer, nullable=False)
    # Extracted text content (stored for re-chunking later)
    content = Column(Text, nullable=True)
    # Number of pages (for PDFs)
    page_count = Column(Integer, nullable=True)
    # processing | ready | error
    status = Column(String, default="processing", nullable=False)
    # Error message if processing failed
    error_message = Column(Text, nullable=True)
    # Number of chunks created from this document
    chunk_count = Column(Integer, default=0)
    # Optional user-provided description
    description = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    # Relationship: access doc.chunks to get all chunks
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",  # Delete chunks when document is deleted
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id!r} filename={self.filename!r} status={self.status!r}>"


class DocumentChunk(Base):
    """
    A single text chunk extracted from a Document.
    
    WHY CHUNKS?
    LLMs have context limits (max tokens). We can't send an entire 100-page
    document to an LLM. Instead, we split it into smaller pieces, embed each
    piece as a vector, and retrieve only the most relevant pieces at query time.
    """
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, index=True)
    # Foreign key → links to parent Document
    document_id = Column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # The actual text content of this chunk
    text = Column(Text, nullable=False)
    # Sequential position of this chunk within the document (0-indexed)
    chunk_index = Column(Integer, nullable=False)
    # Page number where this chunk starts (1-indexed, None for txt)
    page_number = Column(Integer, nullable=True)
    # Character offset within the full document text
    char_start = Column(Integer, nullable=True)
    char_end = Column(Integer, nullable=True)
    # Whether this chunk is indexed in the vector store
    is_indexed = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Back-reference to parent document
    document = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk id={self.id!r} doc={self.document_id!r} "
            f"chunk={self.chunk_index} page={self.page_number}>"
        )


class QueryLog(Base):
    """
    Stores every question asked through the system.
    Used for dashboard statistics and future evaluation.
    """
    __tablename__ = "query_logs"

    id = Column(String, primary_key=True, index=True)
    # The user's question
    question = Column(Text, nullable=False)
    # The generated answer (may be None if retrieval-only mode)
    answer = Column(Text, nullable=True)
    # JSON-serialized list of retrieved chunk IDs
    retrieved_chunk_ids = Column(Text, nullable=True)
    # Number of chunks retrieved
    chunks_retrieved = Column(Integer, default=0)
    # Whether an LLM was used to generate the answer
    llm_used = Column(Boolean, default=False)
    # LLM model name used (if any)
    llm_model = Column(String, nullable=True)
    # Time taken to retrieve + generate (seconds)
    latency_seconds = Column(Float, nullable=True)
    # Estimated token count (for cost tracking)
    tokens_used = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
