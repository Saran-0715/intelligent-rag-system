"""
Intelligent RAG System — System / Dashboard API Routes
=======================================================
Provides system status, health checks, and dashboard statistics.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.config import get_settings
from app.core.database import get_db
from app.core.logging import get_logger
from app.models.database import Document, QueryLog
from app.models.schemas import SystemStatus, DashboardStats
from app.services.vector_store import get_vector_store_stats

router = APIRouter(prefix="/system", tags=["system"])
logger = get_logger(__name__)

APP_VERSION = "0.1.0-phase1"


@router.get("/health", summary="Health check")
async def health_check():
    """Simple liveness probe — returns 200 if the server is running."""
    return {"status": "ok", "version": APP_VERSION}


@router.get("/status", response_model=SystemStatus, summary="System configuration status")
async def get_system_status(db: AsyncSession = Depends(get_db)):
    """
    Returns the system configuration state and document counts.
    Used by the frontend to show appropriate UI states
    (e.g. warn user if LLM is not configured).
    """
    settings = get_settings()
    
    # Document stats
    doc_count = await db.execute(select(func.count(Document.id)))
    total_docs = doc_count.scalar() or 0

    # Query stats
    query_count = await db.execute(select(func.count(QueryLog.id)))
    total_queries = query_count.scalar() or 0

    # Vector store stats
    try:
        vs_stats = get_vector_store_stats()
        total_chunks = vs_stats["total_chunks"]
    except Exception:
        total_chunks = 0

    # Build warnings for the user
    warnings = []
    if not settings.llm_enabled:
        if settings.llm_provider == "none":
            warnings.append(
                "LLM provider is set to 'none'. Add GROQ_API_KEY to .env to enable AI answers."
            )
        elif settings.llm_provider == "groq" and not settings.groq_api_key:
            warnings.append(
                "GROQ_API_KEY is not configured. The system will operate in retrieval-only mode."
            )
    if total_docs == 0:
        warnings.append("No documents uploaded yet. Upload documents to start querying.")

    return SystemStatus(
        status="ok",
        version=APP_VERSION,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        llm_enabled=settings.llm_enabled,
        embedding_model=settings.embedding_model,
        total_documents=total_docs,
        total_chunks=total_chunks,
        total_queries=total_queries,
        warnings=warnings,
    )


@router.get("/dashboard", response_model=DashboardStats, summary="Dashboard statistics")
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)):
    """Returns real statistics for the dashboard page."""
    # Document counts
    doc_result = await db.execute(select(func.count(Document.id)))
    total_docs = doc_result.scalar() or 0

    # Chunk total
    chunk_result = await db.execute(select(func.sum(Document.chunk_count)))
    total_chunks = chunk_result.scalar() or 0

    # Query total
    query_result = await db.execute(select(func.count(QueryLog.id)))
    total_queries = query_result.scalar() or 0

    # Recent queries (last 5)
    recent_q_result = await db.execute(
        select(QueryLog)
        .order_by(QueryLog.created_at.desc())
        .limit(5)
    )
    recent_queries = [
        {
            "id": q.id,
            "question": q.question[:100],
            "created_at": q.created_at.isoformat(),
            "llm_used": q.llm_used,
        }
        for q in recent_q_result.scalars().all()
    ]

    # Document status breakdown
    status_result = await db.execute(
        select(Document.status, func.count(Document.id)).group_by(Document.status)
    )
    docs_by_status = {row[0]: row[1] for row in status_result.all()}

    return DashboardStats(
        total_documents=total_docs,
        total_chunks=total_chunks,
        total_queries=total_queries,
        recent_queries=recent_queries,
        documents_by_status=docs_by_status,
    )
