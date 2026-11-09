"""
Intelligent RAG System — Query / Ask AI API Routes
===================================================
Implements the RAG pipeline: Retrieve → Generate → Return

THE COMPLETE RAG PIPELINE (Phase 1):
┌─────────────────────────────────────────────────────────┐
│ 1. User submits question via POST /query                │
│ 2. question → embed_query() → 384-dim vector           │
│ 3. query vector → Chroma → top-5 similar chunks        │
│ 4. chunks + question → LLM (Groq) → grounded answer   │
│ 5. Return: answer + source citations + timing stats    │
└─────────────────────────────────────────────────────────┘
"""

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import get_db
from app.core.logging import get_logger
from app.models.database import Document, DocumentChunk, QueryLog
from app.models.schemas import QueryRequest, QueryResponse, SourceReference
from app.services.vector_store import search_similar_chunks
from app.services.llm_service import generate_answer

router = APIRouter(prefix="/query", tags=["query"])
logger = get_logger(__name__)
settings = get_settings()


@router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a question using RAG",
)
async def ask_question(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    The core RAG endpoint.
    
    1. Validates the question
    2. Retrieves relevant chunks from the vector store
    3. Generates a grounded answer (or returns retrieval-only if no LLM)
    4. Logs the query for analytics
    5. Returns answer + sources
    """
    total_start = time.time()
    query_id = str(uuid.uuid4())
    top_k = request.top_k or settings.retrieval_top_k

    logger.info("Query received", id=query_id, question=request.question[:80])

    # === STEP 1: Semantic Retrieval ===
    retrieval_start = time.time()
    similar_chunks = search_similar_chunks(
        query=request.question,
        top_k=top_k,
    )
    retrieval_time_ms = round((time.time() - retrieval_start) * 1000, 2)

    logger.info(
        "Retrieval complete",
        chunks_found=len(similar_chunks),
        retrieval_ms=retrieval_time_ms,
    )

    # === STEP 2: Enrich chunk metadata from database ===
    # Chroma metadata only has document_id and page_number.
    # We load the full filename from the Document table.
    source_references: list[SourceReference] = []
    passages_for_generation: list[dict] = []

    if similar_chunks:
        # Gather all unique document IDs from retrieved chunks
        doc_ids = list({c["metadata"].get("document_id") for c in similar_chunks})
        doc_result = await db.execute(
            select(Document).where(Document.id.in_(doc_ids))
        )
        docs_by_id = {doc.id: doc for doc in doc_result.scalars().all()}

        for chunk in similar_chunks:
            meta = chunk["metadata"]
            doc_id = meta.get("document_id", "")
            doc = docs_by_id.get(doc_id)
            filename = doc.filename if doc else meta.get("filename", "Unknown")
            page_num = meta.get("page_number")
            # Chroma stores 0 for None — convert back
            page_num = page_num if page_num and page_num > 0 else None

            source_references.append(SourceReference(
                chunk_id=chunk["chunk_id"],
                document_id=doc_id,
                filename=filename,
                page_number=page_num,
                chunk_index=meta.get("chunk_index", 0),
                passage=chunk["text"],
                relevance_score=chunk["score"],
            ))

            passages_for_generation.append({
                "text": chunk["text"],
                "filename": filename,
                "page_number": page_num,
                "score": chunk["score"],
            })

    # === STEP 3: LLM Generation ===
    gen_result = generate_answer(
        question=request.question,
        retrieved_passages=passages_for_generation,
    )
    generation_time_ms = gen_result.generation_time_ms
    total_time_ms = round((time.time() - total_start) * 1000, 2)

    # Determine if the answer is grounded (not an error or "not found" message)
    answer_grounded = (
        len(similar_chunks) > 0
        and "not contain sufficient information" not in gen_result.answer
        and "encountered an error" not in gen_result.answer
    )

    # === STEP 4: Log the query ===
    import json
    query_log = QueryLog(
        id=query_id,
        question=request.question,
        answer=gen_result.answer,
        retrieved_chunk_ids=json.dumps([c["chunk_id"] for c in similar_chunks]),
        chunks_retrieved=len(similar_chunks),
        llm_used=gen_result.llm_used,
        llm_model=gen_result.model_used,
        latency_seconds=total_time_ms / 1000,
        tokens_used=gen_result.tokens_used,
    )
    db.add(query_log)
    await db.commit()

    logger.info(
        "Query complete",
        id=query_id,
        llm_used=gen_result.llm_used,
        total_ms=total_time_ms,
    )

    return QueryResponse(
        query_id=query_id,
        question=request.question,
        answer=gen_result.answer,
        sources=source_references,
        llm_used=gen_result.llm_used,
        answer_grounded=answer_grounded,
        retrieval_time_ms=retrieval_time_ms,
        generation_time_ms=generation_time_ms,
        total_time_ms=total_time_ms,
        model_used=gen_result.model_used,
    )
