"""
Intelligent RAG System — Document API Routes
=============================================
Handles: document upload, listing, detail view, deletion.

FASTAPI ROUTERS:
A Router groups related endpoints together. This file owns all /documents/* endpoints.
The main app imports this router and mounts it.

HOW FILE UPLOAD WORKS:
1. Client sends a multipart/form-data POST request with the file
2. FastAPI reads the file into memory (UploadFile)
3. We validate: type, size, duplicate name
4. We save the file to disk
5. We create a database record with status="processing"
6. We trigger background processing (extract, chunk, embed, index)
7. We immediately return the document ID so the client can poll for status
"""

import uuid
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete

from app.core.config import get_settings
from app.core.database import get_db
from app.core.logging import get_logger
from app.models.database import Document, DocumentChunk
from app.models.schemas import (
    DocumentUploadResponse,
    DocumentResponse,
    DocumentListResponse,
    ErrorResponse,
)
from app.services.document_processor import (
    save_uploaded_file,
    extract_text,
    chunk_text_with_pages,
    get_file_type,
)
from app.services.vector_store import add_chunks_to_vector_store, delete_document_chunks

router = APIRouter(prefix="/documents", tags=["documents"])
logger = get_logger(__name__)
settings = get_settings()

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


# =============================================================================
# Background Processing Task
# =============================================================================

async def process_document_background(document_id: str, file_path: str, file_type: str) -> None:
    """
    Background task: extract text → chunk → embed → index.
    
    WHY BACKGROUND?
    Embedding can take 5-30 seconds for large documents. We don't want the
    upload request to hang. Instead we:
    1. Accept the upload immediately (fast response to user)
    2. Run the heavy processing in the background
    3. Update the document status when done
    4. Frontend polls /documents/{id} to check status
    """
    from app.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        try:
            # 1. Fetch the document record
            result = await db.execute(select(Document).where(Document.id == document_id))
            doc = result.scalar_one_or_none()
            if not doc:
                logger.error("Document not found in background task", id=document_id)
                return

            logger.info("Processing document", id=document_id, filename=doc.filename)

            # 2. Extract text
            fp = Path(file_path)
            extraction = extract_text(fp, file_type)
            doc.content = extraction.full_text[:50000]  # Store first 50K chars
            doc.page_count = extraction.page_count

            # 3. Chunk
            chunks = chunk_text_with_pages(extraction)
            doc.chunk_count = len(chunks)

            # 4. Save chunks to database
            db_chunks = []
            for chunk in chunks:
                db_chunk = DocumentChunk(
                    id=chunk.id,
                    document_id=document_id,
                    text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    is_indexed=False,
                )
                db_chunks.append(db_chunk)
                db.add(db_chunk)

            await db.flush()  # Write to DB without committing yet

            # 5. Embed and add to vector store
            chunk_ids = [c.id for c in chunks]
            texts = [c.text for c in chunks]
            metadatas = [
                {
                    "document_id": document_id,
                    "filename": doc.filename,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number or 0,  # Chroma needs int, not None
                }
                for c in chunks
            ]

            add_chunks_to_vector_store(chunk_ids, texts, metadatas)

            # 6. Mark chunks as indexed
            for db_chunk in db_chunks:
                db_chunk.is_indexed = True

            # 7. Update document status to ready
            doc.status = "ready"
            await db.commit()

            logger.info(
                "Document processed successfully",
                id=document_id,
                chunks=len(chunks),
                pages=extraction.page_count,
            )

        except Exception as e:
            logger.error("Document processing failed", id=document_id, error=str(e))
            await db.rollback()
            # Update status to error
            async with AsyncSessionLocal() as err_db:
                result = await err_db.execute(select(Document).where(Document.id == document_id))
                doc = result.scalar_one_or_none()
                if doc:
                    doc.status = "error"
                    doc.error_message = str(e)
                    await err_db.commit()


# =============================================================================
# API Endpoints
# =============================================================================

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for indexing",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a PDF or TXT document.
    
    Returns immediately with status=202 and a document ID.
    Use GET /documents/{id} to poll for processing completion.
    """
    # --- Validate file type ---
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    file_type = get_file_type(file.filename) or "txt"

    # --- Read and validate file size ---
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB",
        )

    # --- Save file to disk ---
    file_path, file_size = await save_uploaded_file(file.filename, content)

    # --- Create database record ---
    document_id = str(uuid.uuid4())
    doc = Document(
        id=document_id,
        filename=file.filename,
        file_path=str(file_path),
        file_type=file_type,
        file_size=file_size,
        status="processing",
    )
    db.add(doc)
    await db.commit()

    # --- Schedule background processing ---
    background_tasks.add_task(
        process_document_background,
        document_id=document_id,
        file_path=str(file_path),
        file_type=file_type,
    )

    logger.info("Document upload accepted", id=document_id, filename=file.filename)
    return DocumentUploadResponse(
        id=document_id,
        filename=file.filename,
        file_type=file_type,
        file_size=file_size,
        status="processing",
        message="Document uploaded. Processing will complete in a few seconds.",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all documents",
)
async def list_documents(db: AsyncSession = Depends(get_db)):
    """Return all documents with summary statistics."""
    result = await db.execute(
        select(Document).order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()

    # Get total chunk count
    chunk_result = await db.execute(select(func.sum(Document.chunk_count)))
    total_chunks = chunk_result.scalar() or 0

    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(doc) for doc in documents],
        total=len(documents),
        total_chunks=total_chunks,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get a document by ID",
)
async def get_document(document_id: str, db: AsyncSession = Depends(get_db)):
    """Get details of a specific document, including processing status."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return DocumentResponse.model_validate(doc)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a document and its indexed data",
)
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    """
    Delete a document, its chunks from the database, and its vectors from Chroma.
    This is a complete cleanup — the document is fully removed.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    filename = doc.filename
    file_path = Path(doc.file_path)

    # 1. Remove from vector store
    deleted_vectors = delete_document_chunks(document_id)

    # 2. Delete the document (cascades to chunks in DB)
    await db.delete(doc)
    await db.commit()

    # 3. Delete the file from disk (non-critical — don't fail if already deleted)
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.warning("Could not delete file from disk", path=str(file_path), error=str(e))

    logger.info("Document deleted", id=document_id, filename=filename, vectors_removed=deleted_vectors)
    return {
        "message": f"Document '{filename}' and all its indexed data have been deleted.",
        "document_id": document_id,
        "vectors_removed": deleted_vectors,
    }
