"""
Intelligent RAG System — Document Processing Service
=====================================================
Handles: file saving, text extraction, chunking.

WHAT IS CHUNKING?
When you embed text, you convert it to a vector (a list of numbers that
captures meaning). Embedding models have a maximum input length (e.g. 512
tokens). Long documents must be split into smaller "chunks" before embedding.

A good chunk is:
- Short enough to fit in the embedding model's context window
- Long enough to contain a complete thought or answer
- Has some overlap with its neighbors so context isn't lost at boundaries

PHASE 1 CHUNKING STRATEGY: Fixed-size character chunking with overlap.
This is simple and works well enough for a first version.
Phase 2 will add semantic/sentence-aware chunking.
"""

import uuid
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# =============================================================================
# File Saving
# =============================================================================

async def save_uploaded_file(filename: str, content: bytes) -> tuple[Path, int]:
    """
    Save an uploaded file to the uploads directory.
    
    Returns: (file_path, file_size_bytes)
    """
    settings.upload_path.mkdir(parents=True, exist_ok=True)

    # Sanitize filename: remove path traversal characters
    safe_name = Path(filename).name  # Strip any directory components
    safe_name = safe_name.replace(" ", "_")

    # Add a unique prefix to avoid collisions
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    file_path = settings.upload_path / unique_name

    file_path.write_bytes(content)
    logger.info("File saved", filename=unique_name, size=len(content))
    return file_path, len(content)


def get_file_type(filename: str) -> str | None:
    """Return 'pdf' or 'txt', or None for unsupported types."""
    suffix = Path(filename).suffix.lower()
    return {"pdf": "pdf", ".pdf": "pdf", ".txt": "txt"}.get(suffix) or \
           {"pdf": "pdf", "txt": "txt"}.get(suffix.lstrip("."))


# =============================================================================
# Text Extraction
# =============================================================================

class ExtractionResult:
    """Result of extracting text from a document."""
    def __init__(
        self,
        full_text: str,
        pages: list[dict],  # [{"page": 1, "text": "..."}]
        page_count: int,
    ):
        self.full_text = full_text
        self.pages = pages          # Page-level text for page-aware chunking
        self.page_count = page_count


def extract_text_from_pdf(file_path: Path) -> ExtractionResult:
    """
    Extract text from a PDF using PyMuPDF (fitz).
    
    WHY PYMUPDF?
    It's one of the fastest and most accurate PDF text extractors for Python.
    It preserves page information, which lets us cite exact page numbers.
    
    Returns page-by-page text so we can track which chunk came from which page.
    """
    pages = []
    full_text_parts = []

    with fitz.open(str(file_path)) as doc:
        page_count = len(doc)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            # Clean up excessive whitespace but preserve paragraph structure
            text = "\n".join(
                line.strip() for line in text.splitlines() if line.strip()
            )
            if text:
                pages.append({"page": page_num, "text": text})
                full_text_parts.append(text)

    full_text = "\n\n".join(full_text_parts)
    logger.info("PDF extracted", path=str(file_path), pages=page_count, chars=len(full_text))
    return ExtractionResult(full_text=full_text, pages=pages, page_count=page_count)


def extract_text_from_txt(file_path: Path) -> ExtractionResult:
    """
    Extract text from a plain text file.
    TXT files have no page structure, so page_count=1.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    text = "\n".join(line for line in text.splitlines())
    logger.info("TXT extracted", path=str(file_path), chars=len(text))
    return ExtractionResult(
        full_text=text,
        pages=[{"page": 1, "text": text}],
        page_count=1,
    )


def extract_text(file_path: Path, file_type: str) -> ExtractionResult:
    """Route to the right extractor based on file type."""
    if file_type == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_type == "txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


# =============================================================================
# Text Chunking
# =============================================================================

class TextChunk:
    """A single text chunk with metadata."""
    def __init__(
        self,
        text: str,
        chunk_index: int,
        page_number: int | None,
        char_start: int,
        char_end: int,
    ):
        self.id = str(uuid.uuid4())
        self.text = text
        self.chunk_index = chunk_index
        self.page_number = page_number
        self.char_start = char_start
        self.char_end = char_end


def chunk_text_with_pages(
    extraction: ExtractionResult,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """
    Split extracted text into overlapping chunks, preserving page information.
    
    HOW IT WORKS:
    1. Process each page's text separately to maintain page boundaries
    2. Split each page into fixed-size character chunks
    3. Add overlap between chunks so context isn't lost at boundaries
    
    Example with chunk_size=100, overlap=20:
    "The quick brown fox jumped over the lazy dog sitting by the river"
    Chunk 1: chars 0-99
    Chunk 2: chars 80-179   ← starts 20 chars back (the overlap)
    
    This means if an answer crosses a chunk boundary, it's still captured.
    
    Args:
        extraction: Result from extract_text()
        chunk_size: Max characters per chunk (default from settings)
        chunk_overlap: Characters of overlap (default from settings)
    
    Returns:
        List of TextChunk objects
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    chunks: list[TextChunk] = []
    chunk_index = 0
    global_char_offset = 0  # Track position in the full document

    for page_data in extraction.pages:
        page_num = page_data["page"]
        page_text = page_data["text"]

        if not page_text.strip():
            global_char_offset += len(page_text) + 2  # +2 for "\n\n"
            continue

        # Split this page's text into chunks
        pos = 0
        while pos < len(page_text):
            end = min(pos + chunk_size, len(page_text))
            chunk_text = page_text[pos:end].strip()

            if chunk_text:  # Skip empty chunks
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    page_number=page_num,
                    char_start=global_char_offset + pos,
                    char_end=global_char_offset + end,
                ))
                chunk_index += 1

            # Move forward by (chunk_size - overlap)
            # If overlap >= chunk_size, we'd loop forever — guard against that
            step = max(chunk_size - chunk_overlap, 1)
            pos += step

        global_char_offset += len(page_text) + 2  # +2 for "\n\n" separator

    logger.info(
        "Text chunked",
        total_chunks=len(chunks),
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )
    return chunks
