"""
Intelligent RAG System — Backend Tests
=======================================
Tests for Phase 1 MVP core functionality.

HOW TO RUN:
    # From the backend/ directory:
    pytest tests/ -v

    # Run a specific test file:
    pytest tests/test_document_processor.py -v

    # Run with coverage:
    pytest tests/ -v --cov=app

WHAT WE TEST:
1. Document processing (chunking logic) — no I/O needed, pure functions
2. API endpoints — using FastAPI's TestClient (synchronous test mode)
3. System health endpoint

We test the core logic without depending on:
- LLM API keys (not needed for retrieval tests)
- External services (mocked or skipped if unavailable)
"""

import pytest
import tempfile
from pathlib import Path


# =============================================================================
# Tests for Document Processor (Pure Logic — No I/O)
# =============================================================================

class TestChunking:
    """Tests for text chunking logic."""

    def test_chunk_simple_text(self):
        """Basic chunking test: text should be split into expected number of chunks."""
        from app.services.document_processor import ExtractionResult, chunk_text_with_pages

        # Create a fake extraction result with known text
        long_text = "A" * 2000  # 2000 characters of 'A'
        extraction = ExtractionResult(
            full_text=long_text,
            pages=[{"page": 1, "text": long_text}],
            page_count=1,
        )

        chunks = chunk_text_with_pages(extraction, chunk_size=500, chunk_overlap=50)

        # With 2000 chars, chunk_size=500, overlap=50 → step=450
        # Number of chunks: ceil(2000 / 450) ≈ 5 chunks
        assert len(chunks) > 1, "Long text should produce multiple chunks"
        assert all(len(c.text) <= 500 for c in chunks), "No chunk should exceed chunk_size"

    def test_chunk_preserves_page_numbers(self):
        """Chunks should remember which page they came from."""
        from app.services.document_processor import ExtractionResult, chunk_text_with_pages

        extraction = ExtractionResult(
            full_text="Page one text. Page two text.",
            pages=[
                {"page": 1, "text": "Page one text."},
                {"page": 2, "text": "Page two text."},
            ],
            page_count=2,
        )

        chunks = chunk_text_with_pages(extraction, chunk_size=500, chunk_overlap=0)

        pages_seen = {c.page_number for c in chunks}
        assert 1 in pages_seen, "Should have chunks from page 1"
        assert 2 in pages_seen, "Should have chunks from page 2"

    def test_chunk_has_unique_ids(self):
        """Every chunk must have a unique ID."""
        from app.services.document_processor import ExtractionResult, chunk_text_with_pages

        text = "Some test content. " * 100
        extraction = ExtractionResult(
            full_text=text,
            pages=[{"page": 1, "text": text}],
            page_count=1,
        )

        chunks = chunk_text_with_pages(extraction, chunk_size=200, chunk_overlap=20)
        ids = [c.id for c in chunks]

        assert len(ids) == len(set(ids)), "All chunk IDs must be unique"

    def test_chunk_index_sequential(self):
        """Chunk indices should be sequential starting from 0."""
        from app.services.document_processor import ExtractionResult, chunk_text_with_pages

        text = "Word " * 500
        extraction = ExtractionResult(
            full_text=text,
            pages=[{"page": 1, "text": text}],
            page_count=1,
        )

        chunks = chunk_text_with_pages(extraction, chunk_size=200, chunk_overlap=0)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks))), "Chunk indices must be sequential"

    def test_empty_text_produces_no_chunks(self):
        """Empty document should produce no chunks."""
        from app.services.document_processor import ExtractionResult, chunk_text_with_pages

        extraction = ExtractionResult(
            full_text="",
            pages=[{"page": 1, "text": ""}],
            page_count=1,
        )

        chunks = chunk_text_with_pages(extraction, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 0, "Empty text should produce zero chunks"


class TestFileTypeDetection:
    """Tests for file type detection."""

    def test_pdf_extension(self):
        from app.services.document_processor import get_file_type
        assert get_file_type("document.pdf") == "pdf"
        assert get_file_type("REPORT.PDF") == "pdf"

    def test_txt_extension(self):
        from app.services.document_processor import get_file_type
        assert get_file_type("notes.txt") == "txt"

    def test_unsupported_extension(self):
        from app.services.document_processor import get_file_type
        assert get_file_type("image.jpg") is None
        assert get_file_type("data.csv") is None


class TestTxtExtraction:
    """Tests for TXT text extraction."""

    def test_extract_txt(self):
        """TXT files should be extracted as-is."""
        from app.services.document_processor import extract_text_from_txt

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello, this is a test document.\nSecond line.")
            tmp_path = Path(f.name)

        try:
            result = extract_text_from_txt(tmp_path)
            assert "Hello" in result.full_text
            assert "Second line" in result.full_text
            assert result.page_count == 1
        finally:
            tmp_path.unlink()


# =============================================================================
# Tests for Configuration
# =============================================================================

class TestConfig:
    """Tests for application configuration."""

    def test_settings_load(self):
        """Settings should load with defaults."""
        from app.core.config import get_settings
        settings = get_settings()
        assert settings.embedding_model == "all-MiniLM-L6-v2"
        assert settings.chunk_size == 512
        assert settings.retrieval_top_k == 5

    def test_llm_disabled_when_no_key(self, monkeypatch):
        """LLM should be disabled when no API key is set."""
        # Clear the cache to allow re-reading with monkeypatched env
        import app.core.config as config_module
        config_module.get_settings.cache_clear()
        
        monkeypatch.setenv("GROQ_API_KEY", "")
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        
        settings = config_module.Settings()  # Create fresh instance
        assert settings.llm_enabled is False
        
        config_module.get_settings.cache_clear()  # Reset for other tests

    def test_allowed_origins_parsed(self):
        """Comma-separated origins should be parsed into a list."""
        from app.core.config import Settings
        s = Settings(allowed_origins="http://localhost:5173,http://localhost:3000")
        assert "http://localhost:5173" in s.allowed_origins_list
        assert "http://localhost:3000" in s.allowed_origins_list


# =============================================================================
# API Integration Tests
# =============================================================================

@pytest.fixture
def test_client():
    """
    Create a synchronous test client for the FastAPI app.
    
    WHY AsyncClient isn't used here:
    For simple Phase 1 tests, the synchronous TestClient is easier to use.
    It internally runs the async app synchronously using anyio.
    """
    import asyncio
    from httpx import AsyncClient, ASGITransport
    # We use the actual app but with a test database
    return None  # Placeholder — full async test setup shown below


@pytest.mark.asyncio
async def test_health_endpoint():
    """
    Test that the health endpoint returns 200.
    This verifies the entire app starts up correctly.
    """
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/system/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test that the root endpoint returns expected structure."""
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "docs" in data


@pytest.mark.asyncio
async def test_list_documents_empty():
    """List documents should return empty list when no documents exist."""
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/documents")
    
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)


@pytest.mark.asyncio
async def test_query_empty_knowledge_base():
    """Querying an empty knowledge base should return a graceful message."""
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/query",
            json={"question": "What is the attendance policy?"},
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)


@pytest.mark.asyncio
async def test_upload_invalid_file_type():
    """Uploading an unsupported file type should return 400."""
    import io
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.jpg", io.BytesIO(b"fake image content"), "image/jpeg")},
        )
    
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_empty_file():
    """Uploading an empty file should return 400."""
    import io
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )
    
    assert response.status_code == 400
