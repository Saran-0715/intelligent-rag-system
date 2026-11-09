"""
Intelligent RAG System — FastAPI Application Entry Point
========================================================
This is the root of the backend application.

FASTAPI APPLICATION LIFECYCLE:
1. App starts → lifespan() runs startup code (DB init, model warmup)
2. Requests come in → routers handle them
3. App shuts down → lifespan() runs cleanup

WHY LIFESPAN?
FastAPI's lifespan context manager replaces the older @app.on_event("startup")
pattern. It's the modern way to run setup/teardown code.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.database import init_db
from app.core.logging import setup_logging, get_logger
from app.api import documents, query, system


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Code before 'yield' runs at startup.
    Code after 'yield' runs at shutdown.
    """
    settings = get_settings()
    setup_logging(debug=settings.debug)
    logger = get_logger("app.main")

    logger.info("=" * 50)
    logger.info("Intelligent RAG System starting up", version="0.1.0-phase1")
    logger.info("LLM enabled", enabled=settings.llm_enabled, provider=settings.llm_provider)
    logger.info("Embedding model", model=settings.embedding_model)

    # Initialize database (create tables if they don't exist)
    await init_db()

    # Pre-warm the embedding model so the first query is fast
    # (Loading the model takes a few seconds — do it at startup, not on first request)
    try:
        logger.info("Pre-loading embedding model...")
        from app.services.embedding_service import get_embedding_model
        get_embedding_model()
        logger.info("Embedding model loaded and ready.")
    except Exception as e:
        logger.warning("Could not pre-load embedding model", error=str(e))

    logger.info("Server ready. Navigate to http://localhost:8000/docs for the API docs.")
    logger.info("=" * 50)

    yield  # Application runs here

    logger.info("Intelligent RAG System shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Intelligent RAG System",
        description=(
            "A production-oriented Knowledge Retrieval and Intelligence Platform. "
            "Phase 1 MVP — Document management, semantic search, and RAG generation."
        ),
        version="0.1.0-phase1",
        lifespan=lifespan,
        docs_url="/docs",          # Swagger UI
        redoc_url="/redoc",        # ReDoc UI
        openapi_url="/openapi.json",
    )

    # --- CORS Middleware ---
    # Allows the React frontend (localhost:5173) to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Include Routers ---
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(system.router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "message": "Intelligent RAG System API",
            "version": "0.1.0-phase1",
            "docs": "/docs",
            "status": "/api/v1/system/health",
        }

    return app


# This is the ASGI application object that uvicorn runs
app = create_app()
