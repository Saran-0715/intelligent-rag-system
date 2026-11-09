"""
Intelligent RAG System — Async Database Session Management
===========================================================
Provides the SQLAlchemy async engine and session factory.

WHY ASYNC?
FastAPI is an async web framework. Using async database access means the server
can handle many simultaneous requests without blocking threads. 'aiosqlite'
makes SQLite work with Python's async/await syntax.
"""

from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.database import Base

logger = get_logger(__name__)


def _get_engine():
    """Create and return the async SQLAlchemy engine."""
    settings = get_settings()
    
    # Ensure the data directory exists
    db_url = settings.database_url
    if "sqlite" in db_url:
        # Extract the file path from the URL and create its parent directories
        db_file = db_url.replace("sqlite+aiosqlite:///", "")
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(
        db_url,
        echo=settings.debug,   # Log all SQL queries in debug mode
        future=True,
    )
    return engine


# Module-level engine and session factory (created once)
engine = _get_engine()
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Keep objects accessible after commit
)


async def init_db() -> None:
    """
    Create all database tables if they don't exist.
    Called once at application startup.
    """
    logger.info("Initializing database tables...")
    async with engine.begin() as conn:
        # Creates tables defined in Base's subclasses (Document, DocumentChunk, etc.)
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully.")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session.
    
    Usage in a route:
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...
    
    The 'yield' makes this a context manager — the session is automatically
    closed after the request completes, even if an error occurs.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
