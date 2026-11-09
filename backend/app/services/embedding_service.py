"""
Intelligent RAG System — Embedding Service
==========================================
Converts text chunks into vector embeddings for semantic search.

WHAT ARE EMBEDDINGS?
An embedding is a list of numbers (a vector) that represents the meaning of
a piece of text. Words or sentences with similar meanings have vectors that
are close together in "embedding space".

Example:
  "What is the attendance policy?" → [0.23, -0.11, 0.87, ...]  (384 numbers)
  "Rules about student attendance"  → [0.21, -0.09, 0.84, ...]  (similar!)
  "How to cook pasta?"              → [-0.45, 0.33, -0.12, ...] (very different)

WHY SENTENCE TRANSFORMERS?
- Runs entirely locally — no API calls, no cost, no data sent anywhere
- 'all-MiniLM-L6-v2' is fast, small (~90MB), and very accurate for RAG
- Downloads once and is cached to disk automatically

HOW SIMILARITY WORKS:
We use cosine similarity — it measures the angle between two vectors.
  cos(θ) = 1.0 → vectors point same direction = very similar meaning
  cos(θ) = 0.0 → vectors are perpendicular = unrelated meaning
  cos(θ) = -1.0 → opposite meanings

This is exactly what Chroma does when you ask it to "find similar documents".
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the embedding model.
    
    @lru_cache(maxsize=1) means the model is loaded ONCE and reused for
    every request. Loading it takes a few seconds — we don't want to repeat
    that every time a user asks a question.
    """
    settings = get_settings()
    model_name = settings.embedding_model
    logger.info("Loading embedding model...", model=model_name)
    
    # SentenceTransformer downloads the model on first run, then caches locally
    model = SentenceTransformer(model_name)
    logger.info("Embedding model loaded.", model=model_name)
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into their embedding vectors.
    
    Args:
        texts: List of strings to embed
    
    Returns:
        List of embedding vectors, each a list of floats.
        The length of each vector depends on the model:
        - all-MiniLM-L6-v2 → 384-dimensional vectors
    
    Example:
        embeddings = embed_texts(["Hello world", "Attendance policy"])
        # embeddings[0] → [0.1, -0.3, 0.7, ...]  (384 floats for "Hello world")
        # embeddings[1] → [0.2, 0.4, -0.1, ...]  (384 floats for "Attendance policy")
    """
    model = get_embedding_model()
    
    # batch encoding is efficient — the model processes all texts at once
    # normalize_embeddings=True makes cosine similarity = dot product
    # (Chroma uses this internally for faster computation)
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    
    # Convert numpy arrays to plain Python lists (required by Chroma)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    At search time, we need to embed the user's question to compare it
    against stored chunk embeddings.
    """
    return embed_texts([query])[0]
