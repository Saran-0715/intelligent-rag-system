"""
Intelligent RAG System — Vector Store Service (Chroma)
=======================================================
Manages the vector database: adding, searching, and deleting embeddings.

WHAT IS A VECTOR DATABASE?
A vector database stores embedding vectors and efficiently answers the query:
"Find the N vectors most similar to this query vector."

Traditional databases search by exact match (WHERE name = 'Alice').
Vector databases search by SIMILARITY — which is what we need for RAG because
user questions are paraphrased, not exact copies of document text.

WHY CHROMA?
- Local-first (data stays on your machine)
- Zero infrastructure setup (unlike Pinecone, Weaviate)
- Python-native
- Fast enough for thousands of documents
- Simple API

HOW CHROMA STORES DATA:
Chroma stores vectors in a "collection" (like a table in SQL).
Each entry has:
  - id: a unique identifier (we use chunk ID)
  - embedding: the vector [0.1, -0.3, 0.7, ...]
  - document: the text (Chroma can store it for you)
  - metadata: extra info like filename, page number, document_id

At query time, Chroma:
  1. Embeds the query (or uses a pre-computed embedding)
  2. Computes similarity between query vector and all stored vectors
  3. Returns the top-k most similar results
"""

from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.embedding_service import embed_texts, embed_query

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.ClientAPI:
    """
    Create a persistent Chroma client (cached as singleton).
    PersistentClient saves the vector database to disk so data
    survives application restarts.
    """
    settings = get_settings()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(settings.chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    logger.info("Chroma client initialized", path=str(settings.chroma_path))
    return client


def get_collection() -> chromadb.Collection:
    """
    Get or create the main Chroma collection.
    
    We use cosine similarity because our embeddings are normalized
    (normalize_embeddings=True in embed_texts).
    """
    settings = get_settings()
    client = get_chroma_client()
    
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity metric
    )
    return collection


def add_chunks_to_vector_store(
    chunk_ids: list[str],
    texts: list[str],
    metadatas: list[dict],
) -> None:
    """
    Embed and add text chunks to the vector store.
    
    Args:
        chunk_ids: Unique IDs for each chunk (matches database IDs)
        texts: The text content of each chunk
        metadatas: Metadata dicts (document_id, filename, page_number, etc.)
    
    HOW IT WORKS:
    1. embed_texts() converts each text to a 384-dim vector
    2. Chroma stores: id, embedding, text, metadata for each chunk
    3. Later, when you search, Chroma compares query vector to these stored vectors
    """
    if not texts:
        return

    logger.info("Embedding chunks for vector store", count=len(texts))
    embeddings = embed_texts(texts)

    collection = get_collection()
    
    # Chroma's add() handles batching internally
    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info("Chunks added to vector store", count=len(texts))


def search_similar_chunks(
    query: str,
    top_k: int | None = None,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """
    Search for the most semantically similar chunks to a query.
    
    Args:
        query: The user's natural-language question
        top_k: How many chunks to return
        document_ids: If provided, only search within these documents
                      (used for document-scoped search in Phase 3)
    
    Returns:
        List of dicts, each with:
            - chunk_id: str
            - text: str
            - metadata: dict (document_id, filename, page_number, etc.)
            - distance: float (cosine distance; lower = more similar)
            - score: float (1 - distance; higher = more similar)
    
    HOW SEMANTIC SEARCH WORKS:
    1. The user asks: "What happens if I miss too many classes?"
    2. embed_query() converts the question to a 384-dim vector
    3. Chroma compares this to ALL stored chunk vectors
    4. Returns the top 5 chunks with the smallest angular distance
    5. These chunks are about attendance — even though no word matched exactly!
    """
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k

    collection = get_collection()
    
    # Build query keyword args
    query_kwargs: dict = {
        "n_results": min(top_k, collection.count() or 1),
        "include": ["documents", "metadatas", "distances"],
    }
    
    # Embed the query and pass the vector directly
    # (This avoids Chroma needing its own embedding function)
    query_embedding = embed_query(query)
    query_kwargs["query_embeddings"] = [query_embedding]

    # Optional: filter by specific documents
    if document_ids:
        query_kwargs["where"] = {"document_id": {"$in": document_ids}}

    if collection.count() == 0:
        logger.warning("Vector store is empty — no documents indexed yet.")
        return []

    results = collection.query(**query_kwargs)

    # Flatten Chroma's nested result structure
    output = []
    for i, chunk_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        score = 1.0 - distance  # Convert distance to similarity score
        output.append({
            "chunk_id": chunk_id,
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": distance,
            "score": round(score, 4),
        })

    logger.info(
        "Vector search complete",
        query=query[:60],
        top_k=top_k,
        returned=len(output),
    )
    return output


def delete_document_chunks(document_id: str) -> int:
    """
    Remove all chunks belonging to a document from the vector store.
    Called when a document is deleted from the system.
    
    Returns: number of chunks deleted
    """
    collection = get_collection()
    
    # Query first to know what to delete
    existing = collection.get(
        where={"document_id": document_id},
        include=[],  # We only need IDs
    )
    
    ids_to_delete = existing["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info("Deleted chunks from vector store", document_id=document_id, count=len(ids_to_delete))
    
    return len(ids_to_delete)


def get_vector_store_stats() -> dict:
    """Return basic statistics about the vector store."""
    collection = get_collection()
    return {
        "total_chunks": collection.count(),
        "collection_name": collection.name,
    }
