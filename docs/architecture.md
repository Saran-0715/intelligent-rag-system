# Intelligent RAG System — Architecture

## Phase 1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT BROWSER                       │
│                                                             │
│  ┌─────────┐  ┌─────────────────┐  ┌───────────────────┐  │
│  │Dashboard│  │  Knowledge Base │  │      Ask AI       │  │
│  └────┬────┘  └────────┬────────┘  └─────────┬─────────┘  │
│       │                │                      │             │
│       └────────────────┴──────────────────────┘            │
│                         React (Vite)                        │
└─────────────────────────────────│───────────────────────────┘
                                  │ HTTP (via Vite proxy /api/v1/*)
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND                        │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                      API Layer                         │ │
│  │  /documents/*   /query        /system/*                │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─────────────────┐  ┌──────────────────────────────────┐ │
│  │  Document       │  │         RAG Pipeline             │ │
│  │  Processor      │  │                                  │ │
│  │  ┌───────────┐  │  │  embed_query()                   │ │
│  │  │ PyMuPDF   │  │  │       ↓                          │ │
│  │  │ extractor │  │  │  Chroma.query()                  │ │
│  │  └─────┬─────┘  │  │       ↓                          │ │
│  │        ↓        │  │  build_rag_prompt()              │ │
│  │  ┌───────────┐  │  │       ↓                          │ │
│  │  │  Chunker  │  │  │  Groq API call                   │ │
│  │  │ 512-char  │  │  │       ↓                          │ │
│  │  │ overlap   │  │  │  Return QueryResponse            │ │
│  │  └─────┬─────┘  │  └──────────────────────────────────┘ │
│  │        ↓        │                                        │
│  │  ┌───────────┐  │                                        │
│  │  │ Sentence  │  │                                        │
│  │  │Transformers│  │                                       │
│  │  │ embedding │  │                                        │
│  │  └─────┬─────┘  │                                        │
│  └─────────│────────┘                                       │
│            │                                                │
└────────────│────────────────────────────────────────────────┘
             │
   ┌──────────────────────────────────┐
   │         STORAGE LAYER            │
   │                                  │
   │  ┌──────────┐  ┌──────────────┐  │
   │  │  SQLite  │  │   Chroma DB  │  │
   │  │          │  │              │  │
   │  │ Documents│  │  Embeddings  │  │
   │  │ Chunks   │  │  (vectors)   │  │
   │  │ QueryLog │  │              │  │
   │  └──────────┘  └──────────────┘  │
   └──────────────────────────────────┘
              │
   ┌──────────────────────────────────┐
   │       EXTERNAL SERVICES          │
   │                                  │
   │  Groq API (LLM generation)       │
   │  - llama-3.1-8b-instant          │
   │  - Only called at query time     │
   │  - No document data sent unless  │
   │    explicitly configured         │
   └──────────────────────────────────┘
```

## Key Design Decisions

### Why async (asyncio) throughout?
FastAPI is async-first. Using `async/await` and `aiosqlite` lets the server handle
many simultaneous requests efficiently without spawning new threads per request.

### Why SQLite + Chroma (two databases)?
- **SQLite** stores structured metadata: document filenames, chunk text, page numbers,
  query logs. It's great for `SELECT WHERE` queries and joins.
- **Chroma** stores vectors and answers "find the most similar vector to this query".
  SQL databases are very slow at this — vector databases are purpose-built for it.

### Why background tasks for document processing?
Embedding a large PDF can take 10–30 seconds. If we did this synchronously, the HTTP
upload request would hang. Instead, we return 202 Accepted immediately and process
in the background. The frontend polls `GET /documents/{id}` until status = "ready".

### Why sentence-transformers locally?
- **Privacy**: Documents never leave your machine for embedding
- **Cost**: Zero API cost for embeddings
- **Latency**: Local inference is fast once the model is loaded
- `all-MiniLM-L6-v2` is 90MB, produces 384-dim vectors, is fast and accurate

### Why Groq for LLM?
- Generous free tier (thousands of requests/day)
- Very fast inference (~100 tokens/second)
- Compatible with OpenAI's API format
- Easy to swap to OpenAI or local Ollama later

## Data Flow: Document Upload

```
POST /api/v1/documents/upload
         │
         ▼
   Validate (type, size)
         │
         ▼
   Save file to disk
   (data/uploads/*.pdf)
         │
         ▼
   Create Document record
   in SQLite (status=processing)
         │
         ▼
   Return 202 + document_id
   (immediate response to user)
         │
         ▼ (background task)
   extract_text() via PyMuPDF
         │
         ▼
   chunk_text_with_pages()
   → List[TextChunk]
         │
         ▼
   Save chunks to SQLite
         │
         ▼
   embed_texts() via SentenceTransformers
   → List[List[float]]
         │
         ▼
   add_chunks_to_vector_store()
   → Chroma upsert
         │
         ▼
   Update Document.status = "ready"
```

## Data Flow: Query (RAG Pipeline)

```
POST /api/v1/query {question: "..."}
         │
         ▼
   embed_query(question)
   → [0.23, -0.11, ...] (384 floats)
         │
         ▼
   chroma.query(query_embeddings=[...], n_results=5)
   → [chunk_id, text, metadata, distance] × 5
         │
         ▼
   Enrich with SQLite (get filename from document_id)
         │
         ▼
   build_rag_prompt(question, passages)
   → "Sources: [1] filename.pdf Page 3: ...\nQuestion: ..."
         │
         ▼
   groq.chat.completions.create(prompt)
   → answer text
         │
         ▼
   Log to QueryLog table
         │
         ▼
   Return QueryResponse {answer, sources, timing}
```

## Phase 2+ Planned Changes

| Feature | Phase | Implementation Plan |
|---|---|---|
| BM25 keyword search | 2 | `rank_bm25` library on stored chunk text |
| Hybrid retrieval | 2 | Merge BM25 + vector scores (RRF algorithm) |
| Reranking | 2 | `cross-encoder/ms-marco-MiniLM-L-6-v2` model |
| Semantic chunking | 2 | Sentence boundary + semantic similarity split |
| Document versioning | 3 | Add `version` field + re-index on change |
| Multiple collections | 3 | Chroma collection per knowledge domain |
| Citation checking | 4 | Verify LLM claims against retrieved passages |
| Evaluation suite | 4 | RAGAS or custom metrics |
| Authentication | 5 | FastAPI OAuth2 / JWT |
| Access control | 5 | Document-level permissions in SQLite |
| Monitoring | 5 | OpenTelemetry traces + Prometheus metrics |
