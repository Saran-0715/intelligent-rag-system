# Intelligent RAG System

> **Phase 1 MVP** — A production-oriented Knowledge Retrieval and Intelligence Platform

A full-stack RAG (Retrieval-Augmented Generation) application that lets you upload documents, ask natural-language questions, and get AI-generated answers grounded in your own knowledge base — with source citations and evidence you can verify.

---

## 🚀 Quick Start (Windows / PowerShell)

### Prerequisites
- Python 3.12+ (Python 3.14 works)
- Node.js 18+
- A free [Groq API key](https://console.groq.com) (optional — works without it in retrieval-only mode)

### 1. Setup Backend

```powershell
# Navigate to backend directory
cd D:\Intelligent-RAG-System\backend

# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install dependencies (first time: downloads PyTorch ~2GB — be patient!)
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Then open .env and add your GROQ_API_KEY
notepad .env
```

### 2. Start the Backend

```powershell
# From D:\Intelligent-RAG-System\backend (with venv active)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API is now running at http://localhost:8000
Interactive API docs at http://localhost:8000/docs

### 3. Setup & Start Frontend

```powershell
# In a new PowerShell window
cd D:\Intelligent-RAG-System\frontend
npm install      # first time only
npm run dev
```

Frontend is now running at http://localhost:5173

### 4. Run Tests

```powershell
# From D:\Intelligent-RAG-System\backend (with venv active)
pytest tests/ -v
```

---

## 📁 Project Structure

```
Intelligent-RAG-System/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── documents.py    ← Upload, list, delete documents
│   │   │   ├── query.py        ← RAG pipeline endpoint
│   │   │   └── system.py       ← Health, dashboard, status
│   │   ├── core/
│   │   │   ├── config.py       ← All app settings (env vars)
│   │   │   ├── database.py     ← Async SQLite session management
│   │   │   └── logging.py      ← Structured logging
│   │   ├── models/
│   │   │   ├── database.py     ← SQLAlchemy ORM models
│   │   │   └── schemas.py      ← Pydantic request/response schemas
│   │   ├── services/
│   │   │   ├── document_processor.py  ← PDF/TXT extraction + chunking
│   │   │   ├── embedding_service.py   ← SentenceTransformers embeddings
│   │   │   ├── vector_store.py        ← Chroma vector DB operations
│   │   │   └── llm_service.py         ← Groq LLM generation
│   │   └── main.py             ← FastAPI app + startup
│   ├── tests/
│   │   └── test_phase1.py      ← Comprehensive test suite
│   ├── requirements.txt
│   ├── pytest.ini
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── Sidebar.jsx     ← Navigation sidebar
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx   ← Stats + pipeline overview
│   │   │   ├── KnowledgeBase.jsx ← Document management
│   │   │   └── AskAI.jsx       ← RAG query interface
│   │   ├── services/
│   │   │   └── api.js          ← All backend API calls
│   │   ├── App.jsx             ← Root component + routing
│   │   ├── index.css           ← Dark SaaS theme
│   │   └── main.jsx            ← React entry point
│   └── vite.config.js
├── data/
│   ├── sample_documents/       ← Test documents (fictional, clearly labeled)
│   ├── uploads/                ← Uploaded files (gitignored)
│   └── chroma_db/              ← Vector DB persistent storage (gitignored)
└── docs/
```

---

## 🔄 The RAG Pipeline

```
User Question
     │
     ▼
[1] embed_query()      ← Convert question to 384-dim vector
     │
     ▼
[2] Chroma search      ← Find top-5 most similar document chunks
     │
     ▼
[3] Build prompt       ← Inject retrieved passages + question
     │
     ▼
[4] Groq LLM           ← Generate grounded answer (or retrieval-only)
     │
     ▼
[5] Return response    ← Answer + source citations + timing
```

---

## ⚙️ Configuration

Edit `backend/.env`:

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key (free at console.groq.com) | (empty = retrieval-only) |
| `LLM_MODEL` | Groq model to use | `llama-3.1-8b-instant` |
| `EMBEDDING_MODEL` | SentenceTransformers model | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Characters per chunk | `512` |
| `CHUNK_OVERLAP` | Overlap between chunks | `64` |
| `RETRIEVAL_TOP_K` | Chunks to retrieve per query | `5` |

---

## 🧪 Test Checklist

After starting both servers, verify:
- [ ] http://localhost:8000/docs — Swagger UI loads
- [ ] http://localhost:8000/api/v1/system/health — Returns `{"status": "ok"}`
- [ ] http://localhost:5173 — Frontend loads with dark theme
- [ ] Upload `data/sample_documents/fictional_academic_policy.txt` via Knowledge Base
- [ ] Document status changes to "ready" with chunk count
- [ ] Ask: "What is the attendance policy?" — Returns retrieved passages
- [ ] Ask: "What happens if I fail an exam?" — Returns relevant results

---

## 📋 Development Phases

| Phase | Status | Description |
|---|---|---|
| Phase 1 | ✅ **Current** | Working RAG MVP (this implementation) |
| Phase 2 | 🔜 Planned | BM25 + hybrid search + reranking |
| Phase 3 | 🔜 Planned | Document versioning + cross-doc analysis |
| Phase 4 | 🔜 Planned | Citation checking + evaluation datasets |
| Phase 5 | 🔜 Planned | Auth + monitoring + deployment |

---

## ⚠️ Known Limitations (Phase 1)

1. **No authentication** — Anyone with backend access can read/write all documents
2. **SQLite only** — Single-user database; no concurrent writes in production
3. **Fixed-size chunking** — Semantic/paragraph-aware chunking planned for Phase 2
4. **No reranking** — Retrieved chunks are purely by cosine similarity score
5. **Document-only context** — LLM cannot search the web or access external sources
6. **Groq only** — Other LLM providers (OpenAI, Ollama) planned for Phase 5

---

*Built as a learning project. Phase 1 is a solid foundation — not a production deployment.*
