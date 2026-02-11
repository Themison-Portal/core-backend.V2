# Themison Backend

FastAPI backend for the Themison clinical trials document Q&A platform, powered by RAG (Retrieval-Augmented Generation).

## Prerequisites

- Python 3.10+
- Docker & Docker Compose (for local development)
- OpenAI API key (for embeddings)
- Anthropic API key (for LLM)

## Quick Start with Docker (Recommended)

The easiest way to run the backend locally is with Docker.

### 1. Start Docker services

```bash
cd core-backend.V2
docker-compose up -d
```

This starts:
- **PostgreSQL** with pgvector on `localhost:54322`
- **Redis** on `localhost:6379`

### 2. Configure environment

```bash
cp .env.local .env
```

Edit `.env` and add your API keys:
```env
OPENAI_API_KEY=sk-proj-your-key
ANTHROPIC_API_KEY=sk-ant-your-key
```

### 3. Install Python dependencies

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### 4. Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Verify it works

- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Reset database (delete all data)
docker-compose down -v
docker-compose up -d
```

---

## Manual Installation (Without Docker)

### 1. Clone and navigate to backend

```bash
cd core-backend.V2
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# OpenAI API (for embeddings)
OPENAI_API_KEY=sk-proj-your-key

# Anthropic API (for LLM)
ANTHROPIC_API_KEY=sk-ant-your-key

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_DB_URL=postgresql+asyncpg://postgres:password@host:5432/postgres

# Redis (for caching)
REDIS_URL=redis://localhost:6379

# Frontend URL (for CORS)
FRONTEND_URL=http://localhost:8080
```

### 5. Database setup

Ensure pgvector extension is enabled in your PostgreSQL database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Run the performance optimization migration:

```bash
psql $SUPABASE_DB_URL -f migrations/add_pgvector_index.sql
```

## Running the Server

### Development

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the start script:

```bash
./start.sh
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once the server is running, access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Project Structure

```
app/
├── main.py                 # FastAPI entry point
├── config.py               # Environment configuration
├── api/
│   └── routes/             # API endpoints
│       ├── auth.py         # Authentication routes
│       ├── upload.py       # Document upload/ingestion
│       └── query.py        # RAG query endpoint
├── services/
│   ├── doclingRag/         # RAG implementation
│   │   ├── rag_ingestion_service.py   # PDF parsing & embedding
│   │   ├── rag_retrieval_service.py   # Vector similarity search
│   │   └── rag_generation_service.py  # LLM response generation
│   ├── cache/              # Redis caching layer
│   └── highlighting/       # PDF highlight service
├── models/                 # SQLAlchemy ORM models
├── contracts/              # Pydantic DTOs
├── dependencies/           # FastAPI dependency injection
├── core/                   # Singletons (OpenAI, Supabase clients)
└── db/                     # Database session management
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/auth/me` | Get current user (requires JWT) |
| POST | `/upload/upload-pdf` | Ingest PDF document |
| POST | `/query` | RAG query with citations |
| GET | `/query/highlighted-pdf` | Get PDF with highlighted sources |
| GET | `/query/get-chat-history` | Retrieve chat history |

## Architecture Overview

### RAG Pipeline

1. **Ingestion** (`/upload/upload-pdf`)
   - PDF parsed via Docling
   - Split into 750-token chunks (HybridChunker)
   - Embedded with OpenAI `text-embedding-3-small`
   - Stored in PostgreSQL with pgvector

2. **Query** (`/query`)
   - Query embedded with same model
   - HNSW index search for similar chunks
   - GPT-4o-mini generates answer with citations
   - Response includes bounding boxes for PDF highlighting

### Caching Strategy

Three-tier Redis caching for performance:

| Cache | TTL | Purpose |
|-------|-----|---------|
| Embeddings | 24h | Query embeddings (deterministic) |
| Chunks | 1h | Retrieved document chunks |
| Responses | 30min | LLM-generated answers |

Cache is automatically invalidated when documents are re-uploaded.

### Authentication

All protected endpoints require Supabase JWT bearer tokens:

```bash
curl -H "Authorization: Bearer <jwt_token>" http://localhost:8000/auth/me
```

## Database Tables

| Table | Purpose |
|-------|---------|
| `trial_documents` | Document metadata and storage URLs |
| `document_chunks_docling` | Chunks with pgvector embeddings |
| `chat_sessions` | Conversation sessions |
| `chat_messages` | Individual messages |
| `chat_document_links` | Session-document relations |

## Performance

Recent optimizations (see `LLM_SEARCH_CHANGES.md`):

| Metric | Before | After |
|--------|--------|-------|
| Query (cache miss) | 3-8s | ~2s |
| Query (cache hit) | 3-8s | ~10-50ms |
| Vector search | 500-2000ms | 10-50ms |

## Troubleshooting

### Redis connection error
Ensure Redis is running:
```bash
redis-server
```

### pgvector extension not found
Enable in Supabase SQL Editor:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Slow vector search
Run the index migration:
```bash
psql $SUPABASE_DB_URL -f migrations/add_pgvector_index.sql
```

### Import errors
Ensure you're in the virtual environment:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

## Related Documentation

- `LLM_SEARCH_CHANGES.md` - Performance optimization details
- `.env.example` - Environment variable reference
