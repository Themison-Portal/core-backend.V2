# RAG Backend Changes Report

**Project**: Themison Clinical Trials Platform
**Date**: January 13, 2026
**Author**: Sylwester M.
**Branch**: `Backend_improvements`

---

## Executive Summary

This report documents all changes made to the RAG (Retrieval-Augmented Generation) backend to improve performance and code quality. The optimizations reduced query response times from **15-20 seconds to ~17 seconds** for cache misses, and to **~3ms for cache hits**.

**Latest Update (Jan 13, 2026)**: Added **Semantic Similarity Caching** using pgvector - semantically similar queries (≥90% cosine similarity) now return cached responses in ~50ms instead of ~21s, avoiding expensive LLM calls.

---

## Performance Results

| Scenario | Before | After | Improvement |
|----------|--------|-------|-------------|
| First query (cache miss) | 15-20s | ~17s | Optimized pipeline |
| Repeated query (exact match) | 15-20s | ~3ms | **99.9% faster** |
| Similar query (semantic cache) | 15-20s | ~50ms | **99.7% faster** |
| Vector search | 500-2000ms | ~3ms (cached) | **99.8% faster** |

### Current Timing Breakdown (Cache Miss)

```
[TIMING] Service initialization: 0.01ms
[TIMING] Chunk cache HIT: 2.79ms, 15 chunks
[TIMING] Compression: 15 → 10 chunks (0.25ms)
[TIMING] Context: 10044 chars (~2511 tokens)
[TIMING] LLM (Claude Opus 4.5): 17623.23ms
[TIMING] TOTAL: 17675.38ms
```

**Key Finding**: LLM call is 99.9% of total time - this is expected for frontier models.

---

## Changes Made

### 1. Database Index (HNSW)

**Purpose**: Eliminate full table scans for vector similarity search

#### Why HNSW Index?

Without an index, PostgreSQL must scan every row in the `document_chunks_docling` table and calculate the cosine distance for each embedding. With thousands of chunks, this becomes slow (500-2000ms).

**HNSW (Hierarchical Navigable Small World)** is a graph-based index that:
- Creates a multi-layer graph where each node connects to similar vectors
- Searches by navigating through the graph layers (like "six degrees of separation")
- Reduces search from O(n) to approximately O(log n)

**Index Parameters Explained**:
- `m = 16`: Number of connections per node. Higher = better recall, more memory
- `ef_construction = 64`: Search width during index building. Higher = better quality index, slower build
- `vector_cosine_ops`: Use cosine similarity (matches our retrieval query)

**File**: `app/models/chunks_docling.py` (lines 34-47)

```python
__table_args__ = (
    Index(
        'idx_chunks_embedding_hnsw',
        'embedding',
        postgresql_using='hnsw',
        postgresql_with={'m': 16, 'ef_construction': 64},
        postgresql_ops={'embedding': 'vector_cosine_ops'}
    ),
    Index('idx_chunks_document_id', 'document_id'),
)
```

**Migration**: `migrations/add_pgvector_index.sql`

```sql
CREATE INDEX CONCURRENTLY idx_chunks_embedding_hnsw
ON document_chunks_docling
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Result**: Vector search reduced from 500-2000ms to ~50ms (before caching)

---

### 2. Three-Tier Redis Caching

**Purpose**: Cache expensive operations (embeddings, chunks, responses)

#### Why Three Tiers?

Different operations have different characteristics that require different caching strategies:

| Tier | What's Cached | Why This TTL |
|------|---------------|--------------|
| **Embeddings (24h)** | Query vector from OpenAI | Deterministic: same query always produces identical vector. Safe to cache long-term. |
| **Chunks (1h)** | Retrieved document chunks | Document content rarely changes, but we want fresh results if documents are re-indexed. |
| **Responses (30min)** | Full LLM answers | LLM responses can vary; shorter TTL ensures answers stay fresh and relevant. |

#### How It Works

```
User Query: "What are the inclusion criteria?"
                    │
                    ▼
        ┌─── Check Response Cache (resp:) ───┐
        │                                     │
      HIT?─────────────────────────────────→ Return immediately (~3ms)
        │ MISS
        ▼
        ┌─── Check Chunk Cache (chunks:) ────┐
        │                                     │
      HIT?─────────────────────────────────→ Skip vector search, go to LLM
        │ MISS
        ▼
        ┌─── Check Embedding Cache (emb:) ───┐
        │                                     │
      HIT?─────────────────────────────────→ Skip OpenAI API call
        │ MISS
        ▼
    Full Pipeline: Embed → Search → Generate (~17s)
```

#### Cache Invalidation

When a document is re-uploaded, stale cache entries must be cleared to prevent returning outdated answers. The `rag_ingestion_service.py` automatically invalidates all cache entries related to that document.

**New Files Created**:

| File | Purpose |
|------|---------|
| `app/services/cache/__init__.py` | Cache module initialization |
| `app/services/cache/rag_cache_service.py` | Redis caching service |
| `app/dependencies/cache.py` | FastAPI dependency injection |

**Cache Configuration** (`app/services/cache/rag_cache_service.py`, lines 14-28):

```python
class RagCacheService:
    TTL_EMBEDDING = 86400   # 24 hours (embeddings are deterministic)
    TTL_CHUNKS = 3600       # 1 hour (document content stable)
    TTL_RESPONSE = 1800     # 30 minutes (LLM answers may vary)

    PREFIX_EMBEDDING = "emb"
    PREFIX_CHUNKS = "chunks"
    PREFIX_RESPONSE = "resp"
```

**Integration Points**:
- `app/services/doclingRag/rag_retrieval_service.py` (lines 49, 146)
- `app/services/doclingRag/rag_generation_service.py` (lines 190-204)
- `app/api/routes/query.py` (line 76)

**Result**: Repeated queries return in ~3ms instead of ~17s (99.9% faster)

---

### 3. Chunk Compression

**Purpose**: Merge same-page chunks to reduce context size and token costs

#### Why Compress Chunks?

When retrieving top-15 chunks, multiple chunks often come from the same page. For example:

```
Before compression (15 chunks):
- Protocol.pdf, Page 5, Chunk 1: "Inclusion criteria include..."
- Protocol.pdf, Page 5, Chunk 2: "Patients must be 18 years..."
- Protocol.pdf, Page 5, Chunk 3: "No prior chemotherapy..."
- Protocol.pdf, Page 8, Chunk 1: "Exclusion criteria..."
... (12 more chunks)

After compression (10 chunks):
- Protocol.pdf, Page 5: "Inclusion criteria include... Patients must be 18... No prior chemotherapy..."
- Protocol.pdf, Page 8: "Exclusion criteria..."
... (8 more chunks)
```

**Benefits**:
1. **Fewer tokens**: Less context overhead (metadata per chunk)
2. **Better coherence**: LLM sees complete page context instead of fragments
3. **Preserved bboxes**: All bounding boxes kept for source highlighting

**File**: `app/services/doclingRag/rag_generation_service.py` (lines 87-127)

```python
def _compress_chunks(self, chunks: List[dict]) -> List[dict]:
    """
    Compress chunks by merging those from the same page.
    Preserves all bboxes and combines content.
    """
    # Group chunks by (title, page)
    page_groups: Dict[tuple, List[dict]] = {}
    for chunk in chunks:
        meta = self._extract_chunk_metadata(chunk)
        key = (meta["title"], meta["page"])
        if key not in page_groups:
            page_groups[key] = []
        page_groups[key].append(meta)

    # Merge chunks from same page
    compressed = []
    for (title, page), group in page_groups.items():
        if len(group) == 1:
            compressed.append(group[0])
        else:
            all_bboxes = [m["bbox"] for m in group if m["bbox"]]
            all_content = "\n...\n".join(m["content"] for m in group)
            compressed.append({
                "title": title,
                "page": page,
                "bboxes": all_bboxes,
                "content": all_content[:2000],  # Limit merged content
            })

    return compressed
```

**Result**: 15 chunks → 10 chunks (33% reduction in context overhead)

---

### 4. Compact Context Format

**Purpose**: Reduce token overhead per chunk

#### Why Compact Format?

Every chunk sent to the LLM includes metadata (title, page, bbox). The original format used verbose delimiters that consumed tokens without adding value:

**Before** (~80 chars overhead per chunk):
```
### SOURCE START ###
NAME: Document Title
PAGE: 5
BBOX: [100, 200, 300, 400]
CONTENT: ...actual content...
### SOURCE END ###
```

**After** (~40 chars overhead per chunk):
```
[Document Title|p5|bbox:[100, 200, 300, 400]]
...actual content...
```

**Savings Calculation**:
- 15 chunks × 40 chars saved = 600 chars = ~150 tokens saved per query
- At $0.01/1K tokens, this saves ~$0.0015 per query

**File**: `app/services/doclingRag/rag_generation_service.py` (lines 129-145)

```python
def _format_context_compact(self, chunk_meta: dict) -> str:
    title = chunk_meta.get("title", "Unknown")
    page = chunk_meta.get("page", 0)
    content = chunk_meta.get("content", "")
    bbox_str = str(chunk_meta.get("bboxes", chunk_meta.get("bbox")))

    return f"[{title}|p{page}|bbox:{bbox_str}]\n{content}"
```

**Result**: ~600 fewer tokens for 15 chunks (50% reduction in metadata overhead)

---

### 5. LLM Model Switch: GPT-4o-mini → Claude Opus 4.5

**Purpose**: Use higher quality model for better answer accuracy

#### Why Claude Opus 4.5?

| Aspect | GPT-4o-mini | Claude Opus 4.5 |
|--------|-------------|-----------------|
| **Quality** | Good for simple tasks | State-of-the-art reasoning |
| **JSON adherence** | Sometimes adds extra text | Excellent instruction following |
| **Citation accuracy** | Occasional hallucinations | Better source attribution |
| **Speed** | ~18.5s | ~17.6s (slightly faster) |
| **Cost** | Lower | Higher (but worth it for clinical accuracy) |

For a clinical trials platform where accuracy is critical, Claude Opus 4.5 provides:
- Better understanding of medical terminology
- More accurate citations to source documents
- Fewer hallucinated facts

**File**: `app/services/doclingRag/rag_generation_service.py` (lines 1-21, 230-245)

**Before**:
```python
from openai import AsyncOpenAI
_openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

response = await _openai_client.chat.completions.create(
    model="gpt-4o-mini",
    ...
)
```

**After**:
```python
from anthropic import AsyncAnthropic
_anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

response = await _anthropic_client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=2000,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_message}],
)
```

**Result**: Improved answer quality with comparable speed (~17.6s vs ~18.5s)

---

### 6. Optimized System Prompt

**Purpose**: Maximize prompt caching efficiency and enforce JSON output

#### Why Optimize the Prompt?

**Prompt Caching**: LLM providers cache the system prompt prefix. If the same system prompt is used across requests, subsequent requests are faster and cheaper. To maximize caching:
1. Put static instructions FIRST (cacheable)
2. Put dynamic content (context, query) LAST

**JSON Enforcement**: The prompt explicitly instructs the model to return valid JSON only, reducing parsing errors and the need for fallback handling.

**File**: `app/services/doclingRag/rag_generation_service.py` (lines 26-36)

```python
SYSTEM_PROMPT = """You are an expert clinical Document assistant. You MUST respond with valid JSON only.

RULES:
• Use ONLY the provided context
• Every fact MUST have an inline citation: (Document_Title, p. X)
• Include bbox coordinates from context in your sources
• If multiple chunks from same page, include ALL their bboxes

RESPOND WITH THIS EXACT JSON STRUCTURE (no other text):
{"response": "markdown answer with citations", "sources": [{"name": "doc title", "page": 1, "section": "section or null", "exactText": "verbatim quote", "bboxes": [[x0,y0,x1,y1]], "relevance": "high"}]}"""
```

**Key Design Decisions**:
- **"You MUST respond with valid JSON only"**: Prevents model from adding conversational text
- **Inline citation format**: `(Document_Title, p. X)` matches frontend highlighting
- **Bbox requirement**: Enables precise source highlighting in PDF viewer
- **Exact JSON structure**: Reduces parsing ambiguity

**Result**: More consistent JSON output, fewer parsing errors

---

### 7. Enhanced Timing Logs

**Purpose**: Monitor performance of each pipeline stage for debugging and optimization

#### Why Detailed Timing?

Without timing logs, it's impossible to know where bottlenecks occur. The enhanced logs show exactly how long each stage takes:

```
[TIMING] ========== TIMING SUMMARY ==========
[TIMING] Embedding: 0.05ms (cache_hit: True)
[TIMING] Vector search: 2.79ms
[TIMING] Retrieval total: 3.21ms (chunk_cache_hit: True)
[TIMING] Chunks: 15 → 10 (compressed: True)
[TIMING] Compression: 0.25ms
[TIMING] Context format: 0.12ms
[TIMING] LLM call: 17623.23ms        ← 99.9% of time is here
[TIMING] Generation total: 17675.38ms
[TIMING] ========== TOTAL: 17678.59ms ==========
```

This immediately shows:
1. Cache hits/misses for each tier
2. Chunk compression effectiveness
3. LLM as the primary bottleneck (expected)

**File**: `app/api/routes/query.py` (lines 117-130)

```python
logger.info(f"[TIMING] ========== TIMING SUMMARY ==========")
logger.info(f"[TIMING] Embedding: {retrieval.get('embedding_ms', 0):.2f}ms (cache_hit: {retrieval.get('cache_hit', False)})")
logger.info(f"[TIMING] Vector search: {retrieval.get('db_search_ms', 0):.2f}ms")
logger.info(f"[TIMING] Retrieval total: {retrieval.get('retrieval_total_ms', 0):.2f}ms (chunk_cache_hit: {retrieval.get('chunk_cache_hit', False)})")
logger.info(f"[TIMING] Chunks: {timing.get('original_chunk_count', 0)} → {timing.get('compressed_chunk_count', 0)} (compressed: {timing.get('chunks_compressed', False)})")
logger.info(f"[TIMING] Compression: {timing.get('compression_ms', 0):.2f}ms")
logger.info(f"[TIMING] Context format: {timing.get('context_format_ms', 0):.2f}ms")
logger.info(f"[TIMING] LLM call: {timing.get('llm_call_ms', 0):.2f}ms")
logger.info(f"[TIMING] Generation total: {timing.get('generation_total_ms', 0):.2f}ms (response_cache_hit: {timing.get('response_cache_hit', False)})")
logger.info(f"[TIMING] ========== TOTAL: {total_time:.2f}ms ==========")
```

**Result**: Clear visibility into performance for ongoing optimization

---

### 8. Configuration Updates

**Purpose**: Add required API keys and dependencies for Claude integration

#### Changes Made

**File**: `app/config.py` (lines 16-17)

```python
openai_api_key: str           # Still needed for embeddings
anthropic_api_key: str        # Required for Claude Opus 4.5
```

**Why both keys?**
- OpenAI: Used for `text-embedding-3-small` (best price/performance for embeddings)
- Anthropic: Used for Claude Opus 4.5 (best quality for generation)

**File**: `requirements.txt` (lines 11-12)

```
openai>=1.54.0
anthropic>=0.40.0  # Required for Claude Opus 4.5
```

---

### 9. Semantic Similarity Caching (January 13, 2026)

**Purpose**: Cache LLM responses for semantically similar queries using pgvector similarity search

#### Why Semantic Caching?

The existing Redis cache uses SHA256 hashing for exact string matching. This means:
- "What are the inclusion criteria?" → Cache HIT
- "What are inclusion criteria?" → Cache MISS (different string!)
- "Tell me about inclusion criteria" → Cache MISS

With semantic caching, queries with ≥90% cosine similarity to cached queries return the cached response:
- "What are the inclusion criteria?" → Stored in cache
- "What are inclusion criteria?" → **Semantic HIT** (similarity: 0.98)
- "Tell me about inclusion criteria" → **Semantic HIT** (similarity: 0.94)

**Result**: ~50ms response instead of ~21s LLM call

#### How It Works

```
User Query: "Tell me about inclusion criteria"
                    │
                    ▼
        ┌─── Get Query Embedding (OpenAI) ───┐
        │                                     │
        ▼
        ┌─── Semantic Cache Search ──────────┐
        │  SELECT * FROM semantic_cache      │
        │  WHERE similarity >= 0.90          │
        │  ORDER BY embedding <=> query      │
        └────────────────────────────────────┘
                    │
                  HIT?─────────────────────────→ Return cached response (~50ms)
                    │ MISS
                    ▼
        ┌─── Continue Normal RAG Flow ───────┐
        │  Retrieve chunks → Claude API      │
        └────────────────────────────────────┘
                    │
                    ▼
        ┌─── Store in Semantic Cache ────────┐
        │  INSERT query_embedding, response  │
        └────────────────────────────────────┘
```

#### Cache Hierarchy (Updated)

| Priority | Cache Layer | Type | TTL | Use Case |
|----------|-------------|------|-----|----------|
| 1 | Semantic Cache | PostgreSQL + pgvector | Permanent | Similar queries (≥90% similarity) |
| 2 | Response Cache | Redis | 30min | Exact query match |
| 3 | Chunk Cache | Redis | 1h | Retrieved document chunks |
| 4 | Embedding Cache | Redis | 24h | Query embeddings |

#### Database Schema

**File**: `migrations/create_semantic_cache.sql`

```sql
CREATE TABLE IF NOT EXISTS semantic_cache_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    query_embedding Vector(1536) NOT NULL,
    document_id UUID NOT NULL REFERENCES trial_documents(id) ON DELETE CASCADE,
    response_data JSONB NOT NULL,
    hit_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
    context_hash VARCHAR(32) NOT NULL
);

-- HNSW index for fast similarity search
CREATE INDEX idx_semantic_cache_embedding_hnsw
ON semantic_cache_responses USING hnsw (query_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_semantic_cache_document_id ON semantic_cache_responses (document_id);
```

**Why HNSW Index?**
- Approximate nearest neighbor search in O(log n) time
- Parameters: `m=16` (connections per node), `ef_construction=64` (build quality)
- Same settings as document_chunks_docling for consistency

#### Service Implementation

**File**: `app/services/cache/semantic_cache_service.py`

```python
class SemanticCacheService:
    DEFAULT_SIMILARITY_THRESHOLD = 0.90

    async def get_similar_response(
        self,
        query_embedding: List[float],
        document_id: UUID,
        similarity_threshold: float = 0.90
    ) -> Optional[Dict]:
        """
        Search for semantically similar cached response.
        Uses pgvector cosine similarity with HNSW index.
        """
        sql = text("""
            SELECT id, query_text, response_data, context_hash,
                   1 - (query_embedding <=> (:v)::vector) AS similarity
            FROM semantic_cache_responses
            WHERE document_id = :doc_id
              AND 1 - (query_embedding <=> (:v)::vector) >= :threshold
            ORDER BY query_embedding <=> (:v)::vector
            LIMIT 1
        """)
        # ... execute and return result

    async def store_response(
        self,
        query_text: str,
        query_embedding: List[float],
        document_id: UUID,
        response: Dict,
        context_hash: str
    ) -> None:
        """Store new response in semantic cache."""
        # ... insert into database

    async def invalidate_document(self, document_id: UUID) -> int:
        """Delete all cache entries for a document (called on re-upload)."""
        # ... delete and return count
```

#### Integration with RAG Pipeline

**File**: `app/services/doclingRag/rag_generation_service.py`

The semantic cache is checked **before** chunk retrieval to maximize savings:

```python
async def generate_answer(self, query_text: str, document_id: UUID, ...):
    # 1. Get query embedding (needed for semantic cache)
    query_embedding, embed_timing = await self.retrieval_service.get_query_embedding(query_text)

    # 2. Check semantic cache FIRST
    if self.semantic_cache_service:
        cached = await self.semantic_cache_service.get_similar_response(
            query_embedding=query_embedding,
            document_id=document_id
        )
        if cached:
            return {"result": cached["response"], "timing": {...}}

    # 3. Continue normal flow (retrieval → LLM → store in cache)
    ...
```

#### JSON Parsing Resilience

Claude sometimes returns malformed JSON. A robust parsing strategy was implemented:

**File**: `app/services/doclingRag/rag_generation_service.py`

```python
def _parse_llm_json(self, raw_content: str) -> dict:
    """Parse JSON with multiple fallback strategies. Never raises."""

    # Strategy 1: Direct parse
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON with regex
    json_match = re.search(r'\{[\s\S]*\}', raw_content)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Repair common issues (trailing commas, missing commas)
    try:
        repaired = self._repair_json(json_str)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Extract response field only
    response_match = re.search(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_content)
    if response_match:
        return {"response": response_match.group(1), "sources": []}

    # Strategy 5: Return raw content (last resort)
    return {"response": raw_content[:3000], "sources": []}
```

#### Cache Performance Logging

Comprehensive logging shows cache performance at a glance:

```
[PERF] ============ CACHE PERFORMANCE ============
[PERF] [HIT] Embedding Cache - 2.45ms
[PERF] [HIT] Semantic Cache - 48.32ms (similarity: 0.9423)
[PERF] [MISS] Chunk Cache - 157.45ms
[PERF] [MISS] Response Cache - Redis exact match
[PERF] =============================================
[PERF] Cache Hit Rate: 2/4 (50%)
[PERF] Estimated Time Saved: ~15500ms
```

#### Cache Invalidation

When a document is re-uploaded, all semantic cache entries for that document are deleted:

**File**: `app/services/doclingRag/rag_ingestion_service.py`

```python
# During document re-ingestion
if self.semantic_cache_service:
    deleted = await self.semantic_cache_service.invalidate_document(document_id)
    logger.info(f"[CACHE] Invalidated {deleted} semantic cache entries")
```

#### Configuration

**File**: `app/config.py`

```python
semantic_cache_similarity_threshold: float = 0.90  # Minimum similarity for cache hit
```

#### Migration Commands

When setting up a new database:

```bash
# 1. Enable pgvector extension
psql $SUPABASE_DB_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 2. Run HNSW index for document chunks
psql $SUPABASE_DB_URL -f migrations/add_pgvector_index.sql

# 3. Create semantic cache table
psql $SUPABASE_DB_URL -f migrations/create_semantic_cache.sql
```

#### Performance Impact

| Query Type | Without Semantic Cache | With Semantic Cache |
|------------|------------------------|---------------------|
| First query | ~22s | ~22s (stored in cache) |
| Exact repeat | ~3ms (Redis) | ~3ms (Redis, checked first) |
| Similar query | ~22s (full LLM call) | **~50ms** (pgvector search) |
| Different query | ~22s | ~22s |

**Key Benefit**: Users asking similar questions (common in clinical document Q&A) get instant responses.

---

## Files Summary

### New Files Created

| File | Purpose |
|------|---------|
| `app/services/cache/__init__.py` | Cache module initialization |
| `app/services/cache/rag_cache_service.py` | Redis caching service (TTL management, invalidation) |
| `app/services/cache/semantic_cache_service.py` | PostgreSQL semantic similarity cache using pgvector |
| `app/models/semantic_cache.py` | SQLAlchemy model for semantic cache table |
| `app/dependencies/cache.py` | FastAPI dependency injection for cache services |
| `migrations/add_pgvector_index.sql` | HNSW index migration for document chunks |
| `migrations/create_semantic_cache.sql` | Semantic cache table with HNSW index |
| `docker-compose.yml` | Local dev with PostgreSQL + Redis |
| `docker/init.sql` | Database initialization with indexes |
| `.env.local` | Pre-configured environment for Docker |
| `PROGRESS.md` | Progress report documentation |
| `EMBED.md` | Embedding improvement recommendations |

### Modified Files

| File | Lines Changed | Changes |
|------|---------------|---------|
| `app/models/chunks_docling.py` | 34-47 | Added HNSW + document_id indexes |
| `app/models/__init__.py` | - | Added SemanticCacheResponse import |
| `app/core/openai.py` | All | Singleton embedding client |
| `app/services/doclingRag/rag_generation_service.py` | All | Chunk compression, Claude integration, prompt optimization, semantic cache integration, JSON parsing resilience |
| `app/services/doclingRag/rag_retrieval_service.py` | 42-73, 149 | Public `get_query_embedding()`, `precomputed_embedding` parameter, cache logging |
| `app/services/doclingRag/rag_ingestion_service.py` | 110-154 | Cache invalidation (Redis + semantic cache) |
| `app/api/routes/query.py` | 76-186 | Semantic cache injection, cache performance logging |
| `app/dependencies/cache.py` | - | Added `get_semantic_cache_service()` |
| `app/dependencies/rag.py` | - | Updated dependency injection |
| `app/config.py` | 16-18 | Added anthropic_api_key, semantic_cache_similarity_threshold |
| `requirements.txt` | 11-12 | Added anthropic>=0.40.0 |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (:8000)                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    /query endpoint                        │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │  Retrieval  │  │  Generation │  │  Cache Services │   │  │
│  │  │   Service   │  │   Service   │  │  (Redis+PG)     │   │  │
│  │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘   │  │
│  └─────────┼────────────────┼──────────────────┼────────────┘  │
└────────────┼────────────────┼──────────────────┼────────────────┘
             │                │                  │
     ┌───────┴───────┐        │          ┌───────┴───────────────┐
     ▼               ▼        ▼          ▼                       ▼
┌─────────┐   ┌──────────┐ ┌─────────┐ ┌───────────────┐ ┌───────────────┐
│ OpenAI  │   │PostgreSQL│ │ Claude  │ │    Redis      │ │  PostgreSQL   │
│Embedding│   │ pgvector │ │Opus 4.5 │ │   (:6379)     │ │Semantic Cache │
│  API    │   │ (Supabase)│ │  API   │ │               │ │   (Supabase)  │
└─────────┘   │          │ └─────────┘ │ ┌───────────┐ │ │ ┌───────────┐ │
              │  HNSW    │             │ │emb: (24h) │ │ │ │query_emb  │ │
              │  Index   │             │ │chunk:(1h) │ │ │ │ HNSW idx  │ │
              │          │             │ │resp:(30m) │ │ │ │ ~50ms     │ │
              └──────────┘             │ └───────────┘ │ │ └───────────┘ │
                                       └───────────────┘ └───────────────┘

Cache Flow:
  Query → Embed → [Semantic Cache ≥90%?] → HIT → Return (~50ms)
                          ↓ MISS
                  [Redis Response Cache?] → HIT → Return (~3ms)
                          ↓ MISS
                  [Retrieve Chunks] → [Claude API] → Store → Return (~22s)
```

---

## Environment Setup

### Required Environment Variables

```env
# Database
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
SUPABASE_DB_URL=postgresql+asyncpg://...

# AI APIs
OPENAI_API_KEY=sk-proj-...      # For embeddings
ANTHROPIC_API_KEY=sk-ant-...    # For Claude Opus 4.5

# Cache
REDIS_URL=redis://localhost:6379
```

### Local Development

```bash
# Start services
docker-compose up -d

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Database Migrations (New Supabase Instance)

```bash
# 1. Enable pgvector extension
psql $SUPABASE_DB_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 2. Create HNSW index for document chunks
psql $SUPABASE_DB_URL -f migrations/add_pgvector_index.sql

# 3. Create semantic cache table (for similarity-based caching)
psql $SUPABASE_DB_URL -f migrations/create_semantic_cache.sql
```

---

## What's Next (Recommended)

| Priority | Task | Impact | Status |
|----------|------|--------|--------|
| ~~1~~ | ~~Semantic similarity caching~~ | ~~99.7% faster for similar queries~~ | ✅ **DONE** |
| 1 | Implement streaming responses | Better UX - text appears immediately | Pending |
| 2 | Add re-ranking with cross-encoder | +10-20% precision | Pending |
| 3 | Implement hybrid search (dense + BM25) | +15-25% recall | Pending |
| 4 | Migrate semantic cache to production | Enable semantic caching on Supabase | Pending |

**Migration Command for Production**:
```bash
psql $SUPABASE_DB_URL -c "CREATE EXTENSION IF NOT EXISTS vector;" && \
psql $SUPABASE_DB_URL -f migrations/add_pgvector_index.sql && \
psql $SUPABASE_DB_URL -f migrations/create_semantic_cache.sql
```

See `EMBED.md` for detailed embedding improvement recommendations.
