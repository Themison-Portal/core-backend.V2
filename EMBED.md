# Embedding Improvement Recommendations

**Date**: January 9, 2026
**Purpose**: Document recommendations for improving embedding-based retrieval quality and performance

---

## Current Implementation

### Embedding Model
| Setting | Value |
|---------|-------|
| Provider | OpenAI |
| Model | `text-embedding-3-small` |
| Dimensions | 1536 |
| File | `app/core/openai.py` |

### Vector Storage
| Setting | Value |
|---------|-------|
| Database | PostgreSQL (Supabase) |
| Extension | pgvector |
| Index Type | HNSW |
| Index Params | m=16, ef_construction=64 |
| Distance Metric | Cosine similarity |
| File | `app/models/chunks_docling.py` |

### Retrieval
| Setting | Value |
|---------|-------|
| Top-k | 15 chunks |
| Min Score | 0.04 |
| Caching | Redis, 24h TTL |
| File | `app/services/doclingRag/rag_retrieval_service.py` |

---

## Recommended Improvements

### 1. Hybrid Search (Dense + Sparse)

**Problem**: Dense embeddings excel at semantic similarity but miss exact keyword matches. Queries like "Section 4.2" or specific medical terms may not retrieve relevant chunks.

**Solution**: Combine dense vector search with BM25 (sparse keyword search).

**How it works**:
```
final_score = (alpha * embedding_score) + ((1 - alpha) * bm25_score)
```
- `alpha = 0.7` - favor semantic similarity by default
- Tunable per query type

**Implementation**:

1. **Add full-text search index** (`app/models/chunks_docling.py`):
```python
from sqlalchemy import Index

__table_args__ = (
    # Existing HNSW index...
    Index(
        'idx_chunks_content_fts',
        'content',
        postgresql_using='gin',
        postgresql_ops={'content': 'gin_trgm_ops'}
    ),
)
```

2. **Update retrieval query** (`app/services/doclingRag/rag_retrieval_service.py`):
```python
sql = text("""
    WITH semantic AS (
        SELECT id, content, chunk_metadata,
               1 - (embedding <=> :v::vector) AS semantic_score
        FROM document_chunks_docling
        WHERE document_id = :doc_id
        ORDER BY embedding <=> :v::vector
        LIMIT :k
    ),
    keyword AS (
        SELECT id,
               ts_rank(to_tsvector('english', content), plainto_tsquery(:query)) AS bm25_score
        FROM document_chunks_docling
        WHERE document_id = :doc_id
          AND to_tsvector('english', content) @@ plainto_tsquery(:query)
    )
    SELECT s.*,
           COALESCE(k.bm25_score, 0) AS bm25_score,
           (0.7 * s.semantic_score + 0.3 * COALESCE(k.bm25_score, 0)) AS hybrid_score
    FROM semantic s
    LEFT JOIN keyword k ON s.id = k.id
    ORDER BY hybrid_score DESC
""")
```

**Impact**: 15-25% improvement in recall for keyword-heavy queries

---

### 2. Re-ranking with Cross-Encoder

**Problem**: Bi-encoders (embedding models) encode query and document separately. They're fast but less accurate than models that see both together.

**Solution**: Add a second-stage cross-encoder re-ranker.

**How it works**:
1. Retrieve top-30 chunks with embeddings (fast, high recall)
2. Re-rank with cross-encoder to top-15 (accurate, high precision)

**Implementation**:

1. **Add dependency** (`requirements.txt`):
```
sentence-transformers>=2.2.0
```

2. **Add re-ranking** (`app/services/doclingRag/rag_retrieval_service.py`):
```python
from sentence_transformers import CrossEncoder

# Initialize once at module level
_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

async def _rerank_chunks(self, query: str, chunks: List[dict], top_k: int = 15) -> List[dict]:
    """Re-rank chunks using cross-encoder for better precision."""
    if not chunks:
        return chunks

    # Prepare pairs for cross-encoder
    pairs = [(query, chunk['content']) for chunk in chunks]

    # Get cross-encoder scores
    scores = _reranker.predict(pairs)

    # Combine with original scores and sort
    for i, chunk in enumerate(chunks):
        chunk['rerank_score'] = float(scores[i])

    # Sort by rerank score and return top-k
    reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
    return reranked[:top_k]
```

3. **Integrate into retrieval pipeline**:
```python
async def retrieve_similar_chunks(...):
    # 1. Get top-30 with embeddings
    raw_chunks = await self._search_similar_chunks_docling(query, doc_id, top_k=30)

    # 2. Re-rank to top-15
    reranked_chunks = await self._rerank_chunks(query, raw_chunks, top_k=15)

    return reranked_chunks
```

**Impact**: 10-20% improvement in precision (MRR, nDCG)

**Latency**: +50-100ms per query (cross-encoder is small and fast)

---

### 3. HNSW Index Optimization

**Problem**: Default HNSW parameters are conservative. Better tuning improves both speed and recall.

**Current Parameters**:
- `m = 16` (connections per node)
- `ef_construction = 64` (build-time search width)

**Recommended Parameters**:
- `m = 32` (more connections = better recall)
- `ef_construction = 128` (better index quality)
- `ef_search = 64` (query-time search width, set via SET)

**Implementation**:

1. **Update model** (`app/models/chunks_docling.py`):
```python
Index(
    'idx_chunks_embedding_hnsw',
    'embedding',
    postgresql_using='hnsw',
    postgresql_with={'m': 32, 'ef_construction': 128},
    postgresql_ops={'embedding': 'vector_cosine_ops'}
),
```

2. **Create migration** (`migrations/optimize_hnsw_index.sql`):
```sql
-- Drop existing index
DROP INDEX IF EXISTS idx_chunks_embedding_hnsw;

-- Create optimized index (takes time for large tables)
CREATE INDEX idx_chunks_embedding_hnsw
ON document_chunks_docling
USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 128);

-- Set search parameter for session
SET hnsw.ef_search = 64;
```

**Impact**: 5-15% faster search, 5-10% better recall

---

### 4. Query Expansion

**Problem**: User queries are often short or ambiguous. A single embedding may not capture all relevant aspects.

**Solution**: Generate multiple query variations and combine results.

**Implementation**:

```python
async def _expand_query(self, query: str) -> List[str]:
    """Generate query variations using LLM."""
    prompt = f"""Generate 2 alternative phrasings of this search query.
    Return only the queries, one per line.

    Original: {query}

    Alternatives:"""

    # Use fast model for expansion
    response = await _anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )

    variations = response.content[0].text.strip().split('\n')
    return [query] + variations[:2]  # Original + 2 variations

async def retrieve_with_expansion(self, query: str, document_id: UUID, top_k: int = 15):
    """Retrieve with query expansion for better recall."""
    queries = await self._expand_query(query)

    all_chunks = []
    seen_ids = set()

    for q in queries:
        chunks = await self._search_similar_chunks_docling(q, document_id, top_k=10)
        for chunk in chunks:
            if chunk['id'] not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(chunk['id'])

    # Re-rank combined results
    return await self._rerank_chunks(query, all_chunks, top_k=top_k)
```

**Impact**: 10-15% improvement in recall for ambiguous queries

**Latency**: +200-500ms (LLM call for expansion)

---

## Upgrade Path: Better Embedding Model

### Option A: OpenAI text-embedding-3-large

```python
embedding_client = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=3072,  # or 1536 for cost/speed balance
    api_key=settings.openai_api_key
)
```

**Pros**: Better quality, same API
**Cons**: 2x cost, requires re-embedding all documents

### Option B: Voyage AI (voyage-3)

```python
from langchain_voyageai import VoyageAIEmbeddings

embedding_client = VoyageAIEmbeddings(
    model="voyage-3",
    voyage_api_key=settings.voyage_api_key
)
```

**Pros**: State-of-the-art retrieval quality
**Cons**: Different provider, requires API key

### Option C: Local Model (for cost reduction)

```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding_client = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cuda'}
)
```

**Pros**: No API costs, privacy
**Cons**: Requires GPU, self-hosting

---

## Priority Implementation Order

| Priority | Improvement | Impact | Effort | Latency Impact |
|----------|-------------|--------|--------|----------------|
| 1 | Re-ranking | High | Low | +50-100ms |
| 2 | Hybrid Search | High | Medium | +10-20ms |
| 3 | HNSW Optimization | Medium | Low | -5-15ms |
| 4 | Query Expansion | Medium | Medium | +200-500ms |

**Recommended**: Start with Re-ranking (quick win), then add Hybrid Search.

---

## Files to Modify

| File | Changes |
|------|---------|
| `app/services/doclingRag/rag_retrieval_service.py` | Hybrid search, re-ranking, query expansion |
| `app/models/chunks_docling.py` | Full-text index, HNSW parameter tuning |
| `requirements.txt` | Add `sentence-transformers>=2.2.0` |
| `migrations/` | Index optimization migration |

---

## Benchmarking

After implementing changes, measure:

1. **Retrieval Quality**
   - Precision@k
   - Recall@k
   - MRR (Mean Reciprocal Rank)
   - nDCG (Normalized Discounted Cumulative Gain)

2. **Performance**
   - Embedding latency
   - Search latency
   - Total retrieval time

3. **End-to-End**
   - Answer relevance (human eval)
   - Source accuracy (correct page/section)
