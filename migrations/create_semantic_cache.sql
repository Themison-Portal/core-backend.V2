-- Migration: Create semantic cache responses table for similarity-based RAG caching
-- This table stores cached RAG responses with query embeddings for semantic similarity lookup

-- Ensure pgvector extension is available
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the semantic cache table
CREATE TABLE IF NOT EXISTS semantic_cache_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Query identification
    query_text TEXT NOT NULL,
    query_embedding Vector(1536) NOT NULL,

    -- Scope: same query on different document = different cache entry
    document_id UUID NOT NULL REFERENCES trial_documents(id) ON DELETE CASCADE,

    -- Cached response (JSON blob matching DoclingRagStructuredResponse)
    response_data JSONB NOT NULL,

    -- Metadata for cache management
    hit_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),

    -- Context fingerprint for invalidation detection (hash of chunk content)
    context_hash VARCHAR(32) NOT NULL
);

-- HNSW index for fast semantic similarity search
-- Parameters: m=16 (connections per node), ef_construction=64 (build quality)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_semantic_cache_embedding_hnsw
ON semantic_cache_responses
USING hnsw (query_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- B-tree index for document filtering (semantic search scoped to document)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_semantic_cache_document_id
ON semantic_cache_responses (document_id);

-- Update statistics for the query planner
ANALYZE semantic_cache_responses;

-- Add comment for documentation
COMMENT ON TABLE semantic_cache_responses IS 'Stores RAG responses with query embeddings for semantic similarity-based caching. Queries with cosine similarity >= 0.90 return cached responses.';
