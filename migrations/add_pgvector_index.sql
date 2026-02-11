-- Migration: Add HNSW index for pgvector similarity search
-- This significantly improves vector search performance (10-100x faster)
-- Run this migration during a low-traffic period as index creation can take time

-- Create HNSW index for cosine similarity search on embeddings
-- CONCURRENTLY allows the table to remain accessible during index creation
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_embedding_hnsw
ON document_chunks_docling
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Create B-tree index on document_id for faster filtering and JOINs
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_document_id
ON document_chunks_docling (document_id);

-- Update statistics for the query planner
ANALYZE document_chunks_docling;
