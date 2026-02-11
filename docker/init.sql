-- Initialize Themison Database
-- This script runs automatically when the PostgreSQL container starts

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create trial_documents table
CREATE TABLE IF NOT EXISTS trial_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_name VARCHAR(255) NOT NULL,
    document_url TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    file_size BIGINT,
    mime_type VARCHAR(100),
    tags JSONB DEFAULT '[]'::jsonb,
    user_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create document_chunks_docling table with vector embedding
CREATE TABLE IF NOT EXISTS document_chunks_docling (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES trial_documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    page_number INTEGER,
    chunk_metadata JSONB,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create chat_sessions table
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    title VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create chat_messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    document_chunk_ids UUID[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create chat_document_links junction table
CREATE TABLE IF NOT EXISTS chat_document_links (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES trial_documents(id) ON DELETE CASCADE,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(session_id, document_id)
);

-- Create HNSW index for fast vector similarity search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON document_chunks_docling
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Create index on document_id for faster filtering
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
ON document_chunks_docling (document_id);

-- Create indexes for chat tables
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_document_links_session ON chat_document_links(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_document_links_document ON chat_document_links(document_id);

-- Grant permissions (for local development)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Themison database initialized successfully!';
    RAISE NOTICE 'pgvector extension enabled';
    RAISE NOTICE 'HNSW index created for vector search';
END $$;
