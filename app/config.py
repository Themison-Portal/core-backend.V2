"""
Configuration for the application
"""
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings
    """

    supabase_url: str
    supabase_service_key: str
    openai_api_key: str
    anthropic_api_key: str  # Required for Claude Opus 4.5
    supabase_db_url: str
    supabase_anon_key: str = ""  # Optional
    supabase_db_password: str = ""  # Optional
    redis_url: str = ""
    frontend_url: str = "http://localhost:3000"  # Optional with default
    upload_api_key: str = ""  # API key for upload endpoint (X-API-KEY header)

    # Semantic cache configuration
    semantic_cache_similarity_threshold: float = 0.90  # Cosine similarity threshold for cache hits

    # Hybrid search configuration (Phase 1)
    hybrid_search_enabled: bool = True
    hybrid_search_rrf_k: int = 60  # RRF constant (typically 60)

    # Retrieval configuration
    retrieval_min_score: float = 0.04  # Minimum cosine similarity for vector-only search
    retrieval_top_k: int = 20  # Number of chunks to retrieve

    # Reranking configuration (Phase 2)
    reranker_enabled: bool = False
    reranker_provider: str = "cohere"  # Options: "cohere", "jina", "bge"
    reranker_model: str = "rerank-english-v3.0"
    reranker_top_k: int = 5  # Return top N after reranking
    cohere_api_key: str = ""

    # Embedding model configuration (Phase 3)
    embedding_model: str = "text-embedding-3-small"  # or "text-embedding-3-large"
    embedding_dimensions: int = 1536  # 1536 for small, 2000 for large (HNSW limit)

    # Contextual retrieval configuration (Phase 4)
    contextual_retrieval_enabled: bool = False
    contextual_context_window: int = 3  # Include N surrounding chunks for context

    class Config:
        """
        Configuration for the application settings
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    """
    Get the application settings with caching.
    
    Returns:
        Settings: The application configuration settings.
    """
    return Settings()