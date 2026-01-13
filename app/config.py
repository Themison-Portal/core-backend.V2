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

    # Semantic cache configuration
    semantic_cache_similarity_threshold: float = 0.90  # Cosine similarity threshold for cache hits

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