from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str
    supabase_service_key: str
    openai_api_key: str
    supabase_db_url: str
    embedding_model: str
    supabase_anon_key: str = ""  # Optional
    supabase_db_password: str = ""  # Optional
    frontend_url: str = "http://localhost:3000"  # Optional with default

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()