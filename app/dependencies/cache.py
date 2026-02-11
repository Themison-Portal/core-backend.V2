"""
Cache dependency injection.
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.dependencies.db import get_db
from app.dependencies.redis_client import get_redis_client
from app.services.cache.rag_cache_service import RagCacheService
from app.services.cache.semantic_cache_service import SemanticCacheService


def get_rag_cache_service(
    redis=Depends(get_redis_client)
) -> RagCacheService:
    """Provide RagCacheService instance."""
    return RagCacheService(redis)


def get_semantic_cache_service(
    db: AsyncSession = Depends(get_db)
) -> SemanticCacheService:
    """Provide SemanticCacheService instance with configured threshold."""
    settings = get_settings()
    return SemanticCacheService(
        db=db,
        similarity_threshold=settings.semantic_cache_similarity_threshold
    )
