"""
Cache dependency injection.
"""

from fastapi import Depends

from app.dependencies.redis_client import get_redis_client
from app.services.cache.rag_cache_service import RagCacheService


def get_rag_cache_service(
    redis=Depends(get_redis_client)
) -> RagCacheService:
    """Provide RagCacheService instance."""
    return RagCacheService(redis)
