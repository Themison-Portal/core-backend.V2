from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.dependencies.db import get_db
from app.dependencies.cache import get_rag_cache_service, get_semantic_cache_service
from app.services.doclingRag.rag_ingestion_service import RagIngestionService
from app.services.cache.rag_cache_service import RagCacheService
from app.services.cache.semantic_cache_service import SemanticCacheService


async def get_rag_ingestion_service(
    db: AsyncSession = Depends(get_db),
    cache_service: RagCacheService = Depends(get_rag_cache_service),
    semantic_cache_service: SemanticCacheService = Depends(get_semantic_cache_service)
):
    return RagIngestionService(
        db,
        cache_service=cache_service,
        semantic_cache_service=semantic_cache_service
    )