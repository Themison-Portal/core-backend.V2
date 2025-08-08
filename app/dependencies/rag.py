"""
This module contains the RAG dependencies.
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.embeddings import EmbeddingProvider
from app.core.storage import StorageProvider
from app.services.indexing.document_service import DocumentService
from app.services.interfaces.document_service import IDocumentService
from app.services.retrieval.retrieval_generation_service import (
    RetrievalGenerationService,
)

from .db import get_db
from .providers import get_embedding_provider, get_storage_provider

"""
Dependency to get document service instance with all required dependencies.
"""
async def get_document_service(
    db: AsyncSession = Depends(get_db),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    storage_provider: StorageProvider = Depends(get_storage_provider)
) -> IDocumentService:
    """Get document service instance with all required dependencies"""
    return DocumentService(
        db=db,
        embedding_provider=embedding_provider,
        storage_provider=storage_provider
    )

"""
Dependency to get RAG service instance with all required dependencies.
"""
async def get_rag_service(
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider)
) -> RetrievalGenerationService:
    """Get RAG service instance with all required dependencies"""
    return RetrievalGenerationService(embedding_provider=embedding_provider) 