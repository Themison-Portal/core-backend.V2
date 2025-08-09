"""
This module contains the RAG dependencies.
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.storage import StorageProvider
from app.dependencies.db import get_db
from app.dependencies.providers import get_storage_provider
from app.services.indexing.document_service import DocumentService
from app.services.interfaces.document_service import IDocumentService


async def get_document_service(
    db: AsyncSession = Depends(get_db),
    storage_provider: StorageProvider = Depends(get_storage_provider)
) -> IDocumentService:
    """Get document service instance with all required dependencies"""
    return DocumentService(
        db=db,
        storage_provider=storage_provider
    )