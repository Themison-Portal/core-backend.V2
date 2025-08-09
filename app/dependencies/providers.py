"""
This module contains the provider dependencies.
"""
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from app.core.storage import PostgresVectorStore, StorageProvider
from app.dependencies.db import get_db

# Import at top level
from app.main import app_state


def get_storage_provider(
    db: AsyncSession = Depends(get_db)
) -> StorageProvider:
    """Get vector storage provider instance"""
    return PostgresVectorStore(db) 