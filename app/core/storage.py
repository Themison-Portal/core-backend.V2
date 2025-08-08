"""
This module contains the storage provider.
"""

from abc import ABC, abstractmethod
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.documents import Document


class StorageProvider(ABC):
    """
    An abstract class that provides storage for the application.
    """
    @abstractmethod
    async def similarity_search(
        self,
        query_vector: List[float],
        limit: int = 5
    ) -> List[Document]:
        """
        Search for documents that are similar to the query vector.
        """
        pass

class PostgresVectorStore(StorageProvider):
    """
    A provider that uses PostgreSQL's vector similarity search to get documents.
    """
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def similarity_search(
        self,
        query_vector: List[float],
        limit: int = 5
    ) -> List[Document]:
        # Using PostgreSQL's vector similarity search
        # This is a simplified example - you might want to use pgvector
        query = select(Document).join(Document.embedding).order_by(
            Document.embedding.cosine_distance(query_vector)
        ).limit(limit)
        
        result = await self.db.execute(query)
        return result.scalars().all() 