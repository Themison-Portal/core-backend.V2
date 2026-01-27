from typing import List
from uuid import UUID
from app.contracts.base import BaseContract

class IRagIngestionService:
    async def ingest_pdf(
        self,
        document_url: str,
        document_id: UUID,
        chunk_size: int = 750,
    ) -> None:
        """Complete ingestion pipeline for a document"""
        pass
