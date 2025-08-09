"""
Document service interface
"""
from typing import Any, Dict, List
from uuid import UUID

from langchain_core.documents import Document

from app.contracts.document import DocumentCreate, DocumentResponse, DocumentUpdate
from app.services.interfaces.base import IBaseService


class IDocumentService(IBaseService[DocumentCreate, DocumentUpdate, DocumentResponse]):
    """
    Document service interface
    """

    async def parse_pdf(self, document_url: str) -> str:
        """Extract text content from PDF file"""
        pass
    
    async def insert_document_with_chunks(
        self,
        title: str, 
        document_id: UUID,
        content: str,
        chunks: List[Document],
        embeddings: List[List[float]],
        metadata: Dict[str, Any] = None,
        user_id: UUID = None
    ) -> DocumentResponse:
        """Process existing document and add chunks with embeddings"""
        pass
    
    async def process_pdf_complete(
        self,
        document_url: str,
        document_id: UUID,
        user_id: UUID = None,
        chunk_size: int = 1000,
    ) -> DocumentResponse:
        """Complete PDF processing pipeline for existing document"""
        pass
    
    async def ensure_tables_exist(self):
        """Create tables if they don't exist"""
        pass
    
    