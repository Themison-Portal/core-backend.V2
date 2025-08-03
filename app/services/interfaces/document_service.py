from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain.schema import Document as LangchainDocument

from app.contracts.document import (DocumentCreate, DocumentResponse,
                                    DocumentUpdate)
from app.models.chunks import DocumentChunk

from .base import IBaseService


class IDocumentService(IBaseService[DocumentCreate, DocumentUpdate, DocumentResponse]):
    async def parse_pdf(self, document_url: str) -> str:
        """Extract text content from PDF file"""
        pass
    
    async def preprocess_content(self, content: str) -> str:
        """Preprocess text content"""
        pass
    
    async def chunk_content(
        self, 
        content: str, 
        metadata: Dict[str, Any] = None,
        chunk_size: int = 1000,
    ) -> List[LangchainDocument]:
        """Split content into chunks"""
        pass
    
    async def generate_embeddings(self, chunks: List[LangchainDocument]) -> List[List[float]]:
        """Generate embeddings for chunks"""
        pass
    
    async def insert_document_with_chunks(
        self,
        title: str, 
        document_id: UUID,
        content: str,
        chunks: List[LangchainDocument],
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
    
    