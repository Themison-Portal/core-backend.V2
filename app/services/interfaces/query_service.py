from typing import Any, AsyncGenerator, Dict, List

from langchain_core.documents import Document

from app.contracts.document import DocumentResponse
from app.contracts.query import QueryCreate, QueryResponse, QueryUpdate

from .base import IBaseService


class IRetrievalGenerationService(IBaseService[QueryCreate, QueryUpdate, QueryResponse]):
    async def process_query(self, query: str) -> AsyncGenerator[str, None]:
        """Process a query through the RAG pipeline with streaming response"""
        pass
    
    async def process_query_on_documents(self, query: str, document_ids: List[str]) -> AsyncGenerator[str, None]:
        """Process a query on specific documents by their IDs with streaming response"""
        pass
    
    async def retrieve_documents(self, query: str, limit: int = 5) -> List[Dict[Any, Any]]:
        """Retrieve documents without generation"""
        pass
    
    async def retrieve_documents_by_ids(self, document_ids: List[str], limit: int = 5) -> List[Dict[Any, Any]]:
        """Retrieve chunks from specific documents by their IDs"""
        pass
    
    async def chunk_content(
        self, 
        content: str, 
        metadata: Dict[str, Any] = None,
        chunk_size: int = 1000,
    ) -> List[Document]:
        """Split content into chunks"""
        pass
    
    async def generate_embeddings(self, chunks: List[Document]) -> List[List[float]]:
        """Generate embeddings for chunks"""
        pass