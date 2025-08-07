from typing import Any, AsyncGenerator, Dict, List
from uuid import UUID

from langchain_core.documents import Document
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.embeddings import EmbeddingProvider
from app.db.session import async_session
from app.models.chunks import DocumentChunk
from app.services.indexing.utils.chunking import chunk_documents
from app.services.interfaces.query_service import IRetrievalGenerationService

from .retriever import create_retriever, preprocess_query
from .utils.generation import call_llm_stream, generate_response


class RetrievalGenerationService(IRetrievalGenerationService):
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.retriever = create_retriever(embedding_provider, match_count=5, query_chunk_size=500)

            
    async def process_query(self, query: str) -> AsyncGenerator[str, None]:
        """Process a query through the RAG pipeline"""
        # Preprocess the query
        print(f"step 1")
        processed_query = preprocess_query(query)
        
        print(f"processed_query: {processed_query}")
        
        # Retrieve relevant documents with chunking and embedding generation
        print(f"step 2")
        retrieved_docs = await self.retriever(processed_query)
        
        print(f"retrieved_docs: {retrieved_docs}")
        
        # Generate prompt
        print(f"step 3")
        prompt = generate_response(processed_query, retrieved_docs)
        
        print(f"prompt: {prompt}")
        
        # Get streaming response
        print(f"step 4")
        return await call_llm_stream(prompt)

    async def process_query_on_documents(self, query: str, document_ids: List[str]) -> AsyncGenerator[str, None]:
        """Process a query on specific documents by their IDs"""
        processed_query = preprocess_query(query)
        
        documents = await self.retrieve_documents_by_ids(document_ids)
        
        prompt = generate_response(processed_query, documents)
        
        return await call_llm_stream(prompt)
        
    async def retrieve_documents(self, query: str, limit: int = 5) -> List[Dict[Any, Any]]:
        """Just retrieve documents without generation"""
        processed_query = preprocess_query(query)
        return await self.retriever(processed_query, override_match_count=limit)

    async def retrieve_documents_by_ids(self, document_ids: List[str], limit: int = 5) -> List[Dict[Any, Any]]:
        """Retrieve chunks from specific documents by their IDs"""
        chunks = await self.retrieve_chunks_by_document_ids(document_ids, limit)
        
        documents = [chunk.content for chunk in chunks]
        
        return documents

    async def retrieve_chunks_by_document_ids(self, document_ids: List[str], limit: int = None) -> List[DocumentChunk]:
        """Retrieve document chunks by document IDs from the database"""
        async with async_session() as session:
            uuids = []
            for doc_id in document_ids:
                try:
                    uuids.append(UUID(doc_id))
                except ValueError:
                    continue
            
            if not uuids:
                return []
            
            query = select(DocumentChunk).where(DocumentChunk.document_id.in_(uuids))
            
            if limit:
                query = query.limit(limit)
            
            result = await session.execute(query)
            chunks = result.scalars().all()
            
            return chunks 