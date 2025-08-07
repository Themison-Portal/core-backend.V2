from typing import Any, AsyncGenerator, Dict, List

from langchain_core.documents import Document

from app.core.embeddings import EmbeddingProvider
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
        processed_query = preprocess_query(query)
        
        
        retrieved_docs = await self.retriever(processed_query)
        
        prompt = generate_response(processed_query, retrieved_docs)
        
        
        return await call_llm_stream(prompt)
        
    async def retrieve_documents(self, query: str, limit: int = 5) -> List[Dict[Any, Any]]:
        """Just retrieve documents without generation"""
        processed_query = preprocess_query(query)
        return await self.retriever(processed_query, override_match_count=limit) 