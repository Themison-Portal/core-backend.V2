"""
Retrieval generation service
"""

from typing import Any, AsyncGenerator, Dict, List

from app.services.interfaces.query_service import IRetrievalGenerationService
from app.services.retrieval.retriever import create_retriever, preprocess_query
from app.services.retrieval.utils.generation import call_llm_stream, generate_response


# Entry point for query rag pipeline
class RetrievalGenerationService(IRetrievalGenerationService):
    """
    Retrieval generation service
    """

    def __init__(self):
        self.retriever = create_retriever(match_count=5, query_chunk_size=500)

            
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