from typing import Any, AsyncGenerator, Dict, List

from app.core.embeddings import EmbeddingProvider
from app.services.utils.chunking import chunk_text

from .retriever import create_retriever, preprocess_query
from .utils.generation import call_llm_stream, generate_response


class RetrievalGenerationService:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.retriever = create_retriever(embedding_provider)
    
        
    # TODO: abstract this out to a separate service
    async def generate_embeddings(self, chunks: List[LangchainDocument]) -> List[List[float]]:
        """Generate embeddings for chunks"""
        
        texts = [chunk.page_content for chunk in chunks]
        print(texts)
        
        embeddings = await self.embedding_provider.get_embeddings_batch(texts)
        
        return embeddings
            
    async def process_query(self, query: str) -> AsyncGenerator[str, None]:
        """Process a query through the RAG pipeline"""
        # Preprocess the query
        print(f"step 1")
        processed_query = preprocess_query(query)
        
        # Chunk the query
        print(f"step 1.5")
        chunks = chunk_text(processed_query)
        
        # Retrieve relevant documents
        print(f"step 2")
        retrieved_docs = await self.retriever(processed_query)
        
        # Generate prompt
        print(f"step 3")
        prompt = generate_response(processed_query, retrieved_docs)
        
        # Get streaming response
        return await call_llm_stream(prompt)
        
    async def retrieve_documents(self, query: str, limit: int = 5) -> List[Dict[Any, Any]]:
        """Just retrieve documents without generation"""
        processed_query = preprocess_query(query)
        return await self.retriever(processed_query, override_match_count=limit) 