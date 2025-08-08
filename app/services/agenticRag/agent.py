from typing import List

from app.services.agenticRag.embedding_provider import EmbeddingProvider


class RagAgent: 
    """
    A class that represents a RAG agent.
    """
    
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.tools = []
        
    