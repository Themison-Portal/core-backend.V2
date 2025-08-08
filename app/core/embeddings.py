"""
This module contains the embeddings provider.
"""
from abc import ABC, abstractmethod
from typing import List

from app.core.openAI import embedding_client


class EmbeddingProvider(ABC):
    """
    An abstract class that provides embeddings for text.
    """
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Get an embedding for a text.
        """
        pass
    
    @abstractmethod
    async def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Get embeddings for multiple texts with batching.
        """
        pass

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    A provider that uses OpenAI embeddings via LangChain.
    """
    def __init__(self):
        self.client = embedding_client
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get an embedding for a text using LangChain.
        """
        # LangChain embeddings return a list of floats directly
        return self.client.embed_query(text)
    
    async def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Get embeddings for multiple texts with batching using LangChain.
        """
        if not texts:
            return []
        
        # LangChain handles batching automatically
        embeddings = await self.client.aembed_documents(texts)
        return embeddings

# different embedding models comparison in the future hence the factory design 