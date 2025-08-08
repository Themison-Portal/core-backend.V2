"""
This module contains the embeddings provider.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import List

from app.core.openAI import client


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

class SentenceTransformerProvider(EmbeddingProvider):
    """
    A provider that uses the SentenceTransformer library to get embeddings.
    """
    def __init__(self):
        self.client = client
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get an embedding for a text.
        """
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]
    

    async def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Get embeddings for multiple texts with batching.
        """
        if not texts:
            return []
        
        async def process_batch(batch: List[str]):
            return await asyncio.to_thread(
                lambda: self.get_embedding(batch)
            )

        if len(texts) <= batch_size:
            return await process_batch(texts)
        
        # Process in batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        batch_embeddings = await asyncio.gather(*[process_batch(batch) for batch in batches])
        
        # Combine all batches
        all_embeddings = []
        for batch_embeddings_list in batch_embeddings:
            all_embeddings.extend(batch_embeddings_list)
        
        return all_embeddings

# different embedding models comparison in the future hence the factory design 