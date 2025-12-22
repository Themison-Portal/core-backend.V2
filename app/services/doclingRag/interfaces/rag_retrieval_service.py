from abc import ABC, abstractmethod
from typing import List
from uuid import UUID


class IRagRetrievalService(ABC):
    """
    Interface for RAG retrieval services.
    Defines the public retrieval contract only.
    """

    @abstractmethod
    async def retrieve_similar_chunks(
        self,
        query_text: str,
        document_id: UUID,
        top_k: int = 20,
        min_score: float = 0.04,
    ) -> List[str]:
        """
        Retrieve and format relevant chunks for a RAG query.

        Returns:
            List[str]: Formatted context blocks ready for LLM consumption.
        """
        pass
