"""
Reranker service for improving retrieval precision.
Uses cross-encoder models to re-score retrieved chunks.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Protocol

import cohere

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class IReranker(Protocol):
    """Protocol for reranker implementations."""

    async def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: int = 5
    ) -> List[dict]:
        """
        Rerank documents by relevance to query.

        Args:
            query: The search query
            documents: List of document dicts with 'page_content' key
            top_k: Number of top results to return

        Returns:
            Reranked list of documents with added 'rerank_score'
        """
        ...


class CohereReranker(IReranker):
    """
    Cohere reranker using their rerank API.
    Best for production use - fast and accurate.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.cohere_api_key
        self.model = model or settings.reranker_model

        if not self.api_key:
            raise ValueError("Cohere API key is required for CohereReranker")

        self.client = cohere.AsyncClient(api_key=self.api_key)

    async def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: int = 5
    ) -> List[dict]:
        """
        Rerank documents using Cohere's rerank API.
        """
        if not documents:
            return []

        # Extract text content from documents
        texts = [doc.get("page_content", "") for doc in documents]

        try:
            response = await self.client.rerank(
                model=self.model,
                query=query,
                documents=texts,
                top_n=min(top_k, len(documents)),
                return_documents=False
            )

            # Reorder documents based on reranker scores
            reranked = []
            for result in response.results:
                doc = documents[result.index].copy()
                doc["rerank_score"] = result.relevance_score
                doc["original_index"] = result.index
                reranked.append(doc)

            logger.info(
                f"[RERANK] Cohere reranked {len(documents)} docs -> top {len(reranked)}, "
                f"top score: {reranked[0]['rerank_score']:.4f}" if reranked else ""
            )

            return reranked

        except Exception as e:
            logger.error(f"[RERANK] Cohere rerank failed: {e}")
            # Fallback: return original documents without reranking
            return documents[:top_k]


class NoOpReranker(IReranker):
    """
    No-op reranker that returns documents unchanged.
    Used when reranking is disabled or for testing.
    """

    async def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: int = 5
    ) -> List[dict]:
        """Return documents unchanged, just limited to top_k."""
        return documents[:top_k]


def get_reranker() -> IReranker:
    """
    Factory function to get the configured reranker.

    Returns:
        IReranker: The configured reranker instance
    """
    if not settings.reranker_enabled:
        logger.info("[RERANK] Reranking disabled, using NoOpReranker")
        return NoOpReranker()

    provider = settings.reranker_provider.lower()

    if provider == "cohere":
        if not settings.cohere_api_key:
            logger.warning("[RERANK] Cohere API key not set, using NoOpReranker")
            return NoOpReranker()
        return CohereReranker()

    # Add more providers here as needed (jina, bge, etc.)

    logger.warning(f"[RERANK] Unknown provider '{provider}', using NoOpReranker")
    return NoOpReranker()
