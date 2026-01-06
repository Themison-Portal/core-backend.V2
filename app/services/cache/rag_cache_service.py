"""
Redis caching service for RAG pipeline.
Provides caching for embeddings, chunks, and LLM responses.
"""

import hashlib
import json
from typing import List, Optional
from uuid import UUID

from redis.asyncio import Redis


class RagCacheService:
    """
    Centralized caching service for RAG operations.

    Cache Key Patterns:
    - Embeddings:  emb:{sha256(query)[:16]}
    - Chunks:      chunks:{sha256(query+doc_id)[:16]}
    - Responses:   resp:{sha256(query+doc_id+context_hash)[:16]}
    """

    # TTL Configuration (in seconds)
    TTL_EMBEDDING = 86400       # 24 hours - embeddings are deterministic
    TTL_CHUNKS = 3600           # 1 hour - chunks may change on re-ingestion
    TTL_RESPONSE = 1800         # 30 minutes - LLM responses for freshness

    # Key prefixes
    PREFIX_EMBEDDING = "emb"
    PREFIX_CHUNKS = "chunks"
    PREFIX_RESPONSE = "resp"
    PREFIX_DOC_KEYS = "doc_keys"  # Set of keys per document for invalidation

    def __init__(self, redis: Redis):
        self.redis = redis

    # --------------------------
    # Hash utilities
    # --------------------------
    @staticmethod
    def _hash_key(*parts: str) -> str:
        """Generate deterministic hash from parts."""
        combined = ":".join(str(p) for p in parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    @staticmethod
    def _hash_context(chunks: List[dict]) -> str:
        """Hash chunk content for response cache key."""
        content = json.dumps(
            [c.get("page_content", "") for c in chunks],
            sort_keys=True
        )
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    # --------------------------
    # Embedding cache
    # --------------------------
    async def get_embedding(self, query: str) -> Optional[List[float]]:
        """Retrieve cached embedding for query."""
        key = f"{self.PREFIX_EMBEDDING}:{self._hash_key(query)}"
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    async def set_embedding(
        self,
        query: str,
        embedding: List[float]
    ) -> None:
        """Cache embedding for query."""
        key = f"{self.PREFIX_EMBEDDING}:{self._hash_key(query)}"
        await self.redis.set(
            key,
            json.dumps(embedding),
            ex=self.TTL_EMBEDDING
        )

    # --------------------------
    # Chunk retrieval cache
    # --------------------------
    async def get_chunks(
        self,
        query: str,
        document_id: UUID
    ) -> Optional[List[dict]]:
        """Retrieve cached chunks for query+document."""
        key = f"{self.PREFIX_CHUNKS}:{self._hash_key(query, str(document_id))}"
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    async def set_chunks(
        self,
        query: str,
        document_id: UUID,
        chunks: List[dict]
    ) -> None:
        """Cache chunks for query+document."""
        key = f"{self.PREFIX_CHUNKS}:{self._hash_key(query, str(document_id))}"
        await self.redis.set(
            key,
            json.dumps(chunks),
            ex=self.TTL_CHUNKS
        )
        # Track key for invalidation
        await self._track_document_key(document_id, key)

    # --------------------------
    # LLM response cache
    # --------------------------
    async def get_response(
        self,
        query: str,
        document_id: UUID,
        chunks: List[dict]
    ) -> Optional[dict]:
        """Retrieve cached LLM response."""
        context_hash = self._hash_context(chunks)
        key = f"{self.PREFIX_RESPONSE}:{self._hash_key(query, str(document_id), context_hash)}"
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    async def set_response(
        self,
        query: str,
        document_id: UUID,
        chunks: List[dict],
        response: dict
    ) -> None:
        """Cache LLM response."""
        context_hash = self._hash_context(chunks)
        key = f"{self.PREFIX_RESPONSE}:{self._hash_key(query, str(document_id), context_hash)}"
        await self.redis.set(
            key,
            json.dumps(response),
            ex=self.TTL_RESPONSE
        )
        # Track key for invalidation
        await self._track_document_key(document_id, key)

    # --------------------------
    # Cache invalidation
    # --------------------------
    async def _track_document_key(
        self,
        document_id: UUID,
        cache_key: str
    ) -> None:
        """Track cache key for document-based invalidation."""
        set_key = f"{self.PREFIX_DOC_KEYS}:{document_id}"
        await self.redis.sadd(set_key, cache_key)
        # Set expiry on tracking set (longer than max cache TTL)
        await self.redis.expire(set_key, self.TTL_EMBEDDING + 3600)

    async def invalidate_document(self, document_id: UUID) -> int:
        """
        Invalidate all cached data for a document.
        Called during re-ingestion.

        Returns: Number of keys deleted.
        """
        set_key = f"{self.PREFIX_DOC_KEYS}:{document_id}"
        keys = await self.redis.smembers(set_key)

        deleted = 0
        if keys:
            # Decode bytes to strings if needed
            key_list = [k.decode() if isinstance(k, bytes) else k for k in keys]
            deleted = await self.redis.delete(*key_list)
            await self.redis.delete(set_key)

        return deleted
