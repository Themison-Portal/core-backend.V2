import time
import logging
from typing import List, Optional, TYPE_CHECKING
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from langchain_core.documents import Document

from app.services.doclingRag.interfaces.rag_retrieval_service import IRagRetrievalService
from app.services.utils.threading import run_in_thread  # your helper for async threading

if TYPE_CHECKING:
    from app.services.cache.rag_cache_service import RagCacheService

logger = logging.getLogger(__name__)


class RagRetrievalService(IRagRetrievalService):
    """
    Service for retrieving similar Docling chunks from the database.
    """

    def __init__(
        self,
        db: AsyncSession,
        embedding_client,
        cache_service: Optional["RagCacheService"] = None
    ):
        self.db = db
        self.embedding_client = embedding_client
        self.cache_service = cache_service        

    # --------------------------
    # Private helpers
    # --------------------------
    def _embedding_to_pg_vector(self, emb: List[float]) -> str:
        """
        Convert Python list of floats to PostgreSQL vector format.
        """
        return "[" + ",".join(str(x) for x in emb) + "]"

    async def _get_query_embedding(self, query_text: str) -> tuple[List[float], dict]:
        """Get embedding with caching. Returns (embedding, timing_info)."""
        timing_info = {"cache_hit": False, "embedding_ms": 0.0}

        # Try cache first
        if self.cache_service:
            cache_start = time.perf_counter()
            cached = await self.cache_service.get_embedding(query_text)
            if cached:
                timing_info["cache_hit"] = True
                timing_info["embedding_ms"] = (time.perf_counter() - cache_start) * 1000
                logger.info(f"[TIMING] Embedding cache HIT: {timing_info['embedding_ms']:.2f}ms")
                return cached, timing_info

        # Compute embedding
        embed_start = time.perf_counter()
        embedding = await run_in_thread(
            self.embedding_client.embed_query,
            query_text
        )
        timing_info["embedding_ms"] = (time.perf_counter() - embed_start) * 1000
        logger.info(f"[TIMING] Embedding generation (OpenAI API): {timing_info['embedding_ms']:.2f}ms")

        # Cache for future requests
        if self.cache_service:
            await self.cache_service.set_embedding(query_text, embedding)

        return embedding, timing_info

    async def _search_similar_chunks_docling(
        self,
        query_text: str,
        document_id: UUID,
        top_k: int = 20,
    ) -> tuple[List[dict], dict]:
        """
        Retrieve top-k similar chunks from `document_chunks_docling` using embeddings.
        Returns (chunks, timing_info).
        """
        timing_info = {}

        # Get query embedding (with caching)
        query_vector, embed_timing = await self._get_query_embedding(query_text)
        timing_info.update(embed_timing)
        query_vector = self._embedding_to_pg_vector(query_vector)

        # Query database
        sql = text("""
            SELECT
                pc.content,
                pc.page_number,
                pc.chunk_metadata,
                p.document_name,
                1 - (pc.embedding <=> (:v)::vector) AS similarity
            FROM document_chunks_docling pc
            JOIN trial_documents p ON pc.document_id = p.id
            WHERE pc.document_id = :pid
            ORDER BY pc.embedding <=> (:v)::vector
            LIMIT :k
        """)

        db_start = time.perf_counter()
        async with self.db.begin():
            result = await self.db.execute(sql, {"v": query_vector, "k": top_k, "pid": document_id})
            rows = result.fetchall()
        timing_info["db_search_ms"] = (time.perf_counter() - db_start) * 1000
        logger.info(f"[TIMING] Vector search (pgvector HNSW): {timing_info['db_search_ms']:.2f}ms, found {len(rows)} chunks")

        # Format results
        docs = [
            {
                "page_content": row.content,
                "score": float(row.similarity),
                "metadata": {
                    "title": row.document_name,
                    "page": row.page_number,
                    "docling": row.chunk_metadata,
                },
            }
            for row in rows
        ]

        return docs, timing_info    

    # --------------------------
    # Public interface
    # --------------------------
    async def retrieve_similar_chunks(
        self,
        query_text: str,
        document_id: UUID,
        top_k: int = 20,
        min_score: float = 0.04
    ) -> tuple[List[dict], dict]:
        """
        Public method to retrieve and format top similar chunks for a query.
        Returns (chunks, timing_info).
        """
        retrieval_start = time.perf_counter()
        timing_info = {"chunk_cache_hit": False}

        # Try chunk cache first
        if self.cache_service:
            cache_start = time.perf_counter()
            cached_chunks = await self.cache_service.get_chunks(
                query_text,
                document_id
            )
            if cached_chunks:
                timing_info["chunk_cache_hit"] = True
                timing_info["retrieval_total_ms"] = (time.perf_counter() - retrieval_start) * 1000
                logger.info(f"[TIMING] Chunk cache HIT: {timing_info['retrieval_total_ms']:.2f}ms, {len(cached_chunks)} chunks")
                return cached_chunks, timing_info

        raw_chunks, search_timing = await self._search_similar_chunks_docling(query_text, document_id, top_k)
        timing_info.update(search_timing)

        # Filter by relevance
        filtered_chunks = [d for d in raw_chunks if d["score"] >= min_score]
        logger.info(f"[TIMING] Filtered to {len(filtered_chunks)} chunks (min_score={min_score})")

        # Cache results
        if self.cache_service and filtered_chunks:
            await self.cache_service.set_chunks(
                query_text,
                document_id,
                filtered_chunks
            )

        timing_info["retrieval_total_ms"] = (time.perf_counter() - retrieval_start) * 1000
        logger.info(f"[TIMING] Retrieval total: {timing_info['retrieval_total_ms']:.2f}ms")

        return filtered_chunks, timing_info
