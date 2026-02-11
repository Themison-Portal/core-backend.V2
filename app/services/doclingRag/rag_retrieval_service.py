import asyncio
import time
import logging
from typing import List, Optional, Dict, TYPE_CHECKING
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from langchain_core.documents import Document

from app.services.doclingRag.interfaces.rag_retrieval_service import IRagRetrievalService
from app.services.utils.threading import run_in_thread  # your helper for async threading
from app.config import get_settings

if TYPE_CHECKING:
    from app.services.cache.rag_cache_service import RagCacheService

logger = logging.getLogger(__name__)
settings = get_settings()


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

    async def get_query_embedding(self, query_text: str) -> tuple[List[float], dict]:
        """Get embedding with caching. Returns (embedding, timing_info).

        Public method to allow callers (e.g., semantic cache) to reuse embeddings.
        """
        timing_info = {"cache_hit": False, "embedding_ms": 0.0}

        # Try cache first
        if self.cache_service:
            cache_start = time.perf_counter()
            cached = await self.cache_service.get_embedding(query_text)
            if cached:
                timing_info["cache_hit"] = True
                timing_info["embedding_ms"] = (time.perf_counter() - cache_start) * 1000
                logger.info(f"[CACHE] Embedding [HIT] - Retrieved from Redis in {timing_info['embedding_ms']:.2f}ms (saved ~500ms OpenAI call)")
                return cached, timing_info

        # Compute embedding
        embed_start = time.perf_counter()
        embedding = await run_in_thread(
            self.embedding_client.embed_query,
            query_text
        )
        timing_info["embedding_ms"] = (time.perf_counter() - embed_start) * 1000
        logger.info(f"[CACHE] Embedding [MISS] - Generated via OpenAI API in {timing_info['embedding_ms']:.2f}ms")

        # Cache for future requests
        if self.cache_service:
            await self.cache_service.set_embedding(query_text, embedding)
            logger.info(f"[CACHE] Embedding stored in Redis (TTL: 24h)")

        return embedding, timing_info

    async def _search_similar_chunks_docling(
        self,
        query_text: str,
        document_id: UUID,
        document_name: str,
        top_k: int = 20,
        precomputed_embedding: Optional[List[float]] = None,
    ) -> tuple[List[dict], dict]:
        """
        Retrieve top-k similar chunks from `document_chunks_docling` using embeddings.
        Returns (chunks, timing_info).

        Args:
            document_name: Name of the document (passed from caller, no DB lookup needed).
            precomputed_embedding: If provided, skip embedding generation (for semantic cache flow).
        """
        timing_info = {}

        # Use precomputed embedding or generate new one
        if precomputed_embedding is not None:
            query_vector = precomputed_embedding
            timing_info["cache_hit"] = True  # Embedding was already computed
            timing_info["embedding_ms"] = 0.0
        else:
            query_vector, embed_timing = await self.get_query_embedding(query_text)
            timing_info.update(embed_timing)

        query_vector = self._embedding_to_pg_vector(query_vector)

        # Query database (no JOIN needed - document_name passed from caller)
        sql = text("""
            SELECT
                pc.content,
                pc.page_number,
                pc.chunk_metadata,
                1 - (pc.embedding <=> (:v)::vector) AS similarity
            FROM document_chunks_docling pc
            WHERE pc.document_id = :pid
            ORDER BY pc.embedding <=> (:v)::vector
            LIMIT :k
        """)

        db_start = time.perf_counter()
        result = await self.db.execute(sql, {"v": query_vector, "k": top_k, "pid": document_id})
        rows = result.fetchall()
        timing_info["db_search_ms"] = (time.perf_counter() - db_start) * 1000
        logger.info(f"[TIMING] Vector search (pgvector HNSW): {timing_info['db_search_ms']:.2f}ms, found {len(rows)} chunks")

        # Format results (use provided document_name)
        docs = [
            {
                "page_content": row.content,
                "score": float(row.similarity),
                "metadata": {
                    "title": document_name,
                    "page": row.page_number,
                    "docling": row.chunk_metadata,
                },
            }
            for row in rows
        ]

        return docs, timing_info

    async def _search_bm25(
        self,
        query_text: str,
        document_id: UUID,
        document_name: str,
        top_k: int = 20
    ) -> List[dict]:
        """
        Full-text BM25 search using PostgreSQL tsvector.
        Returns chunks ranked by text relevance.

        Args:
            document_name: Name of the document (passed from caller, no DB lookup needed).
        """
        sql = text("""
            SELECT
                pc.id,
                pc.content,
                pc.page_number,
                pc.chunk_metadata,
                ts_rank(pc.content_tsv, plainto_tsquery('english', :query)) AS bm25_score
            FROM document_chunks_docling pc
            WHERE pc.document_id = :pid
              AND pc.content_tsv @@ plainto_tsquery('english', :query)
            ORDER BY bm25_score DESC
            LIMIT :k
        """)

        db_start = time.perf_counter()
        result = await self.db.execute(sql, {"query": query_text, "k": top_k, "pid": document_id})
        rows = result.fetchall()
        bm25_time = (time.perf_counter() - db_start) * 1000
        logger.info(f"[TIMING] BM25 search (PostgreSQL GIN): {bm25_time:.2f}ms, found {len(rows)} chunks")

        # Format results with unique IDs for RRF fusion (use provided document_name)
        docs = [
            {
                "id": str(row.id),
                "page_content": row.content,
                "score": float(row.bm25_score),
                "metadata": {
                    "title": document_name,
                    "page": row.page_number,
                    "docling": row.chunk_metadata,
                },
            }
            for row in rows
        ]
        return docs

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[dict],
        bm25_results: List[dict],
        k: int = 60
    ) -> List[dict]:
        """
        Combine vector and BM25 results using Reciprocal Rank Fusion (RRF).
        RRF score = sum(1 / (k + rank)) for each result list.

        Args:
            vector_results: Results from vector similarity search
            bm25_results: Results from BM25 full-text search
            k: RRF constant (typically 60)

        Returns:
            Merged and re-ranked results
        """
        # Build a map of doc_id -> document data
        doc_map: Dict[str, dict] = {}
        rrf_scores: Dict[str, float] = {}

        # Process vector results
        for rank, doc in enumerate(vector_results):
            # Use content hash as ID if id not present
            doc_id = doc.get("id") or hash(doc["page_content"])
            doc_id = str(doc_id)

            if doc_id not in doc_map:
                doc_map[doc_id] = doc.copy()
                doc_map[doc_id]["vector_rank"] = rank + 1
                doc_map[doc_id]["vector_score"] = doc.get("score", 0)

            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # Process BM25 results
        for rank, doc in enumerate(bm25_results):
            doc_id = doc.get("id") or hash(doc["page_content"])
            doc_id = str(doc_id)

            if doc_id not in doc_map:
                doc_map[doc_id] = doc.copy()

            doc_map[doc_id]["bm25_rank"] = rank + 1
            doc_map[doc_id]["bm25_score"] = doc.get("score", 0)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build final results with RRF scores
        fused_results = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id]
            doc["score"] = rrf_scores[doc_id]  # Use RRF score as final score
            doc["rrf_score"] = rrf_scores[doc_id]
            fused_results.append(doc)

        logger.info(f"[HYBRID] RRF fusion: {len(vector_results)} vector + {len(bm25_results)} BM25 -> {len(fused_results)} merged")
        return fused_results

    async def _search_hybrid(
        self,
        query_text: str,
        document_id: UUID,
        document_name: str,
        top_k: int = 20,
        precomputed_embedding: Optional[List[float]] = None,
    ) -> tuple[List[dict], dict]:
        """
        Hybrid search combining vector similarity and BM25 full-text search.
        Runs both searches in parallel and fuses results with RRF.

        Args:
            document_name: Name of the document (passed to sub-searches).
        """
        timing_info = {"hybrid_search": True}

        # Run both searches in parallel
        hybrid_start = time.perf_counter()

        vector_task = self._search_similar_chunks_docling(
            query_text, document_id, document_name, top_k, precomputed_embedding
        )
        bm25_task = self._search_bm25(query_text, document_id, document_name, top_k)

        (vector_results, vector_timing), bm25_results = await asyncio.gather(
            vector_task, bm25_task
        )

        timing_info.update(vector_timing)
        timing_info["hybrid_parallel_ms"] = (time.perf_counter() - hybrid_start) * 1000

        # Fuse results using RRF
        rrf_k = settings.hybrid_search_rrf_k
        fused_results = self._reciprocal_rank_fusion(vector_results, bm25_results, k=rrf_k)

        timing_info["vector_count"] = len(vector_results)
        timing_info["bm25_count"] = len(bm25_results)
        timing_info["fused_count"] = len(fused_results)

        logger.info(
            f"[HYBRID] Search complete: vector={len(vector_results)}, "
            f"bm25={len(bm25_results)}, fused={len(fused_results)} in {timing_info['hybrid_parallel_ms']:.2f}ms"
        )

        return fused_results[:top_k], timing_info

    # --------------------------
    # Public interface
    # --------------------------
    async def retrieve_similar_chunks(
        self,
        query_text: str,
        document_id: UUID,
        document_name: str,
        top_k: int = None,
        min_score: float = None,
        precomputed_embedding: Optional[List[float]] = None
    ) -> tuple[List[dict], dict]:
        """
        Public method to retrieve and format top similar chunks for a query.
        Returns (chunks, timing_info).

        Args:
            document_name: Name of the document (no DB lookup needed).
            precomputed_embedding: If provided, skip embedding generation (for semantic cache flow).
        """
        # Use config defaults if not provided
        if top_k is None:
            top_k = settings.retrieval_top_k
        if min_score is None:
            min_score = settings.retrieval_min_score
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
                logger.info(f"[CACHE] Chunks [HIT] - Retrieved {len(cached_chunks)} chunks from Redis in {timing_info['retrieval_total_ms']:.2f}ms (saved ~500ms pgvector search)")
                return cached_chunks, timing_info

        # Use hybrid search if enabled, otherwise vector-only
        if settings.hybrid_search_enabled:
            raw_chunks, search_timing = await self._search_hybrid(
                query_text, document_id, document_name, top_k, precomputed_embedding
            )
        else:
            raw_chunks, search_timing = await self._search_similar_chunks_docling(
                query_text, document_id, document_name, top_k, precomputed_embedding
            )
        timing_info.update(search_timing)

        # Filter by relevance (only for vector-only search, not hybrid)
        # RRF scores are much smaller (0.01-0.03) than cosine similarity (0.5-1.0)
        if settings.hybrid_search_enabled:
            # For hybrid search, RRF already ranks by relevance - just take top results
            filtered_chunks = raw_chunks
            logger.info(f"[CACHE] Chunks [MISS] - Hybrid search returned {len(raw_chunks)} chunks (RRF-ranked, no min_score filter)")
        else:
            # For vector-only search, filter by cosine similarity threshold
            filtered_chunks = [d for d in raw_chunks if d["score"] >= min_score]
            logger.info(f"[CACHE] Chunks [MISS] - Vector search returned {len(raw_chunks)} chunks, filtered to {len(filtered_chunks)} (min_score={min_score})")

        # Cache results
        if self.cache_service and filtered_chunks:
            await self.cache_service.set_chunks(
                query_text,
                document_id,
                filtered_chunks
            )
            logger.info(f"[CACHE] Chunks stored in Redis (TTL: 1h)")

        timing_info["retrieval_total_ms"] = (time.perf_counter() - retrieval_start) * 1000
        logger.info(f"[TIMING] Retrieval total: {timing_info['retrieval_total_ms']:.2f}ms")

        return filtered_chunks, timing_info
