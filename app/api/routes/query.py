"""
Query routes
"""

import json
import time
import logging
from uuid import UUID
from fastapi import APIRouter, Depends, Header, HTTPException, Response, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.dependencies.db import get_db
from app.core.openai import embedding_client
from app.config import get_settings
from app.services.doclingRag.rag_generation_service import RagGenerationService
from app.services.doclingRag.rag_retrieval_service import RagRetrievalService
from app.services.highlighting.interfaces.pdf_highlight_service import IPDFHightlightService
from app.services.highlighting.pdf_highlight_service import PDFHighlightService
from app.services.cache.rag_cache_service import RagCacheService
from app.services.cache.semantic_cache_service import SemanticCacheService
from app.dependencies.redis_client import get_redis_client
from app.dependencies.cache import get_rag_cache_service, get_semantic_cache_service

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    document_id: UUID
    document_name: str

@router.post("")
async def process_query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    cache_service: RagCacheService = Depends(get_rag_cache_service),
    semantic_cache_service: SemanticCacheService = Depends(get_semantic_cache_service),
    x_api_key: str = Header(...),
):
    """
    Main RAG endpoint: runs retrieval + generation.

    Cache hierarchy:
    1. Semantic cache (similarity >= 0.90) - for similar queries
    2. Redis response cache (exact match)
    3. Claude API call (slowest)
    """
    total_start = time.perf_counter()

    # Validate API key
    settings = get_settings()
    if not settings.upload_api_key or x_api_key != settings.upload_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    logger.info(f"[TIMING] ========== QUERY START ==========")
    logger.info(f"[TIMING] Document ID: {request.document_id}")
    logger.info(f"[TIMING] Query: {request.query[:100]}{'...' if len(request.query) > 100 else ''}")

    # Service initialization
    init_start = time.perf_counter()
    retrieval_service = RagRetrievalService(
        db=db,
        embedding_client=embedding_client,
        cache_service=cache_service,
    )
    generation_service = RagGenerationService(
        retrieval_service=retrieval_service,
        cache_service=cache_service,
        semantic_cache_service=semantic_cache_service,
    )
    init_time = (time.perf_counter() - init_start) * 1000
    logger.info(f"[TIMING] Service initialization: {init_time:.2f}ms")

    # Generate answer (includes retrieval + LLM)
    response = await generation_service.generate_answer(
        query_text=request.query,
        document_id=request.document_id,
        document_name=request.document_name,
    )

    total_time = (time.perf_counter() - total_start) * 1000

    # Log comprehensive timing and cache performance summary
    timing = response.get("timing", {})
    retrieval = timing.get("retrieval", {})

    # Extract cache statuses
    embedding_hit = timing.get('embedding_cache_hit', retrieval.get('cache_hit', False))
    semantic_hit = timing.get('semantic_cache_hit', False)
    chunk_hit = retrieval.get('chunk_cache_hit', False)
    response_hit = timing.get('response_cache_hit', False)

    # Calculate time saved by caching
    embedding_ms = timing.get('embedding_ms', retrieval.get('embedding_ms', 0))
    semantic_ms = timing.get('semantic_cache_search_ms', 0)
    llm_ms = timing.get('llm_call_ms', 0)

    # Estimated times without cache (typical values)
    TYPICAL_EMBEDDING_TIME = 500  # ms for OpenAI API call
    TYPICAL_LLM_TIME = 15000  # ms for Claude API call

    time_saved = 0
    if embedding_hit:
        time_saved += TYPICAL_EMBEDDING_TIME - embedding_ms
    if semantic_hit:
        time_saved += TYPICAL_LLM_TIME  # Skipped LLM entirely
    if response_hit:
        time_saved += TYPICAL_LLM_TIME  # Skipped LLM entirely

    # Cache status indicators
    def cache_status(hit: bool, name: str) -> str:
        return f"[HIT] {name}" if hit else f"[MISS] {name}"

    logger.info(f"")
    logger.info(f"[PERF] ============ CACHE PERFORMANCE ============")
    logger.info(f"[PERF] {cache_status(embedding_hit, 'Embedding Cache')} - {embedding_ms:.2f}ms")
    logger.info(f"[PERF] {cache_status(semantic_hit, 'Semantic Cache')} - {semantic_ms:.2f}ms" +
                (f" (similarity: {timing.get('semantic_cache_similarity', 0):.4f})" if semantic_hit else ""))
    logger.info(f"[PERF] {cache_status(chunk_hit, 'Chunk Cache')} - {retrieval.get('retrieval_total_ms', 0):.2f}ms")
    logger.info(f"[PERF] {cache_status(response_hit, 'Response Cache')} - Redis exact match")
    logger.info(f"[PERF] =============================================")

    # Summary statistics
    cache_hits = sum([embedding_hit, semantic_hit, chunk_hit, response_hit])
    cache_total = 4
    hit_rate = (cache_hits / cache_total) * 100

    logger.info(f"[PERF] Cache Hit Rate: {cache_hits}/{cache_total} ({hit_rate:.0f}%)")
    if time_saved > 0:
        logger.info(f"[PERF] Estimated Time Saved: ~{time_saved:.0f}ms")
    logger.info(f"")

    # Detailed timing breakdown
    logger.info(f"[TIMING] ============ TIMING BREAKDOWN ============")
    logger.info(f"[TIMING] Embedding:      {embedding_ms:>8.2f}ms {'(cached)' if embedding_hit else '(computed)'}")
    if semantic_ms > 0:
        logger.info(f"[TIMING] Semantic Search:{semantic_ms:>8.2f}ms {'(HIT - skipped LLM!)' if semantic_hit else ''}")
    logger.info(f"[TIMING] Vector Search:  {retrieval.get('db_search_ms', 0):>8.2f}ms")
    logger.info(f"[TIMING] Chunks: {timing.get('original_chunk_count', 0)} -> {timing.get('compressed_chunk_count', 0)} compressed")
    if llm_ms > 0:
        logger.info(f"[TIMING] LLM (Claude):   {llm_ms:>8.2f}ms")
    logger.info(f"[TIMING] ---------------------------------------------")
    logger.info(f"[TIMING] TOTAL:          {total_time:>8.2f}ms")
    logger.info(f"[TIMING] =============================================")

    return response.get("result")


def get_pdf_highlight_service(
    redis = Depends(get_redis_client),
) -> IPDFHightlightService:
    return PDFHighlightService(redis)


@router.get("/highlighted-pdf")
async def get_highlighted_pdf(
    doc: str,
    page: int,
    bboxes: str | None = Query(None, description="JSON string of list of lists: [[x0,y0,x1,y1],...]"),
    pdf_service: IPDFHightlightService = Depends(get_pdf_highlight_service),
):
    print(f"Generating highlighted PDF for doc: {doc}, page: {page}, bboxes: {bboxes}")
    try:
        # decoded = unquote_plus(bboxes)
        parsed_bboxes = json.loads(bboxes)
        if not isinstance(parsed_bboxes, list):
            raise ValueError("The bboxes parameter must be a JSON list.")
    except Exception:
        raise HTTPException(400, "Invalid bboxes format")
    content = await pdf_service.get_highlighted_pdf(doc, page, parsed_bboxes)
    return Response(content=content, media_type="application/pdf")
