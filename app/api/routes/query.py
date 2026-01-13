"""
Query routes
"""

import re
import json
import time
import logging
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Response, Query
from urllib.parse import unquote_plus
from app.dependencies.db import get_db
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.openai import embedding_client
from app.api.routes.auth import get_current_user
from app.services.agenticRag.agent import RagAgent

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


def normalize_markdown(text: str) -> str:
    """
    Normalize markdown formatting for proper rendering in ReactMarkdown.

    - Ensures double line breaks before/after headers
    - Ensures proper line breaks for list items
    - Cleans up excessive whitespace
    """
    # Ensure double line breaks before headers (## Header)
    text = re.sub(r'\n(#{1,6}\s+)', r'\n\n\1', text)
    text = re.sub(r'(#{1,6}\s+[^\n]+)\n(?!\n)', r'\1\n\n', text)

    # Ensure single line break after each numbered list item (1. item)
    # and double line break before starting a new list
    text = re.sub(r'(\d+\.\s+[^\n]+)\n(\d+\.)', r'\1\n\2', text)

    # Ensure single line break after each bullet list item (- item)
    text = re.sub(r'(-\s+[^\n]+)\n(-\s+)', r'\1\n\2', text)

    # Ensure double line break after lists end (before regular text)
    text = re.sub(r'(\d+\.\s+[^\n]+)\n([^-\d\n#])', r'\1\n\n\2', text)
    text = re.sub(r'(-\s+[^\n]+)\n([^-\d\n#])', r'\1\n\n\2', text)

    # Clean up multiple consecutive line breaks (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Clean up leading/trailing whitespace
    text = text.strip()

    return text

class QueryRequest(BaseModel):
    message: str
    user_id: str
    limit: Optional[int] = 5
    document_ids: Optional[List[UUID]] = None

@router.post("")
async def process_query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    cache_service: RagCacheService = Depends(get_rag_cache_service),
    semantic_cache_service: SemanticCacheService = Depends(get_semantic_cache_service),
):
    """
    Main RAG endpoint: runs retrieval + generation.

    Cache hierarchy:
    1. Semantic cache (similarity >= 0.90) - for similar queries
    2. Redis response cache (exact match)
    3. Claude API call (slowest)
    """
    total_start = time.perf_counter()

    if not request.document_ids:
        raise HTTPException(
            status_code=400,
            detail="document_ids is required for RAG queries",
        )

    document_id = request.document_ids[0]
    query = request.message

    logger.info(f"[TIMING] ========== QUERY START ==========")
    logger.info(f"[TIMING] Document ID: {document_id}")
    logger.info(f"[TIMING] Query: {query[:100]}{'...' if len(query) > 100 else ''}")

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
        query,
        document_id,
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

# class QueryRequest(BaseModel):
#     """
#     Query request
#     """
#     message: str
#     user_id: str
#     limit: Optional[int] = 5
#     document_ids: Optional[List[str]] = None
    
# @router.post("")
# async def process_query(
#     request: QueryRequest,
#     current_user: dict = Depends(get_current_user)
# ):
#     """
#     Process a query and return response
#     """
#     try:
#         rag_agent_instance = RagAgent()
#         compiled_graph = rag_agent_instance.create_graph(document_ids=request.document_ids)
        
#         result = compiled_graph.invoke(
#             {"messages": [HumanMessage(content=request.message)]},
#             config={"configurable": {"thread_id": request.user_id}}
#         )
                
#         response = "No response generated"
#         tool_calls = []
#         used_chunks = []

#         # Extract response and used chunks from messages
#         for message in result.get('messages', []):
#             message_type = message.__class__.__name__
#             print(f"🔍 Message type: {message_type}")

#             # Check if this is a ToolMessage (result of tool execution)
#             if message_type == 'ToolMessage':
#                 # ToolMessage.content has the generated answer
#                 if hasattr(message, 'content'):
#                     print(f"🔧 Found ToolMessage content: {message.content[:100]}...")
#                     # Normalize markdown for proper rendering
#                     response = normalize_markdown(message.content)

#                 # ToolMessage.artifact has the structured data
#                 if hasattr(message, 'artifact') and message.artifact:
#                     artifact = message.artifact
#                     if isinstance(artifact, dict):
#                         used_chunks = artifact.get('used_chunks', [])
#                         print(f"📊 Found {len(used_chunks)} used chunks in artifact")

#             # Check if this is an AIMessage with tool_calls (tool invocation)
#             elif hasattr(message, 'tool_calls') and message.tool_calls:
#                 tool_calls = message.tool_calls

#             # Check if this is an AI response (final answer after tool)
#             elif hasattr(message, 'content') and message_type == 'AIMessage' and message.content:
#                 # Only use if we don't have a response yet
#                 if response == "No response generated":
#                     response = normalize_markdown(message.content)

#         # Map used chunks to sources for frontend
#         sources = []
#         print(f"\n{'='*80}")
#         print(f"🎯 MAPPING CHUNKS TO SOURCES")
#         print(f"{'='*80}")

#         for chunk in used_chunks:
#             page_numbers = chunk.get('page_numbers', [])
#             filename = chunk.get('filename', 'Unknown')
#             exact_quote = chunk.get('exact_quote', '')  # LLM's exact citation
#             chunk_index = chunk.get('chunk_index', 0)

#             # Get first page number
#             page = page_numbers[0] if page_numbers else 1

#             # Format section name
#             if len(page_numbers) == 1:
#                 section = f"Page {page}"
#             elif len(page_numbers) > 1:
#                 section = f"Pages {'-'.join(map(str, page_numbers))}"
#             else:
#                 section = "Page Unknown"

#             # Create source object
#             source = {
#                 "section": section,
#                 "page": page,
#                 "content": exact_quote[:200] + "..." if len(exact_quote) > 200 else exact_quote,
#                 "exactText": exact_quote,  # Exact quote for PDF highlighting
#                 "chunk_index": chunk_index,
#                 "filename": filename,
#                 "relevance": "high",
#                 "context": f"Referenced from {filename}",
#                 "highlightURL": ""
#             }

#             sources.append(source)
#             print(f"  ✅ Source {len(sources)}: {section} - \"{exact_quote[:60]}...\"")

#         print(f"{'='*80}")
#         print(f"✅ TOTAL SOURCES: {len(sources)}")
#         print(f"{'='*80}\n")

#         final_response = {
#             "response": response,
#             "sources": sources,
#             "tool_calls": tool_calls,
#         }

#         print(f"🚀 FINAL RESPONSE TO FRONTEND:")
#         print(f"   📝 Response length: {len(response)}")
#         print(f"   📊 Sources count: {len(sources)}")
#         print(f"   🔧 Tool calls: {len(tool_calls)}")

#         return final_response
 
#     except Exception as e:
#         import traceback
#         error_traceback = traceback.format_exc()
#         print(f"❌ ERROR IN QUERY ENDPOINT:")
#         print(error_traceback)
#         raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/get-chat-history")
async def get_chat_history(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get chat history
    """
    try:
        return RagAgent().get_chat_history(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
