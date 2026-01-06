import time
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from uuid import UUID

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.services.doclingRag.interfaces.rag_generation_service import IRagGenerationService
from app.services.doclingRag.rag_retrieval_service import RagRetrievalService
from app.schemas.rag_docling_schema import DoclingRagStructuredResponse
from app.core.openai import structured_llm

if TYPE_CHECKING:
    from app.services.cache.rag_cache_service import RagCacheService

logger = logging.getLogger(__name__)

# -----------------------------------
# Prompt template
# -----------------------------------
UNIFIED_PROMPT_TEMPLATE = """
You are an expert clinical Document assistant.

⚠️ CRITICAL RULES ⚠️
• Use ONLY the provided context.
• Every fact MUST have an inline citation, e.g., (Document_Title, p. 10, section: Section Title).
• Each context block contains a [BBOX: ...] tag. You must include this exact bbox in your JSON output for the sources you use.

IMPORTANT:
If you use multiple context blocks from the same page,
you MUST return ALL their corresponding BBOX values.
Do NOT merge or discard bboxes.
──────────────────────────────
CONTEXT:
{context}

QUESTION:
{question}

──────────────────────────────
OUTPUT:
INSTRUCTIONS FOR JSON:
1. "response": Markdown answer with inline citations.
2. "sources": An array of objects.
3. "bboxes": This MUST be an array of arrays [[x,y,x,y], [x,y,x,y]]. 
   - Include the BBOX for EVERY context chunk that contributed to the answer.
   - If information comes from 3 chunks on the same page, "bboxes" must contain 3 coordinate arrays.
4. "exactText": A verbatim snippet. If using multiple chunks, join them with "..."
Return ONLY valid JSON:
{{
  "response": "<Markdown answer with inline citations>",
  "sources": [
    {{
      "protocol": "<Title>",
      "page": <page>,
      "section": "<section or null>",
      "exactText": "<verbatim text>",
      "bboxes": [[x0, y0, x1, y1], [x2, y2, x3, y4]],
      "relevance": "high"
    }}
  ]
}}
"""

# -----------------------------------
# Service
# -----------------------------------
class RagGenerationService(IRagGenerationService):
    """
    RAG generation service that combines retrieval and LLM generation.
    """

    def __init__(
        self,
        retrieval_service: RagRetrievalService,
        cache_service: Optional["RagCacheService"] = None
    ):
        self.retrieval_service = retrieval_service
        self.cache_service = cache_service    
    

    def _format_context_docling(self, doc: dict) -> str:
      meta = doc["metadata"]
      # This matches the 'chunk_metadata' structure saved by your Ingestion Service
      dl_meta = meta.get("docling", {}).get("dl_meta", {})
      doc_items = dl_meta.get("doc_items", [])
      
      # 1. Dig deep for the bbox
      bbox = None
      if doc_items:
          prov = doc_items[0].get("prov", [])
          if prov:
              # bbox usually looks like: {"l":..., "t":..., "r":..., "b":...} 
              # or [x0, y0, x1, y1] depending on Docling version
              raw_bbox = prov[0].get("bbox")
              
              # Ensure it's a list for your RagSource Pydantic model
              if isinstance(raw_bbox, dict):
                  bbox = [raw_bbox.get("l"), raw_bbox.get("t"), raw_bbox.get("r"), raw_bbox.get("b")]
              else:
                  bbox = raw_bbox

      # 2. Traditional metadata
      title = meta.get("title", "Unknown")
      page = dl_meta.get("page_no") or meta.get("page") or 0
      headings = dl_meta.get("headings", [])
      section = headings[-1] if headings else "N/A"

      return (
          f"### SOURCE START ###\n"
          f"NAME: {title}\n"
          f"PAGE: {page}\n"
          f"BBOX: {bbox}\n"  # Now this will NOT be None
          f"CONTENT: {doc['page_content']}\n"
          f"### SOURCE END ###"
      )

    async def generate_answer(
        self,
        query_text: str,
        document_id: UUID,
        top_k: int = 15,  # Reduced from 40 to improve LLM response time
        min_score: float = 0.04
    ) -> dict:
        """
        Generate answer with timing information.
        Returns dict with 'result' (DoclingRagStructuredResponse) and 'timing' info.
        """
        generation_start = time.perf_counter()
        timing_info = {"response_cache_hit": False}

        # 1. Retrieve chunks (already cached in retrieval service)
        retrieval_start = time.perf_counter()
        filtered_chunks, retrieval_timing = await self.retrieval_service.retrieve_similar_chunks(
            query_text=query_text,
            document_id=document_id,
            top_k=top_k,
            min_score=min_score
        )
        timing_info["retrieval"] = retrieval_timing

        if not filtered_chunks:
            timing_info["generation_total_ms"] = (time.perf_counter() - generation_start) * 1000
            return {
                "result": DoclingRagStructuredResponse(
                    response="The provided documents do not contain this information.",
                    sources=[]
                ),
                "timing": timing_info
            }

        # 2. Check response cache
        if self.cache_service:
            cache_start = time.perf_counter()
            cached_response = await self.cache_service.get_response(
                query_text,
                document_id,
                filtered_chunks
            )
            if cached_response:
                timing_info["response_cache_hit"] = True
                timing_info["generation_total_ms"] = (time.perf_counter() - generation_start) * 1000
                logger.info(f"[TIMING] Response cache HIT: {timing_info['generation_total_ms']:.2f}ms")
                return {
                    "result": DoclingRagStructuredResponse(**cached_response),
                    "timing": timing_info
                }

        # 3. Format context with metadata tags
        format_start = time.perf_counter()
        formatted_context = "\n\n".join([self._format_context_docling(d) for d in filtered_chunks])
        timing_info["context_format_ms"] = (time.perf_counter() - format_start) * 1000

        # 4. Define the Chain using singleton LLM (avoids per-request instantiation)
        chat_prompt = ChatPromptTemplate.from_template(UNIFIED_PROMPT_TEMPLATE)
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | chat_prompt
            | structured_llm
        )

        # Log estimated token count (rough: 1 token ≈ 4 chars)
        context_chars = len(formatted_context)
        estimated_tokens = context_chars // 4
        logger.info(f"[TIMING] Context size: {context_chars} chars (~{estimated_tokens} tokens)")

        # 5. Execute LLM
        llm_start = time.perf_counter()
        result = await chain.ainvoke({
            "context": formatted_context,
            "question": query_text
        })
        timing_info["llm_call_ms"] = (time.perf_counter() - llm_start) * 1000
        logger.info(f"[TIMING] LLM generation (GPT-4o-mini): {timing_info['llm_call_ms']:.2f}ms")

        # 6. Cache response
        if self.cache_service:
            await self.cache_service.set_response(
                query_text,
                document_id,
                filtered_chunks,
                result.model_dump()
            )

        timing_info["generation_total_ms"] = (time.perf_counter() - generation_start) * 1000
        logger.info(f"[TIMING] Generation total: {timing_info['generation_total_ms']:.2f}ms")

        return {
            "result": result,
            "timing": timing_info
        }