import re
import time
import json
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from uuid import UUID
from openai import AsyncOpenAI

from app.services.doclingRag.interfaces.rag_generation_service import IRagGenerationService
from app.services.doclingRag.rag_retrieval_service import RagRetrievalService
from app.schemas.rag_docling_schema import DoclingRagStructuredResponse, RagSource
from app.config import get_settings

if TYPE_CHECKING:
    from app.services.cache.rag_cache_service import RagCacheService

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize async OpenAI client for predicted outputs
_openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

# -----------------------------------
# Prompt template - Optimized for prompt caching
# Static instructions FIRST (cacheable), dynamic content LAST
# -----------------------------------
SYSTEM_PROMPT = """You are an expert clinical Document assistant. You MUST respond with valid JSON only.

RULES:
• Use ONLY the provided context
• Every fact MUST have an inline citation: (Document_Title, p. X)
• Include bbox coordinates from context in your sources
• If multiple chunks from same page, include ALL their bboxes

RESPOND WITH THIS EXACT JSON STRUCTURE (no other text):
{"response": "markdown answer with citations", "sources": [{"name": "doc title", "page": 1, "section": "section or null", "exactText": "verbatim quote", "bboxes": [[x0,y0,x1,y1]], "relevance": "high"}]}"""

# Prediction template for OpenAI predicted outputs feature
# This helps OpenAI generate structured output faster
PREDICTION_TEMPLATE = '{"response": "'

# -----------------------------------
# Service
# -----------------------------------
class RagGenerationService(IRagGenerationService):
    """
    RAG generation service that combines retrieval and LLM generation.
    Optimized with: prompt caching, chunk compression, predicted outputs.
    """

    def __init__(
        self,
        retrieval_service: RagRetrievalService,
        cache_service: Optional["RagCacheService"] = None
    ):
        self.retrieval_service = retrieval_service
        self.cache_service = cache_service

    def _extract_chunk_metadata(self, doc: dict) -> dict:
        """Extract metadata from a chunk for compression and formatting."""
        meta = doc.get("metadata", {})
        dl_meta = meta.get("docling", {}).get("dl_meta", {})
        doc_items = dl_meta.get("doc_items", [])

        bbox = None
        if doc_items:
            prov = doc_items[0].get("prov", [])
            if prov:
                raw_bbox = prov[0].get("bbox")
                if isinstance(raw_bbox, dict):
                    bbox = [raw_bbox.get("l"), raw_bbox.get("t"), raw_bbox.get("r"), raw_bbox.get("b")]
                else:
                    bbox = raw_bbox

        title = meta.get("title", "Unknown")
        page = dl_meta.get("page_no") or meta.get("page") or 0
        headings = dl_meta.get("headings", [])
        section = headings[-1] if headings else None

        return {
            "title": title,
            "page": page,
            "section": section,
            "bbox": bbox,
            "content": doc.get("page_content", ""),
        }

    def _compress_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Compress chunks by merging those from the same page.
        Preserves all bboxes and combines content.
        """
        if not chunks:
            return []

        # Group chunks by (title, page)
        page_groups: Dict[tuple, List[dict]] = {}
        for chunk in chunks:
            meta = self._extract_chunk_metadata(chunk)
            key = (meta["title"], meta["page"])
            if key not in page_groups:
                page_groups[key] = []
            page_groups[key].append(meta)

        # Merge chunks from same page
        compressed = []
        for (title, page), group in page_groups.items():
            if len(group) == 1:
                # Single chunk, no compression needed
                compressed.append(group[0])
            else:
                # Merge multiple chunks from same page
                all_bboxes = [m["bbox"] for m in group if m["bbox"]]
                all_content = "\n...\n".join(m["content"] for m in group)
                # Use first section found, or None
                section = next((m["section"] for m in group if m["section"]), None)

                compressed.append({
                    "title": title,
                    "page": page,
                    "section": section,
                    "bboxes": all_bboxes,  # List of bboxes for merged chunk
                    "content": all_content[:2000],  # Limit merged content size
                    "merged_count": len(group),
                })

        logger.info(f"[COMPRESSION] {len(chunks)} chunks → {len(compressed)} compressed ({len(chunks) - len(compressed)} merged)")
        return compressed

    def _format_context_compact(self, chunk_meta: dict) -> str:
        """
        Compact context format for reduced token usage.
        ~40 chars overhead vs ~80 chars in original format.
        """
        title = chunk_meta.get("title", "Unknown")
        page = chunk_meta.get("page", 0)
        content = chunk_meta.get("content", "")

        # Handle both single bbox and multiple bboxes (from compression)
        if "bboxes" in chunk_meta:
            bbox_str = str(chunk_meta["bboxes"])
        else:
            bbox_str = str(chunk_meta.get("bbox"))

        return f"[{title}|p{page}|bbox:{bbox_str}]\n{content}"

    def _format_context_docling(self, doc: dict) -> str:
        """Legacy format - kept for compatibility."""
        meta = self._extract_chunk_metadata(doc)
        return self._format_context_compact(meta)

    async def generate_answer(
        self,
        query_text: str,
        document_id: UUID,
        top_k: int = 15,
        min_score: float = 0.04
    ) -> dict:
        """
        Generate answer with timing information.
        Optimized with: prompt caching, chunk compression, predicted outputs.
        Returns dict with 'result' (DoclingRagStructuredResponse) and 'timing' info.
        """
        generation_start = time.perf_counter()
        timing_info = {
            "response_cache_hit": False,
            "chunks_compressed": False,
        }

        # 1. Retrieve chunks
        filtered_chunks, retrieval_timing = await self.retrieval_service.retrieve_similar_chunks(
            query_text=query_text,
            document_id=document_id,
            top_k=top_k,
            min_score=min_score
        )
        timing_info["retrieval"] = retrieval_timing
        timing_info["original_chunk_count"] = len(filtered_chunks)

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

        # 3. Compress chunks (merge same-page chunks)
        compression_start = time.perf_counter()
        compressed_chunks = self._compress_chunks(filtered_chunks)
        timing_info["compression_ms"] = (time.perf_counter() - compression_start) * 1000
        timing_info["compressed_chunk_count"] = len(compressed_chunks)
        timing_info["chunks_compressed"] = len(compressed_chunks) < len(filtered_chunks)

        # 4. Format context with compact format
        format_start = time.perf_counter()
        formatted_context = "\n\n".join([
            self._format_context_compact(chunk) for chunk in compressed_chunks
        ])
        timing_info["context_format_ms"] = (time.perf_counter() - format_start) * 1000

        # Log token estimates
        context_chars = len(formatted_context)
        estimated_tokens = context_chars // 4
        logger.info(f"[TIMING] Context: {context_chars} chars (~{estimated_tokens} tokens), {len(compressed_chunks)} chunks")

        # 5. Call OpenAI with predicted outputs for faster structured generation
        llm_start = time.perf_counter()

        user_message = f"CONTEXT:\n{formatted_context}\n\nQUESTION: {query_text}"

        try:
            response = await _openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                # Use JSON mode for reliable structured output
                response_format={"type": "json_object"},
            )

            timing_info["llm_call_ms"] = (time.perf_counter() - llm_start) * 1000
            logger.info(f"[TIMING] LLM (GPT-4o-mini + JSON mode): {timing_info['llm_call_ms']:.2f}ms")

            # Parse response
            raw_content = response.choices[0].message.content
            logger.debug(f"[DEBUG] Raw LLM response: {raw_content[:500]}...")

            # Try to extract JSON from response (handle cases where model adds extra text)
            try:
                parsed = json.loads(raw_content)
            except json.JSONDecodeError:
                # Try to find JSON object in response
                json_match = re.search(r'\{[\s\S]*\}', raw_content)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON from response: {raw_content[:200]}")

            # Convert to Pydantic model
            sources = []
            for s in parsed.get("sources", []):
                # Handle bboxes - ensure it's a list of lists
                bboxes = s.get("bboxes", [])
                if bboxes and not isinstance(bboxes[0], list):
                    bboxes = [bboxes]  # Wrap single bbox in list

                sources.append(RagSource(
                    name=s.get("name", s.get("protocol", "Unknown")),
                    page=s.get("page", 0),
                    section=s.get("section"),
                    exactText=s.get("exactText", ""),
                    bboxes=bboxes,
                    relevance=s.get("relevance", "high"),
                ))

            result = DoclingRagStructuredResponse(
                response=parsed.get("response", ""),
                sources=sources,
            )

        except Exception as e:
            logger.error(f"[ERROR] OpenAI call failed: {e}")
            # Fallback: return error response
            timing_info["llm_call_ms"] = (time.perf_counter() - llm_start) * 1000
            timing_info["error"] = str(e)
            return {
                "result": DoclingRagStructuredResponse(
                    response=f"Error generating response: {str(e)}",
                    sources=[]
                ),
                "timing": timing_info
            }

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