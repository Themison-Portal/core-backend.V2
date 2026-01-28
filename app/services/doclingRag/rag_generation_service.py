import re
import time
import json
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from uuid import UUID
from anthropic import AsyncAnthropic

from app.services.doclingRag.interfaces.rag_generation_service import IRagGenerationService
from app.services.doclingRag.rag_retrieval_service import RagRetrievalService
from app.schemas.rag_docling_schema import DoclingRagStructuredResponse, RagSource
from app.config import get_settings

if TYPE_CHECKING:
    from app.services.cache.rag_cache_service import RagCacheService
    from app.services.cache.semantic_cache_service import SemanticCacheService
    from app.services.reranking.reranker_service import IReranker

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize async Anthropic client for Claude Opus 4.5
_anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

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
        cache_service: Optional["RagCacheService"] = None,
        semantic_cache_service: Optional["SemanticCacheService"] = None,
        reranker: Optional["IReranker"] = None
    ):
        self.retrieval_service = retrieval_service
        self.cache_service = cache_service
        self.semantic_cache_service = semantic_cache_service
        # Initialize reranker if not provided and enabled in settings
        if reranker is not None:
            self.reranker = reranker
        elif settings.reranker_enabled:
            from app.services.reranking.reranker_service import get_reranker
            self.reranker = get_reranker()
        else:
            self.reranker = None

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

    def _repair_json(self, json_str: str) -> str:
        """
        Attempt to repair common JSON formatting issues from LLM responses.
        """
        repaired = json_str

        # Fix unescaped newlines in strings
        repaired = re.sub(r'(?<!\\)\n(?=(?:[^"]*"[^"]*")*[^"]*"[^"]*$)', '\\n', repaired)

        # Fix unescaped quotes inside strings (tricky - be conservative)
        # Look for patterns like ..."text with "quotes" inside"...

        # Fix trailing commas before closing brackets
        repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)

        # Fix missing commas between array elements or object properties
        # Pattern: }" or ]" followed by a new property/element
        repaired = re.sub(r'(\})\s*(")', r'\1,\2', repaired)
        repaired = re.sub(r'(\])\s*(")', r'\1,\2', repaired)

        # Fix missing commas after string values
        repaired = re.sub(r'(")\s+(")', r'\1,\2', repaired)

        # Remove any control characters except \n, \r, \t
        repaired = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', repaired)

        return repaired

    def _parse_llm_json(self, raw_content: str) -> dict:
        """
        Parse JSON from LLM response with multiple fallback strategies.
        Always returns a valid dict, never raises.
        """
        # Strategy 1: Direct parse
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError as e:
            logger.warning(f"[JSON] Direct parse failed: {e}")

        # Strategy 2: Extract JSON with regex
        json_match = re.search(r'\{[\s\S]*\}', raw_content)
        if json_match:
            json_str = json_match.group()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"[JSON] Regex extract failed: {e}")

            # Strategy 3: Repair and parse
            try:
                repaired = self._repair_json(json_str)
                result = json.loads(repaired)
                logger.info("[JSON] Repair successful")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"[JSON] Repair failed: {e}")

        # Strategy 4: Try to extract just the response field
        response_match = re.search(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,}]', raw_content)
        if response_match:
            logger.info("[JSON] Extracted response field only")
            return {
                "response": response_match.group(1).replace('\\"', '"').replace('\\n', '\n'),
                "sources": []
            }

        # Strategy 5: Return raw content as response (last resort)
        logger.warning("[JSON] All parsing strategies failed, returning raw content")
        # Clean up the content - remove JSON artifacts
        clean_content = raw_content
        clean_content = re.sub(r'^\s*\{?\s*"response"\s*:\s*"?', '', clean_content)
        clean_content = re.sub(r'"?\s*,?\s*"sources"\s*:.*$', '', clean_content, flags=re.DOTALL)
        clean_content = clean_content.strip().strip('"').strip()

        return {
            "response": clean_content[:3000] if clean_content else "Unable to parse response from AI.",
            "sources": []
        }

    async def generate_answer(
        self,
        query_text: str,
        document_id: UUID,
        document_name: str,
        top_k: int = 15,
        min_score: float = 0.04
    ) -> dict:
        """
        Generate answer with timing information.
        Optimized with: prompt caching, chunk compression, predicted outputs, semantic caching.

        Cache hierarchy:
        1. Semantic cache (similarity >= 0.90, fastest for similar queries)
        2. Redis response cache (exact match)
        3. Claude API call (slowest)

        Args:
            document_name: Name of the document (passed to retrieval, no DB lookup needed).

        Returns dict with 'result' (DoclingRagStructuredResponse) and 'timing' info.
        """
        generation_start = time.perf_counter()
        timing_info = {
            "response_cache_hit": False,
            "semantic_cache_hit": False,
            "chunks_compressed": False,
        }

        # 1. Get query embedding first (needed for semantic cache)
        query_embedding, embed_timing = await self.retrieval_service.get_query_embedding(query_text)
        timing_info["embedding_ms"] = embed_timing.get("embedding_ms", 0)
        timing_info["embedding_cache_hit"] = embed_timing.get("cache_hit", False)

        # 2. Check semantic cache FIRST (before retrieval)
        if self.semantic_cache_service:
            semantic_start = time.perf_counter()
            cached = await self.semantic_cache_service.get_similar_response(
                query_embedding=query_embedding,
                document_id=document_id
            )
            timing_info["semantic_cache_search_ms"] = (time.perf_counter() - semantic_start) * 1000

            if cached:
                timing_info["semantic_cache_hit"] = True
                timing_info["semantic_cache_similarity"] = cached["similarity"]
                timing_info["generation_total_ms"] = (time.perf_counter() - generation_start) * 1000

                logger.info(
                    f"[TIMING] Semantic cache HIT: similarity={cached['similarity']:.4f}, "
                    f"total={timing_info['generation_total_ms']:.2f}ms"
                )

                return {
                    "result": DoclingRagStructuredResponse(**cached["response"]),
                    "timing": timing_info
                }

        # 3. Retrieve chunks (using precomputed embedding to avoid double computation)
        filtered_chunks, retrieval_timing = await self.retrieval_service.retrieve_similar_chunks(
            query_text=query_text,
            document_id=document_id,
            document_name=document_name,
            top_k=top_k,
            min_score=min_score,
            precomputed_embedding=query_embedding
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

        # 3a. Rerank chunks if reranker is enabled
        timing_info["reranker_enabled"] = self.reranker is not None
        if self.reranker:
            rerank_start = time.perf_counter()
            reranked_chunks = await self.reranker.rerank(
                query=query_text,
                documents=filtered_chunks,
                top_k=settings.reranker_top_k
            )
            timing_info["rerank_ms"] = (time.perf_counter() - rerank_start) * 1000
            timing_info["pre_rerank_count"] = len(filtered_chunks)
            timing_info["post_rerank_count"] = len(reranked_chunks)
            filtered_chunks = reranked_chunks
            logger.info(
                f"[RERANK] Reranked {timing_info['pre_rerank_count']} -> {timing_info['post_rerank_count']} chunks "
                f"in {timing_info['rerank_ms']:.2f}ms"
            )

        # 4. Check response cache (Redis exact match)
        if self.cache_service:
            cached_response = await self.cache_service.get_response(
                query_text,
                document_id,
                filtered_chunks
            )
            if cached_response:
                timing_info["response_cache_hit"] = True
                timing_info["generation_total_ms"] = (time.perf_counter() - generation_start) * 1000
                logger.info(f"[CACHE] Response [HIT] - Exact match found in Redis! Total: {timing_info['generation_total_ms']:.2f}ms (SAVED ~15s LLM call!)")
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

        # 5. Call Claude Opus 4.5 for generation
        llm_start = time.perf_counter()

        user_message = f"CONTEXT:\n{formatted_context}\n\nQUESTION: {query_text}"

        try:
            response = await _anthropic_client.messages.create(
                model="claude-opus-4-5-20251101",
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_message},
                ],
            )

            timing_info["llm_call_ms"] = (time.perf_counter() - llm_start) * 1000
            logger.info(f"[CACHE] LLM [CALL] - Claude Opus 4.5 responded in {timing_info['llm_call_ms']:.2f}ms (no cache available)")

            # Parse response - Claude returns content as a list of blocks
            raw_content = response.content[0].text
            logger.debug(f"[DEBUG] Raw LLM response: {raw_content[:500]}...")

            # Try to extract JSON from response (handle cases where model adds extra text)
            parsed = self._parse_llm_json(raw_content)

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
            logger.error(f"[ERROR] Claude API call failed: {e}")
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

        # 6. Cache response in Redis (exact match cache)
        if self.cache_service:
            await self.cache_service.set_response(
                query_text,
                document_id,
                filtered_chunks,
                result.model_dump()
            )
            logger.info(f"[CACHE] Response [STORE] - Saved to Redis for exact match (TTL: 30m)")

        # 7. Store in semantic cache (similarity-based cache)
        if self.semantic_cache_service:
            from app.services.cache.semantic_cache_service import SemanticCacheService
            context_hash = SemanticCacheService.hash_context(filtered_chunks)
            await self.semantic_cache_service.store_response(
                query_text=query_text,
                query_embedding=query_embedding,
                document_id=document_id,
                response=result.model_dump(),
                context_hash=context_hash
            )

        timing_info["generation_total_ms"] = (time.perf_counter() - generation_start) * 1000
        logger.info(f"[CACHE] === Generation Complete: {timing_info['generation_total_ms']:.2f}ms ===")

        return {
            "result": result,
            "timing": timing_info
        }