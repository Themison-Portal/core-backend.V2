from typing import Dict, Any, List
from uuid import UUID

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.services.doclingRag.interfaces.rag_generation_service import IRagGenerationService
from app.services.doclingRag.rag_retrieval_service import RagRetrievalService
from app.schemas.rag_docling_schema import DoclingRagStructuredResponse


LLM_MODEL_NAME = "gpt-4o-mini"

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

    def __init__(self, retrieval_service: RagRetrievalService):
        self.retrieval_service = retrieval_service
        self.llm_model_name = LLM_MODEL_NAME    
    

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
        top_k: int = 40,
        min_score: float = 0.04
    ) -> DoclingRagStructuredResponse:
        
        # 1. Retrieve raw chunks
        filtered_chunks = await self.retrieval_service.retrieve_similar_chunks(
            query_text=query_text,
            document_id=document_id,
            top_k=top_k,
            min_score=min_score
        )

        if not filtered_chunks:
            return DoclingRagStructuredResponse(
                response="The provided documents do not contain this information.",
                sources=[]
            )
        
        # 2. Format context with metadata tags
        formatted_context = "\n\n".join([self._format_context_docling(d) for d in filtered_chunks])

        # 3. Setup the Structured LLM
        # We use your Pydantic model here to force the LLM to follow the schema
        structured_chat_model = ChatOpenAI(
            model=self.llm_model_name,
            temperature=0.0
        ).with_structured_output(DoclingRagStructuredResponse)

        # 4. Define the Chain
        chat_prompt = ChatPromptTemplate.from_template(UNIFIED_PROMPT_TEMPLATE)
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | chat_prompt
            | structured_chat_model
        )

        # 5. Execute
        # result will be an instance of DoclingRagStructuredResponse
        result = await chain.ainvoke({
            "context": formatted_context,
            "question": query_text
        })
        
        return result