"""Tools for document retrieval in agentic RAG system."""
from typing import Any, Dict, List
import numpy as np
from langchain.tools import tool
from app.core.openai import embedding_client, llm
from app.core.supabase_client import supabase_client
from app.services.utils.preprocessing import preprocess_text
import json
import re

def preprocess_query(query: str) -> str:
    """Clean and normalize the query text using the same preprocessing as documents."""
    return preprocess_text(query, clean_whitespace=True)

def _ensure_serializable(data):
    """Recursively convert any NumPy arrays to lists to ensure JSON serializability."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: _ensure_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_ensure_serializable(item) for item in data]
    else:
        return data

def generate_response(
    query: str,
    retrieved_documents: List[Dict[Any, Any]]
) -> tuple[str, List[Dict]]:
    """
    Generate a response to the query based on the retrieved documents.
    Returns: (answer_text, list_of_citations)
    """
    try:
        context = ""
        for i, doc in enumerate(retrieved_documents):
            chunk_metadata = doc.get('chunk_metadata', {})
            metadata = doc.get('metadata', {})

            source = (
                metadata.get('filename') or
                chunk_metadata.get('filename') or
                metadata.get('source', 'Unknown')
            )

            page_numbers = (
                chunk_metadata.get('page_numbers') or
                metadata.get('page_numbers') or
                ([metadata.get('page')] if metadata.get('page') else [])
            )

            if page_numbers and any(p for p in page_numbers if p is not None):
                valid_pages = [p for p in page_numbers if p is not None]
                page_display = f"Page {valid_pages[0]}" if len(valid_pages) == 1 else f"Pages {'-'.join(map(str, valid_pages))}"
            else:
                page_display = "Page Unknown"

            content = doc.get('content', 'No content available')
            context += f"\n[CHUNK {i}] ({page_display}, Source: {source}):\n{content}\n"

        prompt = f"""You are an expert medical document analyst. Answer the user's question using ONLY information from the provided document chunks.

DOCUMENT CHUNKS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Analyze all chunks and provide a detailed, well-structured answer
2. Use ONLY information from the chunks above
3. When listing items (criteria, requirements, tests), use markdown numbered lists (1. 2. 3.) or bullet points (-)
4. Use markdown headers (## Title) to organize sections when appropriate
5. For each piece of information you use, extract the EXACT relevant text snippet (verbatim quote)
6. Include the page number for each citation
7. Keep your answer under 2000 characters
8. If a fact cannot be directly supported by a quote, respond: "Not enough information."
9. Do NOT infer or guess beyond the document text
10. Prefer shorter, factual answers strictly grounded in the retrieved chunks

CRITICAL: Respond with VALID JSON ONLY in this exact format:
{{
  "answer": "Your detailed, markdown-formatted answer here (max 2000 chars)",
  "citations": [
    {{
      "chunk_index": 0,
      "exact_quote": "The exact verbatim text from the chunk",
      "page": 35
    }}
  ],
  "confidence": 0.95
}}"""

        response = llm.invoke(prompt)
        content = response.content.strip()

        # Try to extract JSON if wrapped in markdown or extra text
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
        else:
            parsed = json.loads(content)

        answer_text = parsed.get('answer', '')
        citations = parsed.get('citations', [])
        confidence = parsed.get('confidence', 0.8)

        return answer_text, citations

    except json.JSONDecodeError as e:
        fallback_citations = [
            {
                "chunk_index": i,
                "exact_quote": doc.get('content', '')[:300],
                "page": doc.get('chunk_metadata', {}).get('page_numbers', [1])[0]
            }
            for i, doc in enumerate(retrieved_documents[:3])
        ]
        return content, fallback_citations
    except Exception as e:
        return f"Sorry, I encountered an error generating a response: {str(e)}", []

def weighted_hybrid_rerank(
    query_embedding: List[float],
    retrieved_docs: List[Dict[str, Any]],
    weight_supabase: float = 0.3,
    weight_embedding: float = 0.7,
    min_score: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Rerank retrieved documents using a weighted hybrid of Supabase score and embedding similarity.
    Only returns docs above min_score.
    """
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    scored_docs = []
    for doc in retrieved_docs:
        supabase_score = doc.get('score', 0.0)
        chunk_embedding = doc.get('chunk_metadata', {}).get('embedding')
        if chunk_embedding is not None:
            embed_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding))
        else:
            embed_score = 0.0
        final_score = weight_supabase * supabase_score + weight_embedding * embed_score
        if final_score >= min_score:
            scored_docs.append((final_score, doc))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs]

@tool(response_format="content_and_artifact")
def documents_retrieval_generation_tool(
    query: str,
    match_count: int = 10,
    document_ids: List[str] = None,
    query_chunk_size: int = 500,
    weight_supabase: float = 0.3,
    weight_embedding: float = 0.7,
    min_score: float = 0.0
) -> Dict[str, Any]:
    """
    Retrieve relevant documents, rerank using weighted hybrid scoring, and generate response.
    """
    import time
    start_time = time.time()

    try:
        print("🔍 RAG TOOL STARTED")
        processed_query = preprocess_query(query)
        query_embedding = embedding_client.embed_query(processed_query)

        # Step 1: Retrieve more chunks for reranking
        rpc_params = {
            "query_text": processed_query,
            "query_embedding": query_embedding,
            "match_count": match_count * 10,  # dynamic over-retrieval
            "document_ids": document_ids,
        }

        result = supabase_client().rpc("hybrid_search", rpc_params).execute()
        data = result.data if hasattr(result, "data") else []
        retrieved_docs = _ensure_serializable(data or [])
        print(f"RAG TOOL: Retrieved {len(retrieved_docs)} documents from Supabase in {time.time() - start_time:.2f}s")
        print(f"retrieved dosc : {retrieved_docs[0:2]}")

        if not retrieved_docs:
            return (
                "No relevant documents found.",
                {
                    "retrieved_documents": [],
                    "generated_response": "No relevant documents found.",
                    "used_chunks": [],
                    "confidence": 0.0,
                    "success": False
                }
            )

        # Step 2: Weighted hybrid rerank
        retrieved_docs = weighted_hybrid_rerank(
            query_embedding,
            retrieved_docs,
            weight_supabase=weight_supabase,
            weight_embedding=weight_embedding,
            min_score=min_score
        )
        retrieved_docs = retrieved_docs[:match_count]
        print(f"RAG TOOL: Reranked {len(retrieved_docs)} documents using hybrid scoring in {time.time() - start_time:.2f}s")

        # Step 3: Generate response
        print("RAG TOOL: Proceeding to generate response...")
        answer_text, citations = generate_response(query, retrieved_docs)
        print(f"RAG TOOL: Generated response in {time.time() - start_time:.2f}s")

        # Step 4: Build used_chunks metadata
        used_chunks_with_metadata = []
        for citation in citations:
            chunk_idx = citation.get("chunk_index")
            exact_quote = citation.get("exact_quote", "")
            page = citation.get("page")
            if chunk_idx is not None and 0 <= chunk_idx < len(retrieved_docs):
                chunk = retrieved_docs[chunk_idx]
                chunk_metadata = chunk.get('chunk_metadata', {})
                used_chunks_with_metadata.append({
                    "chunk_index": chunk_idx,
                    "content": chunk.get('content', ''),
                    "exact_quote": exact_quote,
                    "page_numbers": chunk_metadata.get('page_numbers', [page] if page else []),
                    "filename": chunk_metadata.get('filename', 'Unknown'),
                    "metadata": chunk_metadata
                })

        return answer_text, {
            "retrieved_documents": retrieved_docs,
            "used_chunks": used_chunks_with_metadata,
            "generated_response": answer_text,
            "success": True
        }

    except Exception as e:
        print(f"❌ Error in RAG tool: {e}")
        return (
            f"An error occurred: {str(e)}",
            {
                "retrieved_documents": [],
                "generated_response": f"An error occurred: {str(e)}",
                "success": False
            }
        )
