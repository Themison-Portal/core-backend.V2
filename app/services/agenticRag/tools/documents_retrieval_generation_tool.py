"""Tools for document retrieval in agentic RAG system."""

from typing import Any, Dict, List

import numpy as np
from langchain.tools import tool

from app.core.openai import embedding_client, llm
from app.core.supabase_client import supabase_client
from app.services.utils.preprocessing import preprocess_text


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
    where each citation is: {"chunk_index": int, "exact_quote": str, "page": int}
    """
    try:
        context = ""
        for i, doc in enumerate(retrieved_documents):
            # Extract source from filename or chunk_metadata
            chunk_metadata = doc.get('chunk_metadata', {})
            metadata = doc.get('metadata', {})

            source = (
                metadata.get('filename') or
                chunk_metadata.get('filename') or
                metadata.get('source', 'Unknown')
            )

            # Extract page numbers - could be in different places
            page_numbers = (
                chunk_metadata.get('page_numbers') or
                metadata.get('page_numbers') or
                ([metadata.get('page')] if metadata.get('page') else [])
            )

            # Format page display
            if page_numbers and any(p for p in page_numbers if p is not None):
                valid_pages = [p for p in page_numbers if p is not None]
                if len(valid_pages) == 1:
                    page_display = f"Page {valid_pages[0]}"
                else:
                    page_display = f"Pages {'-'.join(map(str, valid_pages))}"
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
7. Keep your answer under 2000 characters for optimal readability

FORMATTING GUIDELINES:
- Use markdown lists for enumerated items
- Use bold (**text**) for emphasis on key terms
- Group related information under headers (## Header)
- Keep each list item concise but complete

Example format:
## Inclusion Criteria
1. Written informed consent obtained prior to study procedures
2. Age ≥ 18 years at signing of informed consent
3. Histologically confirmed diagnosis

## Exclusion Criteria
1. Prior therapy with anti-PD-1 agents
2. Active autoimmune disease

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
}}

Where:
- "answer": Your markdown-formatted response (string, MAXIMUM 2000 characters)
- "citations": Array of objects with chunk_index, exact_quote (VERBATIM text), and page number
- "confidence": Your confidence level 0.0-1.0 (number)

IMPORTANT:
- The "exact_quote" MUST be verbatim text from the chunk (for PDF highlighting)
- Keep answer UNDER 2000 characters or it will be truncated
- Use markdown formatting for lists and structure
- Be detailed and well-organized

Return ONLY valid JSON, no additional text before or after."""

        print("🔍 ANTHROPIC REQUEST:")
        print(f"📝 Query: {query}")
        print(f"📄 Context: {len(retrieved_documents)} chunks, {len(context)} characters")
        print("🚀 Sending to Anthropic Claude...")

        response = llm.invoke(prompt)
        content = response.content.strip()

        print("✅ ANTHROPIC RESPONSE RECEIVED:")
        print(f"📝 Response length: {len(content)} characters")
        print(f"🔗 Raw response preview: {content[:300]}...")

        # Parse JSON response
        import json
        import re

        # Try to extract JSON if wrapped in markdown or extra text
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
        else:
            # Fallback: try to parse entire content
            parsed = json.loads(content)

        answer_text = parsed.get('answer', '')
        citations = parsed.get('citations', [])
        confidence = parsed.get('confidence', 0.8)

        print(f"✅ JSON PARSED SUCCESSFULLY:")
        print(f"  📝 Answer length: {len(answer_text)} chars")
        print(f"  📊 Citations: {len(citations)}")
        print(f"  🎯 Confidence: {confidence}")

        for i, citation in enumerate(citations[:3]):
            chunk_idx = citation.get('chunk_index', '?')
            page = citation.get('page', '?')
            quote_preview = citation.get('exact_quote', '')[:100]
            print(f"    Citation {i+1}: Chunk {chunk_idx}, Page {page} - \"{quote_preview}...\"")

        return answer_text, citations


    except json.JSONDecodeError as e:
        print(f"❌ JSON PARSE ERROR: {str(e)}")
        print(f"Raw content: {content[:500]}")
        # Fallback: return content as-is and generate generic citations
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
        error_msg = f"Sorry, I encountered an error generating a response: {str(e)}"
        print(f"❌ GENERATION ERROR: {error_msg}")
        return error_msg, []

@tool(response_format="content_and_artifact")
def documents_retrieval_generation_tool(
    query: str,
    match_count: int = 6,  # Balanced between coverage and cost
    document_ids: List[str] = None,
    query_chunk_size: int = 500
) -> Dict[str, Any]:
    """
    Search for relevant documents and generate a response based on the retrieved content.
    
    Args:
        query: The search query
        match_count: Maximum number of results to return
        document_ids: Optional list of specific document IDs (UUID strings) to filter
        query_chunk_size: Unused; kept for signature compatibility
        
    Returns:
        Dictionary containing both retrieved documents and generated response
    """
    
    try:
        import time
        start_time = time.time()

        print("🔍 RAG TOOL STARTED:")
        print(f"📝 Original query: {query}")
        print(f"🎯 Document filter: {document_ids}")
        print(f"📊 Max results: {match_count}")

        # Step 1: Document Retrieval
        t1 = time.time()
        processed_query = preprocess_query(query)
        print(f"🔧 Processed query: {processed_query}")

        embedding = embedding_client.embed_query(processed_query)
        print(f"🎯 Generated embedding vector length: {len(embedding)}")
        print(f"⏱️  Embedding generation took: {time.time() - t1:.2f}s")

        rpc_params = {
            "query_text": processed_query,
            "query_embedding": embedding,
            "match_count": match_count,
            "document_ids": document_ids,
        }

        t2 = time.time()
        print("🔍 Searching Supabase vector database...")
        result = supabase_client().rpc("hybrid_search", rpc_params).execute()
        print(f"⏱️  Hybrid search took: {time.time() - t2:.2f}s")

        data = result.data if hasattr(result, "data") else []

        retrieved_docs = _ensure_serializable(data or [])

        print(f"📚 RETRIEVED DOCUMENTS: {len(retrieved_docs)} chunks found")
        for i, doc in enumerate(retrieved_docs[:3], 1):  # Show first 3
            # Debug: Show all metadata structures
            chunk_metadata = doc.get('chunk_metadata', {})
            metadata = doc.get('metadata', {})

            print(f"  🔍 Doc {i} RAW METADATA:")
            print(f"    - chunk_metadata: {chunk_metadata}")
            print(f"    - metadata: {metadata}")

            source = (
                metadata.get('filename') or
                chunk_metadata.get('filename') or
                metadata.get('source', 'Unknown')
            )

            # Extract page numbers - could be in different places
            page_numbers = (
                chunk_metadata.get('page_numbers') or
                metadata.get('page_numbers') or
                ([metadata.get('page')] if metadata.get('page') else [])
            )

            print(f"    🔧 EXTRACTED page_numbers: {page_numbers}")

            if page_numbers and any(p for p in page_numbers if p is not None):
                valid_pages = [p for p in page_numbers if p is not None]
                if len(valid_pages) == 1:
                    page_display = f"Page {valid_pages[0]}"
                else:
                    page_display = f"Pages {'-'.join(map(str, valid_pages))}"
                print(f"    ✅ FINAL page_display: {page_display}")
            else:
                page_display = "Page Unknown"
                print(f"    ❌ NO VALID PAGES - page_numbers: {page_numbers}")

            content_preview = doc.get('content', '')[:100] + "..." if len(doc.get('content', '')) > 100 else doc.get('content', '')
            print(f"  📄 Doc {i}: {source} ({page_display}) - \"{content_preview}\"")

        # Step 2: Response Generation
        if not retrieved_docs or (len(retrieved_docs) == 1 and "error" in retrieved_docs[0]):
            print(f"⏱️  TOTAL RAG TOOL TIME: {time.time() - start_time:.2f}s")
            return {
                "retrieved_documents": [],
                "generated_response": "I couldn't find any relevant documents to answer your question. Please try rephrasing your query or check if the documents are available.",
                "used_chunks": [],
                "confidence": 0.0,
                "success": False
            }

        # Generate response with JSON output (returns citations with exact quotes)
        t3 = time.time()
        answer_text, citations = generate_response(query, retrieved_docs)
        print(f"⏱️  LLM generation took: {time.time() - t3:.2f}s")

        # Get metadata for citations with exact quotes
        used_chunks_with_metadata = []
        for citation in citations:
            chunk_idx = citation.get('chunk_index')
            exact_quote = citation.get('exact_quote', '')
            page = citation.get('page')

            if chunk_idx is not None and 0 <= chunk_idx < len(retrieved_docs):
                chunk = retrieved_docs[chunk_idx]
                chunk_metadata = chunk.get('chunk_metadata', {})

                used_chunks_with_metadata.append({
                    "chunk_index": chunk_idx,
                    "content": chunk.get('content', ''),  # Full chunk content
                    "exact_quote": exact_quote,  # Exact text snippet LLM used
                    "page_numbers": chunk_metadata.get('page_numbers', [page] if page else []),
                    "filename": chunk_metadata.get('filename', 'Unknown'),
                    "metadata": chunk_metadata
                })

        print(f"🎯 FINAL TOOL RESPONSE:")
        print(f"📝 Answer (length: {len(answer_text)}): {answer_text[:100]}...")
        print(f"📊 Citations: {len(citations)}")
        print(f"📚 Sources to return: {len(used_chunks_with_metadata)} chunks with exact quotes")
        print(f"⏱️  TOTAL RAG TOOL TIME: {time.time() - start_time:.2f}s")

        # Return answer as content, structured data as artifact
        return answer_text, {
            "retrieved_documents": retrieved_docs,  # All retrieved docs
            "used_chunks": used_chunks_with_metadata,  # Only chunks LLM used
            "generated_response": answer_text,
            "success": True
        }

    except Exception as e:
        error_msg = f"An error occurred while processing your request: {str(e)}"
        print(f"⏱️  TOTAL RAG TOOL TIME (with error): {time.time() - start_time:.2f}s")
        return error_msg, {
            "retrieved_documents": [],
            "generated_response": f"An error occurred while processing your request: {str(e)}",
            "success": False
        }