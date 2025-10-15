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
) -> str:
    """
    Generate a response to the query based on the retrieved documents.
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
            context += f"\nDocument {i+1} ({page_display}, Source: {source}):\n{content}\n"
        
        prompt = f"""You are an expert medical document analyst. Answer the user's question using ONLY information from the provided document excerpts.

DOCUMENT EXCERPTS:
{context}

USER QUESTION: {query}

INTELLIGENT CITATION STRATEGY:
Before citing, analyze the TYPE of information you're providing:

A) STRUCTURED LISTS (eligibility criteria, exclusion criteria, endpoints, procedures):
   - If presenting a complete list (e.g., all inclusion criteria), cite the SECTION ONCE at the beginning
   - Format: "According to Section X.X (Page Y), the inclusion criteria include:"
   - Then list items WITHOUT individual citations
   - Example: "Section 5.1 Inclusion Criteria (Page 34) specifies: 1) Age 18-70 years, 2) Active UC confirmed by colonoscopy..."

B) SPECIFIC FACTS or ISOLATED CLAIMS:
   - Cite each claim individually
   - Format: [Page X: "exact quote"]
   - Use when answering questions about specific details, dates, dosages, or single facts
   - Example: "The study duration is 52 weeks [Page 15: 'study duration of 52 weeks']"

C) MIXED CONTENT (narrative with specific data points):
   - Cite the overall section reference first
   - Add specific citations only for critical numbers, dates, or controversial claims
   - Example: "The study design (Section 3, Pages 10-12) uses a randomized approach. The primary endpoint is clinical remission at week 12 [Page 11: 'primary endpoint: clinical remission at week 12']"

CITATION FORMATTING RULES:
1. Section citations: "Section X.X Title (Page Y)" or "(Pages Y-Z)"
2. Individual citations: [Page X: "exact quote under 80 chars"]
3. For lists >5 items: Cite the section, not each item
4. Exact quotes only - no paraphrasing in citations
5. If unsure, default to section citation

RESPONSE STYLE:
- Be comprehensive and clear
- Do NOT repeat the question back
- Choose the citation strategy that provides the MOST USEFUL reference for verification
- Prioritize readability while maintaining citation accuracy

Provide your answer now:"""
        
        print("🔍 ANTHROPIC REQUEST:")
        print(f"📝 Query: {query}")
        print(f"📄 Context length: {len(context)} characters")
        print("🚀 Sending to Anthropic Claude...")

        response = llm.invoke(prompt)
        content = response.content

        print("✅ ANTHROPIC RESPONSE RECEIVED:")
        print(f"📝 Response length: {len(content)} characters")
        print(f"🔗 Raw response preview: {content[:200]}...")

        # Extract and log citations
        import re
        citation_pattern = r'\[Page (\d+): [\'"]([^\'"]+)[\'"]\]'
        citations_found = re.findall(citation_pattern, content)

        print(f"🎯 CITATIONS EXTRACTED: {len(citations_found)} citations found")
        for i, (page, quote) in enumerate(citations_found, 1):
            print(f"  📖 Citation {i}: Page {page} - \"{quote[:50]}{'...' if len(quote) > 50 else ''}\"")

        if not citations_found:
            print("⚠️  NO CITATIONS FOUND - Checking for general quotes...")
            general_quotes = re.findall(r'[\'"]([^\'"]{20,})[\'"]', content)
            if general_quotes:
                print(f"📝 Found {len(general_quotes[:3])} general quotes without page numbers:")
                for i, quote in enumerate(general_quotes[:3], 1):
                    print(f"  💬 Quote {i}: \"{quote[:50]}{'...' if len(quote) > 50 else ''}\"")

        return content
        
    except Exception as e:
        return f"Sorry, I encountered an error generating a response: {str(e)}"

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
        print("🔍 RAG TOOL STARTED:")
        print(f"📝 Original query: {query}")
        print(f"🎯 Document filter: {document_ids}")
        print(f"📊 Max results: {match_count}")

        # Step 1: Document Retrieval
        processed_query = preprocess_query(query)
        print(f"🔧 Processed query: {processed_query}")

        embedding = embedding_client.embed_query(processed_query)
        print(f"🎯 Generated embedding vector length: {len(embedding)}")

        rpc_params = {
            "query_text": processed_query,
            "query_embedding": embedding,
            "match_count": match_count,
            "document_ids": document_ids,
        }
        
        print("🔍 Searching Supabase vector database...")
        result = supabase_client().rpc("hybrid_search", rpc_params).execute()

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
            return {
                "retrieved_documents": [],
                "generated_response": "I couldn't find any relevant documents to answer your question. Please try rephrasing your query or check if the documents are available.",
                "success": False
            }
        
        generation = generate_response(query, retrieved_docs)
        retrieved_docs_metadata = [doc.get('chunk_metadata', {}) for doc in retrieved_docs]
        retrieved_docs_content = [doc.get('content', '') for doc in retrieved_docs]

        print(f"🎯 FINAL TOOL RESPONSE:")
        print(f"📝 Generation (length: {len(generation)}): {generation[:100]}...")
        print(f"📚 Docs metadata: {len(retrieved_docs_metadata)} items")

        return generation, {
            "retrieved_documents": retrieved_docs_content,
            "generated_response": generation,
            "success": True
        }

    except Exception as e:
        error_msg = f"An error occurred while processing your request: {str(e)}"
        return error_msg, {
            "retrieved_documents": [],
            "generated_response": f"An error occurred while processing your request: {str(e)}",
            "success": False
        }