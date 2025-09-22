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
            source = doc.get('metadata', {}).get('source', 'Unknown')
            page = doc.get('metadata', {}).get('page', 'Unknown')
            content = doc.get('content', 'No content available')
            context += f"\nDocument {i+1} (Source: {source}, Page: {page}):\n{content}\n"
        
        prompt = f"""
        User Query: {query}
        
        Retrieved Information:
        {context}
        
        Based on the above information only, provide a comprehensive answer to the user's query.
        If the information is not sufficient to answer the query, acknowledge this limitation.
        Include citations to the specific documents you reference.
        
        Remember to present the documents retrieved with the page numbers based on the metadata accompanying the documents and clarify which document the information is from.
        """
        
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Sorry, I encountered an error generating a response: {str(e)}"

@tool(response_format="content_and_artifact")
def documents_retrieval_generation_tool(
    query: str,
    match_count: int = 5,
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
        # Step 1: Document Retrieval
        processed_query = preprocess_query(query)

        embedding = embedding_client.embed_query(processed_query)

        rpc_params = {
            "query_text": processed_query,
            "query_embedding": embedding,
            "match_count": match_count,
            "document_ids": document_ids,
        }
        
        result = supabase_client().rpc("hybrid_search", rpc_params).execute()
        
        data = result.data if hasattr(result, "data") else []
        
        retrieved_docs = _ensure_serializable(data or [])
        
        # Step 2: Response Generation
        if not retrieved_docs or (len(retrieved_docs) == 1 and "error" in retrieved_docs[0]):
            return {
                "retrieved_documents": [],
                "retrieved_documents_metadata": [],
                "generated_response": "I couldn't find any relevant documents to answer your question. Please try rephrasing your query or check if the documents are available.",
                "success": False
            }
        
        generation = generate_response(query, retrieved_docs)
        retrieved_docs_metadata = [doc.get('chunk_metadata', {}) for doc in retrieved_docs]
        
        return generation, {
            "retrieved_documents": retrieved_docs,
            "retrieved_documents_metadata": retrieved_docs_metadata,
            "generated_response": generation,
            "success": True
        }

    except Exception as e:
        error_msg = f"An error occurred while processing your request: {str(e)}"
        return error_msg, {
            "retrieved_documents": [],
            "retrieved_documents_metadata": [],
            "generated_response": f"An error occurred while processing your request: {str(e)}",
            "success": False
        }