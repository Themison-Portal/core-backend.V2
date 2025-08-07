import asyncio
import re
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.documents import Document

from app.core.embeddings import EmbeddingProvider
from app.core.supabase_client import supabase_client
from app.services.indexing.utils.chunking import chunk_documents
from app.services.utils.preprocessing import preprocess_text

supabase = supabase_client()

def preprocess_query(query: str) -> str:
    """Clean and normalize the query text using the same preprocessing as documents."""
    return preprocess_text(query, clean_whitespace=True)

def chunk_query(query: str, chunk_size: int = 500) -> List[str]:
    """
    Chunk a long query into smaller pieces for better retrieval
    
    Args:
        query: The original query
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of query chunks
    """
    if len(query) <= chunk_size:
        return [query]
    
    # Create a document for chunking
    doc = Document(page_content=query, metadata={})
    chunks = chunk_documents([doc], chunk_size)
    return [chunk.page_content for chunk in chunks]

async def aggregate_search_results(
    chunk_results: List[List[Dict[Any, Any]]], 
    max_results: int = 10
) -> List[Dict[Any, Any]]:
    """
    Aggregate results from multiple query chunks, removing duplicates
    and ranking by frequency and relevance
    """
    # Count document occurrences across chunks
    doc_scores = {}
    
    for chunk_result in chunk_results:
        for doc in chunk_result:
            doc_id = doc.get('id') or doc.get('content', '')[:100]  # Use content as fallback
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'doc': doc,
                    'count': 0,
                    'total_score': 0.0
                }
            doc_scores[doc_id]['count'] += 1
            doc_scores[doc_id]['total_score'] += doc.get('combined_score', 0.0)
    
    # Sort by frequency and average score
    ranked_docs = []
    for doc_info in doc_scores.values():
        avg_score = doc_info['total_score'] / doc_info['count']
        # Boost score based on frequency across chunks
        final_score = avg_score * (1 + 0.1 * doc_info['count'])
        ranked_docs.append({
            **doc_info['doc'],
            'final_score': final_score,
            'chunk_frequency': doc_info['count']
        })
    
    # Sort by final score and return top results
    ranked_docs.sort(key=lambda x: x['final_score'], reverse=True)
    return ranked_docs[:max_results]

def create_retriever(
    embedding_provider: EmbeddingProvider,
    match_count: int = 10,
    query_chunk_size: int = 500,
):  
    async def retrieve(
        query: str,
        override_match_count: Optional[int] = None
    ) -> List[Dict[Any, Any]]:
        # Preprocess the query
        processed_query = preprocess_query(query)
        
        # Check if query needs chunking
        if len(processed_query) > query_chunk_size:
            # Chunk the query
            query_chunks = chunk_query(processed_query, query_chunk_size)
            
            # Search with each chunk
            chunk_results = []
            for i, chunk in enumerate(query_chunks):
                
                # Generate embedding for this chunk
                embeddings = await embedding_provider.get_embeddings_batch([chunk])
                chunk_embedding = embeddings[0]
                print(f"chunk_embedding: {chunk_embedding}")
                
                # Convert embedding to PostgreSQL vector format
                embedding_vector = f"[{','.join(map(str, chunk_embedding))}]"
                
                # Search with this chunk
                count = override_match_count if override_match_count is not None else match_count
                result = await asyncio.to_thread(
                    lambda: supabase.rpc(
                        "hybrid_search",
                        {
                            "query_text": chunk,
                            "query_embedding": embedding_vector,  # Pass as vector string
                            "match_count": count
                        }
                    ).execute()
                )
                
                print(f"DEBUG: hybrid_search returned {len(result.data)} results")
                chunk_results.append(_ensure_serializable(result.data))
            
            # Aggregate results from all chunks
            final_results = await aggregate_search_results(chunk_results, match_count)
            return final_results
        else:
            # Use original single-query approach for short queries
            embeddings = await embedding_provider.get_embeddings_batch([processed_query])
            query_embedding = embeddings[0]
            print(f"query_embedding: {type(query_embedding)}")
            
            # Convert embedding to PostgreSQL vector format
            embedding_vector = f"[{','.join(map(str, query_embedding))}]"
            
            count = override_match_count if override_match_count is not None else match_count        
            result = await asyncio.to_thread(
                lambda: supabase.rpc(
                    "hybrid_search",
                    {
                        "query_text": processed_query,
                        "query_embedding": embedding_vector,  # Pass as vector string
                        "match_count": count
                    }
                ).execute()
            )
            
            return _ensure_serializable(result.data)
        
    return retrieve

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