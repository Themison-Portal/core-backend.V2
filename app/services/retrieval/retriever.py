import asyncio
import re
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.embeddings import EmbeddingProvider
from app.services.utils.preprocessing import preprocess_text
from app.supabase_client.supabase_client import supabase_client

supabase = supabase_client()

def preprocess_query(query: str) -> str:
    """Clean and normalize the query text using the same preprocessing as documents."""
    return preprocess_text(query, clean_whitespace=True)

def create_retriever(
    embedding_provider: EmbeddingProvider,
    match_count: int = 10,
):  
    async def retrieve(
        query: str,
        override_match_count: Optional[int] = None
    ) -> List[Dict[Any, Any]]:
        # Generate embedding for the query
        embeddings = await embedding_provider.get_embeddings_batch([query])
        query_embedding = embeddings[0]
        
        # Use override parameters if provided, otherwise use defaults
        count = override_match_count if override_match_count is not None else match_count        
        # Call the hybrid_search function using RPC
        result = await asyncio.to_thread(
            lambda: supabase.rpc(
                "hybrid_search",
                {
                    "query_text": query,
                    "query_embedding": query_embedding,
                    "match_count": count
                }
            ).execute()
        )
        
        # Ensure all data is JSON serializable
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