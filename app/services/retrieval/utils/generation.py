"""
Generation utilities for retrieval
"""

import logging
from typing import Any, Dict, List

from dotenv import load_dotenv

from app.core.openAI import chat_client

load_dotenv()

logger = logging.getLogger(__name__)

def generate_response(query: str, retrieved_documents: List[Dict[Any, Any]]) -> str:
    """
    Generate a response to a query based on retrieved documents
    """

    # Format retrieved documents into context
    context = ""
    print(f"retrieved_documents: {retrieved_documents}")
    for i, doc in enumerate(retrieved_documents):
        source = doc.get('metadata', {}).get('source', 'Unknown')
        page = doc.get('metadata', {}).get('page', 'Unknown')
        context += f"\nDocument {i+1} (Source: {source}, Page: {page}):\n{doc['content']}\n"
    
    # Create a structured prompt
    prompt = f"""
    User Query: {query}
    
    Retrieved Information:
    {context}
    
    Based on the above information only, provide a comprehensive answer to the user's query.
    If the information is not sufficient to answer the query, acknowledge this limitation.
    Include citations to the specific documents you reference.
    """
    
    return prompt

async def call_llm_stream(prompt):
    """
    Call the LLM API and stream the response using LangChain
    """
    try:
        # LangChain streaming approach
        async for chunk in chat_client.astream(prompt):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        logger.error("Error calling LLM API: %s", e)
        yield "Sorry, I encountered an error generating a response."