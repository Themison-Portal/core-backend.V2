"""Tools for response generation in agentic RAG system."""

import logging
from typing import Any, Dict, List

from langchain.tools import tool

from app.core.openai import llm

logger = logging.getLogger(__name__)


@tool
async def generate_response_tool(
    query: str,
    retrieved_documents: List[Dict[Any, Any]]
) -> str:
    """
    Generate a comprehensive response based on retrieved documents and user query.
    
    Use this tool after retrieving documents to create a well-structured answer
    that cites specific sources and acknowledges limitations when information is insufficient.
    
    Args:
        query: The user's original question
        retrieved_documents: List of relevant documents retrieved from the knowledge base
    
    Returns:
        A comprehensive response that answers the user's query using the retrieved documents
    """
    try:
        # Format retrieved documents into context
        context = ""
        for i, doc in enumerate(retrieved_documents):
            source = doc.get('metadata', {}).get('source', 'Unknown')
            page = doc.get('metadata', {}).get('page', 'Unknown')
            content = doc.get('content', 'No content available')
            context += f"\nDocument {i+1} (Source: {source}, Page: {page}):\n{content}\n"
        
        # Create a structured prompt
        prompt = f"""
        User Query: {query}
        
        Retrieved Information:
        {context}
        
        Based on the above information only, provide a comprehensive answer to the user's query.
        If the information is not sufficient to answer the query, acknowledge this limitation.
        Include citations to the specific documents you reference.
        """
        
        # Call the LLM and return response
        response = await llm.ainvoke(prompt)
        return response.content
        
    except Exception as e:
        logger.error("Error generating response: %s", e)
        return f"Sorry, I encountered an error generating a response: {str(e)}"