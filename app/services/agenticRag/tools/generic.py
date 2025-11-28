"""Generic tools for agentic RAG system."""

from langchain.tools import tool


@tool
def generic_tool(query: str):
    """
    Generic tool for any task.
    
    Use this tool when you cannot figure what other tools to implement. This tool should cover 
    generic knowledge that is not relevant to medical knowledge such protocols, procedures, patient information, etc.
    
    Args:
        query: The query from user to search for relevant information
    
    Returns:
        A response to the user's query with your best guess/ knowledge of the answer
    """
    # Add your generic knowledge logic here
    # This could be a simple response, call to an external API, or basic reasoning
    return f"I understand you're asking about: {query}. This appears to be a general question outside the scope of my specialized medical tools."