"""Tools for document retrieval in agentic RAG system."""

from typing import List

from langchain_core.tools import tool


@tool
def retrieve_documents_tool(query: str, document_ids: List[str]):
    """Retrieve documents based on query and document IDs."""
    pass