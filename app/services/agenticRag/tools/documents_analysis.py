"""Tools for document analysis in agentic RAG system."""

from typing import List

from langchain_core.tools import tool


@tool
def documents_analysis_tool(query: str, document_ids: List[str]):
    """Analyze documents based on query and document IDs."""
    pass