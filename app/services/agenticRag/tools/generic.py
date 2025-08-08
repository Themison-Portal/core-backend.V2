"""Generic tools for agentic RAG system."""

from langchain_core.tools import tool


@tool
def generic_tool(query: str):
    """Generic tool for any task."""
    pass