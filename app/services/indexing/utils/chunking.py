"""
Chunking utilities for indexing documents
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
) -> List[Document]:
    """
    Split documents into chunks using recursive character splitting
    
    Args:
        documents: List of documents to split
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    return text_splitter.split_documents(documents)