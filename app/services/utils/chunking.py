from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    content: str,
    metadata: Dict[str, Any] = None,
    chunk_size: int = 750,
    chunk_overlap: int = 150
) -> List[Document]:
    """Chunk content into chunks with overlap for better context"""
    print(f"🔧 CHUNKING CONFIG: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    doc = Document(
        page_content=content,
        metadata=metadata or {}
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
        add_start_index=True
    )
    chunks = text_splitter.split_documents([doc])

    print(f"📊 CHUNKING RESULTS: {len(chunks)} chunks created from {len(content)} characters")
    if len(chunks) > 0:
        print(f"   - First chunk size: {len(chunks[0].page_content)} chars")
        print(f"   - Last chunk size: {len(chunks[-1].page_content)} chars")
        if len(chunks) > 1:
            # Verify overlap by checking if end of chunk N appears in chunk N+1
            overlap_sample = chunks[0].page_content[-50:]
            has_overlap = overlap_sample in chunks[1].page_content
            print(f"   - Overlap verified: {'✅ YES' if has_overlap else '❌ NO'}")

    return chunks