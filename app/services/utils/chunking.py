from langchain_core.documents import Document, LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(content: str, metadata: Dict[str, Any] = None, chunk_size: int = 1000) -> List[LangchainDocument]:
    """Chunk content into chunks"""
    doc = LangchainDocument(
        page_content=content,
        metadata=metadata or {}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    return text_splitter.split_documents([doc])