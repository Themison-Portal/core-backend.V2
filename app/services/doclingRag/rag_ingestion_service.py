from typing import List
from uuid import UUID
from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.documents import Document
from uuid import uuid4

from app.models.documents import Document as DocumentTable
from app.models.chunks_docling import DocumentChunkDocling
from app.services.doclingRag.interfaces.rag_ingestion_service import IRagIngestionService
from app.core.openai import embedding_client
from docling.chunking import HybridChunker
from langchain_docling.loader import DoclingLoader, ExportType
from app.services.utils.tokenizer import get_tokenizer

class RagIngestionService(IRagIngestionService):
    """
    Service for PDF ingestion and chunking using Docling + embeddings.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedding_client = embedding_client

    # --------------------------
    # Private helper functions
    # --------------------------
    def _extract_docling_citation_metadata(self, metadata_json):
        """
        Returns a dict with page_number and headings for a chunk.
        """
        try:
            dl_meta = metadata_json.get("dl_meta", {})
            doc_items = dl_meta.get("doc_items", [])
            headings = dl_meta.get("headings", [])

            page_number = None
            if doc_items:
                prov_list = doc_items[0].get("prov", [])
                if prov_list:
                    page_number = prov_list[0].get("page_no")

            return {"page_number": page_number, "headings": headings or []}

        except Exception:
            return {"page_number": None, "headings": []}

    async def _insert_docling_chunks(
        self,
        document_id: UUID,
        chunks: List[Document],
        embeddings: List[List[float]],
        user_id: UUID = None,
    ):
        """
        Private helper to insert Docling chunks into DB.
        """
        await self.ensure_tables_exist()  # Make sure table exists

        try:
            document = await self.db.get(DocumentTable, document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found.")

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                citation_meta = self._extract_docling_citation_metadata(chunk.metadata)
                chunk_record = DocumentChunkDocling(
                    id=uuid4(),
                    document_id=document.id,
                    content=chunk.page_content,
                    page_number=citation_meta["page_number"],
                    chunk_metadata={**chunk.metadata, "chunk_index": i},
                    embedding=embedding,
                    created_at=datetime.now(),
                )
                self.db.add(chunk_record)

            await self.db.commit()

            return document

        except Exception as e:
            await self.db.rollback()
            raise RuntimeError(f"Failed to insert chunks: {str(e)}")

    # --------------------------
    # Public interface methods
    # --------------------------
    async def ingest_pdf(
        self,
        document_url: str,
        document_id: UUID,
        chunk_size: int = 750,
        user_id: UUID = None,
    ):
        """
        Complete ingestion pipeline for a PDF:
        1. Load PDF via DoclingLoader
        2. Chunk with HybridChunker
        3. Generate embeddings
        4. Insert into DB
        """
        try:
            tokenizer = get_tokenizer()
            loader = DoclingLoader(
                file_path=document_url,
                export_type=ExportType.DOC_CHUNKS,
                chunker=HybridChunker(tokenizer=tokenizer, chunk_size=chunk_size),
            )
            docs = loader.load()  # list of Document objects
            texts = [doc.page_content for doc in docs]

            chunk_embeddings = await self.embedding_client.aembed_documents(texts)
            document_record = await self._insert_docling_chunks(document_id, docs, chunk_embeddings, user_id)

            print("✅ PDF ingestion complete")
            return document_record

        except Exception as e:
            raise RuntimeError(f"PDF ingestion failed: {str(e)}")

    async def ensure_tables_exist(self):
        """
        Ensure DB tables for Docling chunks exist.
        """
        from app.db.session import engine
        from app.models.base import Base

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
