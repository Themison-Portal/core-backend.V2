from typing import List, Optional, TYPE_CHECKING
from uuid import UUID
from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete
from langchain_core.documents import Document
from uuid import uuid4

from app.models.documents import Document as DocumentTable
from app.models.chunks_docling import DocumentChunkDocling
from app.services.doclingRag.interfaces.rag_ingestion_service import IRagIngestionService
from app.core.openai import embedding_client
from docling.chunking import HybridChunker
from langchain_docling.loader import DoclingLoader, ExportType
from app.services.utils.tokenizer import get_tokenizer

if TYPE_CHECKING:
    from app.services.cache.rag_cache_service import RagCacheService
    from app.services.cache.semantic_cache_service import SemanticCacheService


class RagIngestionService(IRagIngestionService):
    """
    Service for PDF ingestion and chunking using Docling + embeddings.
    """

    def __init__(
        self,
        db: AsyncSession,
        cache_service: Optional["RagCacheService"] = None,
        semantic_cache_service: Optional["SemanticCacheService"] = None
    ):
        self.db = db
        self.embedding_client = embedding_client
        self.cache_service = cache_service
        self.semantic_cache_service = semantic_cache_service

    # --------------------------
    # Private helper functions
    # --------------------------
    async def _delete_existing_chunks(self, document_id: UUID) -> int:
        """Delete existing chunks before re-ingestion."""
        stmt = delete(DocumentChunkDocling).where(
            DocumentChunkDocling.document_id == document_id
        )
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.rowcount

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
        1. Invalidate cache for document
        2. Delete existing chunks (for re-ingestion)
        3. Load PDF via DoclingLoader
        4. Chunk with HybridChunker
        5. Generate embeddings
        6. Insert into DB
        """
        try:
            # Invalidate Redis cache before re-ingestion
            if self.cache_service:
                deleted_count = await self.cache_service.invalidate_document(document_id)
                if deleted_count > 0:
                    print(f"Invalidated {deleted_count} Redis cached entries for document {document_id}")

            # Invalidate semantic cache before re-ingestion
            if self.semantic_cache_service:
                deleted_semantic = await self.semantic_cache_service.invalidate_document(document_id)
                if deleted_semantic > 0:
                    print(f"Invalidated {deleted_semantic} semantic cache entries for document {document_id}")

            # Delete existing chunks for re-ingestion
            deleted_chunks = await self._delete_existing_chunks(document_id)
            if deleted_chunks > 0:
                print(f"Deleted {deleted_chunks} existing chunks for document {document_id}")

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

            print("PDF ingestion complete")
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
