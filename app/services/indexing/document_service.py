"""
This module contains the document service.
"""
import io
from datetime import datetime
from typing import Any, Dict, List
from uuid import UUID, uuid4

import pypdf as PyPDF2
import requests
from langchain_core.documents import Document
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from app.contracts.document import DocumentResponse
from app.core.openai import embedding_client
from app.core.storage import StorageProvider
from app.db.session import engine
from app.models.base import Base
from app.models.chunks import DocumentChunk
from app.models.chunks_docling import DocumentChunkDocling
from app.models.documents import Document as DocumentTable
from app.services.interfaces.document_service import IDocumentService
from app.services.utils.chunking import chunk_text
from app.services.utils.semantic_chunking import chunk_text_semantic
from app.services.utils.preprocessing import preprocess_text

from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from transformers import AutoTokenizer
from langchain_docling.loader import ExportType

# Import your utils
# from .utils.chunking import chunk_documents

LLM_MODEL_NAME = "gpt-4o-mini"

TOKENIZER_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2" 
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)

class DocumentService(IDocumentService):
    """
    A service that handles document indexing and chunking.
    """
    def __init__(
        self,
        db: AsyncSession,
        storage_provider: StorageProvider
    ):
        self.db = db
        self.storage_provider = storage_provider
        self.embedding_client = embedding_client
    
    async def parse_pdf(self, document_url: str) -> str:
        """
        Extract text content from PDF file.
        """
        try:
            # Read PDF content
            response = requests.get(document_url, timeout=10)
            content = response.content
            pdf_file = io.BytesIO(content)
            
            # Extract text using PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = ""
            
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            if not text_content.strip():
                raise ValueError("No text content found in PDF")
                            
            return text_content
            
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")


    def extract_docling_citation_metadata(self, metadata_json):
        """
        Returns a dict with page_number and headings for a chunk.
        """
        try:
            dl_meta = metadata_json.get("dl_meta", {})
            doc_items = dl_meta.get("doc_items", [])
            headings = dl_meta.get("headings", [])

            # Docling provides provenance info for each doc_item
            page_number = None
            if doc_items:
                prov_list = doc_items[0].get("prov", [])
                if prov_list:
                    page_number = prov_list[0].get("page_no")

            return {
                "page_number": page_number,
                "headings": headings or []
            }

        except Exception:
            return {
                "page_number": None,
                "headings": []
            }
        
    async def insert_docling_chunks(
        self,
        document_id: UUID,
        chunks: List[Document],
        embeddings: List[List[float]],
        user_id: UUID = None,
    ) -> DocumentResponse:
        """
        Process Docling chunks and add them to `document_chunks_docling` table with embeddings.
        Mirrors the company's insert_document_with_chunks pattern.
        """
        await self.ensure_tables_exist()  # Ensure tables exist first

        try:
            # Get the existing document (trial_documents)
            document = await self.db.get(DocumentTable, document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found. Frontend should create it first.")

            # Add chunks
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                citation_meta = self.extract_docling_citation_metadata(chunk.metadata)
                page_number = citation_meta["page_number"]
                chunk_record = DocumentChunkDocling(
                    id=uuid4(),
                    document_id=document.id,  # Reference the existing document
                    content=chunk.page_content,
                    page_number=page_number,
                    chunk_metadata={**chunk.metadata, "chunk_index": i},
                    embedding=embedding,
                    created_at=datetime.now(),
                )
                self.db.add(chunk_record)

            await self.db.commit()
            
            # return DocumentResponse.model_validate(document)
            return

        except ValueError as e:
            await self.db.rollback()
            raise e
        except IntegrityError as e:
            await self.db.rollback()
            raise ValueError(f"Database integrity error: {str(e)}")
        except Exception as e:
            await self.db.rollback()
            raise RuntimeError(f"Failed to process document Docling chunks: {str(e)}")
                

    async def insert_document_with_chunks(
        self,
        title: str, 
        document_id: UUID,  # Existing document ID from frontend
        content: str,
        chunks: List[Document],
        embeddings: List[List[float]],
        metadata: Dict[str, Any] = None,
        user_id: UUID = None
    ) -> DocumentResponse:
        """
        Process existing document and add chunks with embeddings.
        """
        
        await self.ensure_tables_exist()
        
        try:
            # Find existing document that frontend already created
            document = await self.db.get(DocumentTable, document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found. Frontend should create it first.")
                        
            # Document already exists, no need to update non-existent columns
            # Just proceed to add chunks
                                    
            # Add chunks that reference the existing document
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

                chunk_record = DocumentChunk(
                    id=uuid4(),
                    document_id=document.id,  # Reference existing document
                    content=chunk.page_content,
                    chunk_index=i,
                    chunk_metadata={**chunk.metadata, "chunk_index": i},
                    embedding=embedding,
                    created_at=datetime.now()
                )
                self.db.add(chunk_record)  # Add NEW chunk
            
            await self.db.commit()
            await self.db.refresh(document)
            
            return DocumentResponse.model_validate(document)
        
        except ValueError as e:
            await self.db.rollback()
            raise e  # Re-raise ValueError as-is
        except IntegrityError as e:
            await self.db.rollback()
            raise ValueError(f"Database integrity error: {str(e)}")
        except Exception as e:
            await self.db.rollback()
            raise RuntimeError(f"Failed to process document chunks: {str(e)}")
    
    async def parse_pdf_with_page_info(self, document_url: str, chunk_size: int = 750) -> Dict[str, Any]:
        """
        Extract text content from PDF file with precise page boundaries and TOC detection.
        Returns: {
            "content": str,
            "page_boundaries": List[Dict],
            "toc_page_range": Dict[str, int] | None
        }
        """
        try:
            # Read PDF content
            response = requests.get(document_url, timeout=10)
            content = response.content
            pdf_file = io.BytesIO(content)

            # Extract text using PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pages_data = []
            char_offset = 0
            toc_start = None
            toc_end = None

            # Phase 1: Extract page-by-page with precise character tracking
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()

                if not page_text.strip():
                    continue

                # Detect TOC pages
                page_upper = page_text.upper()
                if toc_start is None and ("TABLE OF CONTENTS" in page_upper or "LIST OF CONTENTS" in page_upper):
                    toc_start = page_num
                    print(f"📋 TOC detected starting at page {page_num}")

                pages_data.append({
                    "page_number": page_num,
                    "text": page_text,
                    "start_char": char_offset,
                    "end_char": char_offset + len(page_text)
                })

                char_offset += len(page_text) + 1  # +1 for newline between pages

            if not pages_data:
                raise ValueError("No text content found in PDF")

            # Phase 2: Detect TOC end (first page with substantial non-TOC content)
            if toc_start is not None:
                # Look for first page after TOC start that has section-like content
                for i, page_data in enumerate(pages_data):
                    if page_data["page_number"] <= toc_start:
                        continue

                    page_text = page_data["text"]

                    # Heuristic: TOC ends when we find a page with:
                    # - Paragraphs (multiple sentences, not just "Title .... XX")
                    # - Low density of dots and numbers
                    lines = page_text.split('\n')
                    toc_like_lines = 0
                    content_like_lines = 0

                    for line in lines:
                        line_stripped = line.strip()
                        if not line_stripped:
                            continue

                        # TOC-like: ends with dots and numbers, e.g., "Section .... 45"
                        if line_stripped.count('.') >= 3 and line_stripped[-1].isdigit():
                            toc_like_lines += 1
                        # Content-like: longer lines without TOC pattern
                        elif len(line_stripped) > 50:
                            content_like_lines += 1

                    # If this page has more content than TOC patterns, TOC likely ended
                    if content_like_lines > toc_like_lines and content_like_lines >= 3:
                        toc_end = page_data["page_number"] - 1
                        print(f"📋 TOC detected ending at page {toc_end} (content starts at {page_data['page_number']})")
                        break

                # If we never found end, assume TOC is just a few pages
                if toc_end is None:
                    toc_end = min(toc_start + 5, pages_data[-1]["page_number"])
                    print(f"📋 TOC end estimated at page {toc_end}")

            # Build full content and page boundaries
            full_content = "\n".join([p["text"] for p in pages_data])
            page_boundaries = [
                {
                    "page_number": p["page_number"],
                    "start_char": p["start_char"],
                    "end_char": p["end_char"]
                }
                for p in pages_data
            ]

            # Build TOC range
            toc_page_range = None
            if toc_start and toc_end:
                toc_page_range = {"start": toc_start, "end": toc_end}
                print(f"✅ TOC range: pages {toc_start}-{toc_end}")

            return {
                "content": full_content,
                "page_boundaries": page_boundaries,
                "toc_page_range": toc_page_range
            }

        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")

    def add_page_metadata_to_chunks(
        self,
        chunks: List[Document],
        page_boundaries: List[Dict[str, Any]],
        toc_page_range: Dict[str, int] = None
    ) -> List[Document]:
        """
        Add page number metadata to chunks using start_index directly.
        Mark TOC chunks based on page range.
        """
        enhanced_chunks = []

        for chunk_idx, chunk in enumerate(chunks):
            # Use start_index from semantic chunker directly
            start_index = chunk.metadata.get("start_index", 0)
            end_index = start_index + len(chunk.page_content)

            # Find all pages that intersect with this chunk's character range
            chunk_pages = []
            for boundary in page_boundaries:
                if start_index < boundary["end_char"] and end_index > boundary["start_char"]:
                    chunk_pages.append(boundary["page_number"])

            # Fallback to first page if no intersection found
            if not chunk_pages:
                chunk_pages = [1]

            # Detect if chunk is in TOC range
            is_toc = False
            if toc_page_range and chunk_pages:
                is_toc = any(
                    toc_page_range["start"] <= p <= toc_page_range["end"]
                    for p in chunk_pages
                )

            # Debug logging
            toc_marker = "📋 TOC" if is_toc else ""
            print(f"📄 Chunk {chunk_idx}: pages {chunk_pages} (char {start_index}-{end_index}) {toc_marker}")

            enhanced_metadata = chunk.metadata.copy()
            enhanced_metadata.update({
                "page_numbers": chunk_pages,
                "total_pages_spanned": len(chunk_pages),
                "is_toc": is_toc,
                "start_index": start_index,
                "end_index": end_index
            })

            enhanced_chunk = Document(
                page_content=chunk.page_content,
                metadata=enhanced_metadata
            )
            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks


    async def process_pdf_complete(
        self,
        document_url: str,
        document_id: UUID,
        user_id: UUID = None,
        chunk_size: int = 750,
    ) -> DocumentResponse:
        """
        Complete PDF processing pipeline for existing document with page tracking.
        """
        print("****************************************")
        print("\n Ingesion starts here \n")
        print("****************************************")
        try:
            

            # 1. Load and chunk with Docling + HybridChunker
            loader = DoclingLoader(
                file_path=document_url,
                export_type=ExportType.DOC_CHUNKS,
                chunker=HybridChunker(tokenizer=tokenizer, chunk_size=chunk_size),
            )
            docs = loader.load()  # list of Document objects

            texts = [doc.page_content for doc in docs]

            # 2. generate embeddings
            chunk_embeddings = await self.embedding_client.aembed_documents(texts)

            await self.insert_docling_chunks(document_id, docs, chunk_embeddings)                        

            print('✅ PDF processing complete')

            

        except Exception as e:
            raise RuntimeError(f"PDF processing failed: {str(e)}")
        
    
        
    # async def process_pdf_complete(
    #     self,
    #     document_url: str,
    #     document_id: UUID,
    #     user_id: UUID = None,
    #     chunk_size: int = 750,
    # ) -> DocumentResponse:
    #     """
    #     Complete PDF processing pipeline for existing document with page tracking.
    #     """

    #     try:
    #         # Step 1: Parse PDF with page information and TOC detection
    #         extraction_result = await self.parse_pdf_with_page_info(document_url, chunk_size)
    #         content = extraction_result["content"]
    #         page_boundaries = extraction_result["page_boundaries"]
    #         toc_page_range = extraction_result["toc_page_range"]

    #         document_filename = document_url.split("/")[-1]

    #         # Step 2: Chunk content using semantic chunking
    #         metadata = {
    #             "filename": document_filename,
    #             "content_type": "application/pdf",
    #             "total_pages": len(page_boundaries),
    #             "toc_page_range": toc_page_range  # Store TOC range in document metadata
    #         }

    #         # Use semantic chunking to respect document structure
    #         chunks = chunk_text_semantic(
    #             content,
    #             metadata,
    #             chunk_size=1500,  # Larger to fit complete sections
    #             chunk_overlap=200
    #         )

    #         # Step 3: Add page metadata to chunks with TOC marking
    #         enhanced_chunks = self.add_page_metadata_to_chunks(chunks, page_boundaries, toc_page_range)

    #         # Step 4: Preprocess each chunk's content for embedding
    #         preprocessed_chunks = []
    #         for chunk in enhanced_chunks:
    #             preprocessed_content = preprocess_text(chunk.page_content)
    #             preprocessed_chunk = Document(
    #                 page_content=preprocessed_content,
    #                 metadata=chunk.metadata  # Keep all the page metadata including is_toc
    #             )
    #             preprocessed_chunks.append(preprocessed_chunk)


    #         # Step 5: Generate embeddings for each chunk
    #         texts = [chunk.page_content for chunk in preprocessed_chunks]
    #         chunk_embeddings = await self.embedding_client.aembed_documents(texts)

    #         # Step 6: Process existing document and add chunks
    #         preprocessed_content = preprocess_text(content)
    #         document_title = document_filename or "Untitled Document"
    #         result = await self.insert_document_with_chunks(
    #             title=document_title,
    #             document_id=document_id,
    #             content=preprocessed_content,
    #             chunks=preprocessed_chunks,  # Use enhanced chunks with page info and is_toc
    #             embeddings=chunk_embeddings,
    #             metadata=metadata,  # Includes toc_page_range
    #             user_id=user_id
    #         )

    #         print('✅ PDF processing complete')

    #         return result

    #     except Exception as e:
    #         raise RuntimeError(f"PDF processing failed: {str(e)}")
    
    async def ensure_tables_exist(self):
        """
        Create tables if they don't exist.
        """
        try:
            async with engine.begin() as conn:
                # Drop and recreate document_chunks table to fix the Vector dimension
                await conn.run_sync(Base.metadata.create_all)
        except Exception as e:
            raise RuntimeError(f"Failed to create tables: {str(e)}")