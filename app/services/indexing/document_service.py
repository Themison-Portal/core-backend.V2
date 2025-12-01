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
from sqlalchemy import text

from app.contracts.document import DocumentResponse
from app.core.openai import embedding_client
from app.core.storage import StorageProvider
from app.db.session import engine
from app.models.base import Base
from app.models.chunks import DocumentChunk
from app.models.documents import Document as DocumentTable
from app.services.interfaces.document_service import IDocumentService
from app.services.utils.chunking import chunk_text
from app.services.utils.semantic_chunking import chunk_text_semantic
from app.services.utils.preprocessing import preprocess_text

# Import your utils
# from .utils.chunking import chunk_documents


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
                page_text = page.extract_text() or ""
                text_content += page_text + "\n"
            
            if not text_content.strip():
                raise ValueError("No text content found in PDF")
                            
            return text_content
            
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")

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
                # Ensure metadata exists and enrich it with page_start/page_end if possible
                chunk_meta = dict(chunk.metadata or {})
                # Preserve chunk_index
                chunk_meta["chunk_index"] = i

                # Add page_start/page_end derived fields if present in metadata
                page_numbers = chunk_meta.get("page_numbers", [])
                if page_numbers:
                    chunk_meta["page_start"] = page_numbers[0]
                    chunk_meta["page_end"] = page_numbers[-1]

                # Build DB record
                chunk_record = DocumentChunk(
                    id=uuid4(),
                    document_id=document.id,  # Reference existing document
                    content=chunk.page_content,
                    chunk_index=i,
                    chunk_metadata=chunk_meta,
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
            global_offset = 0  # precise char offset across pages
            toc_start = None
            toc_end = None

            # Phase 1: Extract page-by-page with precise character tracking
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text() or ""

                # Skip pages with no meaningful text
                if not page_text.strip():
                    # still advance offset by 1 to keep consistent offsets if needed
                    pages_data.append({
                        "page_number": page_num,
                        "text": "",
                        "start_char": global_offset,
                        "end_char": global_offset
                    })
                    # no char length added for empty page text, but keep a delimiter
                    global_offset += 1
                    continue

                # Detect TOC pages
                page_upper = page_text.upper()
                if toc_start is None and ("TABLE OF CONTENTS" in page_upper or "LIST OF CONTENTS" in page_upper):
                    toc_start = page_num
                    print(f"📋 TOC detected starting at page {page_num}")

                start_char = global_offset
                end_char = start_char + len(page_text)

                pages_data.append({
                    "page_number": page_num,
                    "text": page_text,
                    "start_char": start_char,
                    "end_char": end_char
                })

                # Advance global offset; +1 for the newline separator we use when joining pages
                global_offset = end_char + 1

            if not pages_data:
                raise ValueError("No text content found in PDF")

            # Phase 2: Detect TOC end (first page with substantial non-TOC content)
            if toc_start is not None:
                # Look for first page after TOC start that has section-like content
                for i, page_data in enumerate(pages_data):
                    if page_data["page_number"] <= toc_start:
                        continue

                    page_text = page_data["text"]
                    if not page_text:
                        continue

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
            # Require start_index from semantic chunker to be present
            start_index = chunk.metadata.get("start_index")
            if start_index is None:
                raise ValueError(f"Semantic chunk missing start_index for chunk {chunk_idx} — indexing can't proceed")

            end_index = start_index + len(chunk.page_content)

            # Fast page-matching: include boundary if ranges overlap
            chunk_pages = [
                b["page_number"]
                for b in page_boundaries
                if not (end_index < b["start_char"] or start_index > b["end_char"])
            ]

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

            enhanced_metadata = dict(chunk.metadata or {})
            # ensure we preserve any existing metadata and then add enriched fields
            enhanced_metadata.update({
                "page_numbers": chunk_pages,
                "page_start": chunk_pages[0],
                "page_end": chunk_pages[-1],
                "total_pages_spanned": len(chunk_pages),
                "start_index": start_index,
                "end_index": end_index,
                "is_toc": is_toc,
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

        try:
            # Step 1: Parse PDF with page information and TOC detection
            extraction_result = await self.parse_pdf_with_page_info(document_url, chunk_size)
            content = extraction_result["content"]
            page_boundaries = extraction_result["page_boundaries"]
            toc_page_range = extraction_result["toc_page_range"]

            document_filename = document_url.split("/")[-1]

            # Step 2: Chunk content using semantic chunking
            metadata = {
                "filename": document_filename,
                "content_type": "application/pdf",
                "total_pages": len(page_boundaries),
                "toc_page_range": toc_page_range  # Store TOC range in document metadata
            }

            # Use semantic chunking to respect document structure
            chunks = chunk_text_semantic(
                content,
                metadata,
                chunk_size=1200,  # Optimized for Claude Opus
                chunk_overlap=200
            )

            # Step 3: Add page metadata to chunks with TOC marking
            enhanced_chunks = self.add_page_metadata_to_chunks(chunks, page_boundaries, toc_page_range)

            # Step 4: Preprocess each chunk's content for embedding
            preprocessed_chunks = []
            for chunk in enhanced_chunks:
                preprocessed_content = preprocess_text(chunk.page_content)
                preprocessed_chunk = Document(
                    page_content=preprocessed_content,
                    metadata=chunk.metadata  # Keep all the page metadata including is_toc
                )
                preprocessed_chunks.append(preprocessed_chunk)


            # Step 5: Generate embeddings for each chunk
            texts = [chunk.page_content for chunk in preprocessed_chunks]
            chunk_embeddings = await self.embedding_client.aembed_documents(texts)

            # Step 6: Process existing document and add chunks
            preprocessed_content = preprocess_text(content)
            document_title = document_filename or "Untitled Document"
            result = await self.insert_document_with_chunks(
                title=document_title,
                document_id=document_id,
                content=preprocessed_content,
                chunks=preprocessed_chunks,  # Use enhanced chunks with page info and is_toc
                embeddings=chunk_embeddings,
                metadata=metadata,  # Includes toc_page_range
                user_id=user_id
            )

            print('✅ PDF processing complete')

            return result

        except Exception as e:
            raise RuntimeError(f"PDF processing failed: {str(e)}")
    
    async def ensure_tables_exist(self):
        """
        Create tables if they don't exist and add indexes for faster retrieval.
        """
        try:
            async with engine.begin() as conn:
                # Create tables
                await conn.run_sync(Base.metadata.create_all)

                # Index for quick lookup by document_id
                try:
                    await conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                        ON document_chunks (document_id);
                    """))
                except Exception as e:
                    print(f"⚠️ Failed to create document_id index: {e}")

                # Index for filtering by page_start (stored in chunk_metadata JSON)
                # cast to integer for numeric ordering when appropriate
                try:
                    await conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_chunks_page_start
                        ON document_chunks ( ( (chunk_metadata ->> 'page_start')::int ) );
                    """))
                except Exception as e:
                    # If casting fails on some DBs/rows, fall back to text index
                    print(f"⚠️ Failed to create page_start int index (trying text): {e}")
                    try:
                        await conn.execute(text("""
                            CREATE INDEX IF NOT EXISTS idx_chunks_page_start_text
                            ON document_chunks ( (chunk_metadata ->> 'page_start') );
                        """))
                    except Exception as e2:
                        print(f"⚠️ Failed to create page_start text index: {e2}")

                # Attempt to create a vector index for embeddings using pgvector (guarded)
                # If pgvector or the HNSW operator isn't available this will fail silently
                try:
                    await conn.execute(text("""
                        -- Try to create a vector index (HNSW) for embeddings. This requires pgvector >= 0.4
                        -- and Postgres compiled with the necessary operator classes.
                        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
                        ON document_chunks
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 64);
                    """))
                except Exception as e:
                    # If HNSW is not supported, try ivfflat as fallback (requires pgvector)
                    print(f"⚠️ HNSW index creation failed: {e}. Trying ivfflat fallback...")
                    try:
                        await conn.execute(text("""
                            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat
                            ON document_chunks
                            USING ivfflat (embedding vector_l2_ops)
                            WITH (lists = 100);
                        """))
                    except Exception as e2:
                        print(f"⚠️ ivfflat index creation also failed: {e2}. You may need to create vector index manually depending on pgvector version.")
        except Exception as e:
            raise RuntimeError(f"Failed to create tables or indexes: {str(e)}")
