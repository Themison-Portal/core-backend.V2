from typing import List
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from langchain_core.documents import Document

from app.services.doclingRag.interfaces.rag_retrieval_service import IRagRetrievalService
from app.services.utils.threading import run_in_thread  # your helper for async threading


class RagRetrievalService(IRagRetrievalService):
    """
    Service for retrieving similar Docling chunks from the database.
    """

    def __init__(self, db: AsyncSession, embedding_client):
        self.db = db
        self.embedding_client = embedding_client        

    # --------------------------
    # Private helpers
    # --------------------------
    def _embedding_to_pg_vector(self, emb: List[float]) -> str:
        """
        Convert Python list of floats to PostgreSQL vector format.
        """
        return "[" + ",".join(str(x) for x in emb) + "]"

    async def _search_similar_chunks_docling(
        self,
        query_text: str,
        document_id: UUID,
        top_k: int = 20,
    ) -> List[dict]:
        """
        Retrieve top-k similar chunks from `document_chunks_docling` using embeddings.
        """
        # Compute query embedding
        query_vector = await run_in_thread(self.embedding_client.embed_query, query_text)
        query_vector = self._embedding_to_pg_vector(query_vector)

        # Query database
        sql = text("""
            SELECT
                pc.content,
                pc.page_number,
                pc.chunk_metadata,
                p.document_name,
                1 - (pc.embedding <=> (:v)::vector) AS similarity
            FROM document_chunks_docling pc
            JOIN trial_documents p ON pc.document_id = p.id
            WHERE pc.document_id = :pid
            ORDER BY pc.embedding <=> (:v)::vector
            LIMIT :k
        """)

        async with self.db.begin():
            result = await self.db.execute(sql, {"v": query_vector, "k": top_k, "pid": document_id})
            rows = result.fetchall()

        # Format results
        docs = [
            {
                "page_content": row.content,
                "score": float(row.similarity),
                "metadata": {
                    "title": row.document_name,
                    "page": row.page_number,
                    "docling": row.chunk_metadata,
                },
            }
            for row in rows
        ]

        return docs    

    # --------------------------
    # Public interface
    # --------------------------
    async def retrieve_similar_chunks(
        self,
        query_text: str,
        document_id: UUID,
        top_k: int = 20,
        min_score: float = 0.04
    ) -> List[dict]:
        """
        Public method to retrieve and format top similar chunks for a query.
        """
        raw_chunks = await self._search_similar_chunks_docling(query_text, document_id, top_k)

        # Filter by relevance
        filtered_chunks = [d for d in raw_chunks if d["score"] >= min_score]        

        return filtered_chunks
