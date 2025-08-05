import uuid
from datetime import UTC, datetime
from typing import Dict, List

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, relationship

from .base import Base
from .documents import Document


# New table for individual chunks
class DocumentChunk(Base):
    __tablename__ = 'document_chunks'
    
    id: Mapped[UUID] = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[UUID] = Column(UUID(as_uuid=True), ForeignKey('trial_documents.id'), nullable=False)
    content: Mapped[str] = Column(Text, nullable=False)
    chunk_index: Mapped[int] = Column(Integer, nullable=False)
    chunk_metadata: Mapped[Dict] = Column("chunk_metadata", JSON)
    embedding: Mapped[List[float]] = Column(Vector(1536))
    created_at: Mapped[datetime] = Column(DateTime, default=lambda: datetime.now(UTC))
    
    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
