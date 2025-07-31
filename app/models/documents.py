import uuid
from datetime import UTC, datetime
# from .chat_sessions import ChatSession  # Removed to fix circular import
from typing import List

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, relationship

from .base import Base


class Document(Base):
    __tablename__ = 'documents'
    
    id: Mapped[UUID] = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[UUID] = Column(UUID(as_uuid=True), nullable=False)  # Owner
    original_filename: Mapped[str] = Column(String(255), nullable=False)
    storage_url: Mapped[str] = Column(Text, nullable=False)
    file_size: Mapped[int] = Column(Integer)
    processing_status: Mapped[str] = Column(String(50), default="pending")
    doc_metadata: Mapped[JSON] = Column("metadata", JSON)
    chunks: Mapped[JSON] = Column(JSON)
    content: Mapped[Text] = Column(Text)
    total_pages: Mapped[int] = Column(Integer)
    total_chunks: Mapped[int] = Column(Integer)
    created_at: Mapped[DateTime] = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[DateTime] = Column(DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))
    chat_sessions: Mapped[list["ChatSession"]] = relationship("ChatSession", secondary="chat_document_links", back_populates="documents")
    
    # Relationship to document chunks
    chunks: Mapped[list["DocumentChunk"]] = relationship("DocumentChunk", back_populates="document")
    
