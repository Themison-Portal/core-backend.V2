"""
This module contains the document model.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    ARRAY,
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ENUM, UUID
from sqlalchemy.orm import Mapped, relationship

from .base import Base


class Document(Base):
    """
    A model that represents a document.
    """
    __tablename__ = 'trial_documents'
    
    id: Mapped[UUID] = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    document_name: Mapped[str] = Column(Text, nullable=False)
    document_type: Mapped[str] = Column(String, nullable=False)
    document_url: Mapped[str] = Column(String, nullable=False)
    trial_id: Mapped[Optional[UUID]] = Column(UUID(as_uuid=True), nullable=True)
    updated_at: Mapped[Optional[datetime]] = Column(DateTime(timezone=True), nullable=True, default=lambda: datetime.now(timezone.utc))
    uploaded_by: Mapped[Optional[UUID]] = Column(UUID(as_uuid=True), nullable=True)
    status: Mapped[Optional[str]] = Column(Text, nullable=True)
    file_size: Mapped[Optional[int]] = Column(BigInteger, nullable=True)
    mime_type: Mapped[Optional[str]] = Column(String(255), nullable=True)
    version: Mapped[Optional[int]] = Column(Integer, nullable=True, default=1)
    amendment_number: Mapped[Optional[int]] = Column(Integer, nullable=True)
    is_latest: Mapped[Optional[bool]] = Column(Boolean, nullable=True, default=True)
    description: Mapped[Optional[str]] = Column(Text, nullable=True)
    tags: Mapped[Optional[List[str]]] = Column(ARRAY(Text), nullable=True)
    warning: Mapped[Optional[bool]] = Column(Boolean, nullable=True)
    
    # Relationships
    chat_sessions: Mapped[List["ChatSession"]] = relationship("ChatSession", secondary="chat_document_links", back_populates="documents")
    # chunks: Mapped[List["DocumentChunk"]] = relationship("DocumentChunk", back_populates="document")
    docling_chunks: Mapped[List["DocumentChunkDocling"]] = relationship("DocumentChunkDocling", back_populates="document")
    
