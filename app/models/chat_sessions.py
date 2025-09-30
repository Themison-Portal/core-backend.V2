"""
This module contains the chat session model.
"""

import uuid
from datetime import datetime, timezone
from typing import List

from sqlalchemy import JSON, Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, relationship

from .base import Base
from .chat_messages import ChatMessage

# from .documents import Document  # Removed to fix circular import

class ChatSession(Base):
    """
    A model that represents a chat session.
    """
    __tablename__ = 'chat_sessions'
    
    id: Mapped[UUID] = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[UUID] = Column(UUID(as_uuid=True))
    title: Mapped[str] = Column(String(255))
    created_at: Mapped[datetime] = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationship to messages
    messages: Mapped[List[ChatMessage]] = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    
    # Many-to-many with documents
    documents: Mapped[list["Document"]] = relationship(
        "Document",
        secondary="chat_document_links",
        back_populates="chat_sessions"
    )