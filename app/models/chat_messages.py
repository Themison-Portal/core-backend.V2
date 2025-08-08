"""
This module contains the chat message model.
"""

import uuid
from datetime import datetime

from sqlalchemy import ARRAY, Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from .base import Base


class ChatMessage(Base):
    """
    A model that represents a chat message.
    """
    __tablename__ = 'chat_messages'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('chat_sessions.id'))
    content = Column(Text)
    role = Column(String(50))  # "user" or "assistant"
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Reference to specific document chunks used
    document_chunk_ids = Column(ARRAY(String))  # IDs of document chunks used
    
    # Relationship to session
    session = relationship("ChatSession", back_populates="messages")