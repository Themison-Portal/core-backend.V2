"""
This module contains the chat document link model.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID

from .base import Base


class ChatDocumentLink(Base):
    """
    A model that links a chat session to a document.
    """
    __tablename__ = 'chat_document_links'
    
    chat_session_id = Column(UUID(as_uuid=True), ForeignKey('chat_sessions.id'), primary_key=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey('trial_documents.id'), primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    usage_count = Column(Integer, default=1)
    first_used_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_used_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
