import uuid
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, relationship

from .base import Base


class Query(Base):
    __tablename__ = 'queries'
    
    id: Mapped[UUID] = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query: Mapped[str] = Column(String)
    embeddings: Mapped[List[float]] = Column(Vector(1536))
    created_at: Mapped[datetime] = Column(DateTime, default=lambda: datetime.now(UTC))