"""
This module contains the user model.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import Column, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped

from .base import Base


class UserRole(PyEnum):
    ADMIN = "admin"
    USER = "user"

class User(Base):
    """
    A model that represents a user.
    """
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True)
    password = Column(UUID(as_uuid=True))
    role: Mapped[UserRole] = Column(SQLEnum(UserRole), default=UserRole.USER)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
