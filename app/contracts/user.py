"""
This module contains the contracts for the user.
"""

from typing import Optional
from uuid import UUID

from pydantic import EmailStr, Field

from .base import BaseContract, TimestampedContract


class UserBase(BaseContract):
    """
    A base contract for all users.
    """
    email: EmailStr
    name: Optional[str] = None

class UserCreate(UserBase):
    """
    A contract for creating a user.
    """
    password: str = Field(min_length=8, description="User password, minimum 8 characters")

class UserUpdate(BaseContract):
    """
    A contract for updating a user.
    """
    email: Optional[EmailStr] = None
    name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)

class UserResponse(UserBase, TimestampedContract):
    """
    A contract for a user.
    """
    id: UUID
    # Note: password is intentionally excluded from response
