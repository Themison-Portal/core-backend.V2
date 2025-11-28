"""
This module contains the contracts for the query.
"""
from uuid import UUID

from .base import BaseContract, TimestampedContract


class QueryBase(BaseContract):
    """
    A base contract for all queries.
    """
    query: str

class QueryCreate(QueryBase):
    """
    A contract for creating a query.
    """
    pass

class QueryUpdate(QueryBase):
    """
    A contract for updating a query.
    """
    pass

class QueryResponse(QueryBase, TimestampedContract):
    """
    A contract for a query.
    """
    id: UUID 