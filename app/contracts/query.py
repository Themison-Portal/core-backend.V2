from uuid import UUID

from .base import BaseContract, TimestampedContract


class QueryBase(BaseContract):
    query: str

class QueryCreate(QueryBase):
    pass

class QueryUpdate(QueryBase):
    pass

class QueryResponse(QueryBase, TimestampedContract):
    id: UUID 