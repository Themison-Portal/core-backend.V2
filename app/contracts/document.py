from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from .base import BaseContract, TimestampedContract


class DocumentBase(BaseContract):
    document_name: str
    document_type: str
    document_url: str
    trial_id: Optional[UUID] = None
    uploaded_by: Optional[UUID] = None
    status: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    version: Optional[int] = None
    amendment_number: Optional[int] = None
    is_latest: Optional[bool] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    warning: Optional[bool] = None

class DocumentCreate(DocumentBase):
    pass

class DocumentUpdate(BaseContract):
    document_name: Optional[str] = None
    document_type: Optional[str] = None
    document_url: Optional[str] = None
    trial_id: Optional[UUID] = None
    uploaded_by: Optional[UUID] = None
    status: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    version: Optional[int] = None
    amendment_number: Optional[int] = None
    is_latest: Optional[bool] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    warning: Optional[bool] = None

class DocumentResponse(DocumentBase, TimestampedContract):
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    
class DocumentUpload(BaseContract):
    document_url: str
