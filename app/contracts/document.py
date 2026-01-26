"""
This module contains the contracts for the document.
"""
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from .base import BaseContract, TimestampedContract


class DocumentBase(BaseContract):
    """
    A base contract for all documents.
    """
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
    """
    A contract for creating a document.
    """
    pass

class DocumentUpdate(BaseContract):
    """
    A contract for updating a document.
    """
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
    """
    A contract for a document.
    """
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {
        "from_attributes": True
    }
    
class UploadPdfResponse(BaseContract):
    """
    Response contract for the upload-pdf endpoint.
    """
    success: bool
    document_id: UUID
    status: str
    chunks_count: int
    created_at: datetime

class DocumentUpload(BaseContract):
    """
    A contract for uploading a document.
    """
    document_url: str
