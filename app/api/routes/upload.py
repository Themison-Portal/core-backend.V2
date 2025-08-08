"""
Upload routes
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.contracts.document import DocumentResponse
from app.dependencies.auth import get_current_user
from app.dependencies.rag import get_document_service
from app.services.indexing.document_service import DocumentService

router = APIRouter()

class UploadDocumentRequest(BaseModel):
    """
    Upload document request
    """
    document_url: str
    document_id: UUID
    chunk_size: Optional[int] = 1000

@router.post("/upload-pdf", response_model=DocumentResponse)
async def upload_pdf_document(
    request: UploadDocumentRequest,
    user = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Upload a PDF document
    """
    # Validate file type
    if not request.document_url.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Process existing document through RAG pipeline
        result = await document_service.process_pdf_complete(
            document_url=request.document_url,
            document_id=request.document_id,  # Reference existing document
            user_id=user["id"],  # Fixed: user is dict, not object
            chunk_size=request.chunk_size,
        )
        
        return result
        
    except ValueError as e:
        # Document not found or validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Processing or database errors
        raise HTTPException(status_code=500, detail=str(e))
