"""
Upload routes
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.contracts.document import DocumentResponse
from app.dependencies.auth import get_current_user
from app.dependencies.documents import get_document_service
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
        print(f"Processing document ID: {request.document_id}")
        result = await document_service.process_pdf_complete(
            document_url=request.document_url,
            document_id=request.document_id,
            user_id=user["id"],
            chunk_size=request.chunk_size,
        )
        
        return result
        
    except ValueError as e:
        # Document not found or validation errors
        print(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Processing or database errors
        print(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Catch-all for unexpected errors
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
