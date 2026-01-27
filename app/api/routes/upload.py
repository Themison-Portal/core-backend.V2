"""
Upload routes
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from app.contracts.document import UploadPdfResponse
from app.config import get_settings
from app.dependencies.rag import get_rag_ingestion_service
from app.services.doclingRag.rag_ingestion_service import RagIngestionService

router = APIRouter()

class UploadDocumentRequest(BaseModel):
    """
    Upload document request
    """
    document_url: str
    document_id: UUID
    chunk_size: Optional[int] = 750

@router.post("/upload-pdf", response_model=UploadPdfResponse)
async def upload_pdf_document(
    request: UploadDocumentRequest,
    rag_service: RagIngestionService = Depends(get_rag_ingestion_service),
    x_api_key: str = Header(...),
):
    """
    Upload a PDF document
    """
    settings = get_settings()
    if not settings.upload_api_key or x_api_key != settings.upload_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Validate file type
    if not request.document_url.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Process existing document through RAG pipeline
        # IMPORTANT: Always use 750 for optimal balance between quality and cost
        print(f"Processing document ID: {request.document_id}")
        result = await rag_service.ingest_pdf(
            document_url=request.document_url,
            document_id=request.document_id,
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

# @router.post("/upload-pdf", response_model=DocumentResponse)
# async def upload_pdf_document(
#     request: UploadDocumentRequest,
#     user = Depends(get_current_user),
#     document_service: DocumentService = Depends(get_document_service)
# ):
#     """
#     Upload a PDF document
#     """
#     # Validate file type
#     if not request.document_url.endswith('.pdf'):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
#     try:
#         # Process existing document through RAG pipeline
#         # IMPORTANT: Always use 750 for optimal balance between quality and cost
#         print(f"Processing document ID: {request.document_id}")
#         result = await document_service.process_pdf_complete(
#             document_url=request.document_url,
#             document_id=request.document_id,
#             user_id=user["id"],
#             chunk_size=750,  # Fixed at 750 for consistent chunking strategy
#         )
        
#         return result
        
#     except ValueError as e:
#         # Document not found or validation errors
#         print(f"Validation error: {str(e)}")
#         raise HTTPException(status_code=400, detail=str(e))
#     except RuntimeError as e:
#         # Processing or database errors
#         print(f"Runtime error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#     except Exception as e:
#         # Catch-all for unexpected errors
#         print(f"Unexpected error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
