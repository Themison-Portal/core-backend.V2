"""
Upload routes with async processing support.
"""

import asyncio
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.contracts.document import AsyncUploadResponse, JobStatusResponse, UploadPdfResponse
from app.config import get_settings
from app.dependencies.rag import get_rag_ingestion_service
from app.dependencies.jobs import get_ingestion_job_service
from app.services.doclingRag.rag_ingestion_service import RagIngestionService
from app.services.jobs.ingestion_job_service import IngestionJobService

logger = logging.getLogger(__name__)

router = APIRouter()


class UploadDocumentRequest(BaseModel):
    """
    Upload document request
    """
    document_url: str
    document_id: UUID
    chunk_size: Optional[int] = 750


async def process_pdf_background(
    document_url: str,
    document_id: UUID,
    rag_service: RagIngestionService,
    job_service: IngestionJobService,
):
    """
    Background task for PDF processing.
    Updates job status at each stage via Redis.
    """
    async def progress_callback(stage: str, chunks_count: Optional[int] = None):
        await job_service.update_stage(document_id, stage, chunks_count)

    try:
        logger.info(f"Starting background ingestion for document {document_id}")

        result = await rag_service.ingest_pdf_with_progress(
            document_url=document_url,
            document_id=document_id,
            progress_callback=progress_callback,
        )

        await job_service.mark_completed(document_id, result["chunks_count"])
        logger.info(f"Background ingestion completed for document {document_id}")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Background ingestion failed for document {document_id}: {error_msg}")
        await job_service.mark_failed(document_id, error_msg)


@router.post("/upload-pdf", response_model=AsyncUploadResponse, status_code=201)
async def upload_pdf_document(
    request: UploadDocumentRequest,
    background_tasks: BackgroundTasks,
    rag_service: RagIngestionService = Depends(get_rag_ingestion_service),
    job_service: IngestionJobService = Depends(get_ingestion_job_service),
    x_api_key: str = Header(...),
):
    """
    Upload a PDF document for async processing.

    Returns 201 immediately and processes the document in the background.
    Use GET /upload/status/{document_id} to poll for completion.
    """
    settings = get_settings()
    if not settings.upload_api_key or x_api_key != settings.upload_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Validate file type
    if not request.document_url.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Create job entry in Redis
        await job_service.create_job(request.document_id)

        # Schedule background processing
        background_tasks.add_task(
            process_pdf_background,
            document_url=request.document_url,
            document_id=request.document_id,
            rag_service=rag_service,
            job_service=job_service,
        )

        logger.info(f"Scheduled background ingestion for document {request.document_id}")

        return AsyncUploadResponse(
            document_id=request.document_id,
            status="pending",
            message="PDF processing started. Poll /upload/status/{document_id} for progress.",
        )

    except Exception as e:
        logger.error(f"Failed to schedule ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")


@router.get("/status/{document_id}", response_model=JobStatusResponse)
async def get_upload_status(
    document_id: UUID,
    job_service: IngestionJobService = Depends(get_ingestion_job_service),
    x_api_key: str = Header(...),
):
    """
    Get the status of a PDF ingestion job.

    Poll this endpoint to track progress of async PDF processing.
    """
    settings = get_settings()
    if not settings.upload_api_key or x_api_key != settings.upload_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    job = await job_service.get_job(document_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"No ingestion job found for document {document_id}. Job may have expired or never existed.",
        )

    return JobStatusResponse(
        document_id=UUID(job["document_id"]),
        status=job["status"],
        stage=job["stage"],
        progress=job["progress"],
        chunks_count=job["chunks_count"],
        error=job["error"],
        started_at=job["started_at"],
        updated_at=job["updated_at"],
        completed_at=job["completed_at"],
    )


# Legacy synchronous endpoint (kept for backwards compatibility)
@router.post("/upload-pdf-sync", response_model=UploadPdfResponse)
async def upload_pdf_document_sync(
    request: UploadDocumentRequest,
    rag_service: RagIngestionService = Depends(get_rag_ingestion_service),
    x_api_key: str = Header(...),
):
    """
    Upload a PDF document (synchronous processing).

    This endpoint blocks until processing completes.
    Use /upload-pdf for async processing with progress tracking.
    """
    settings = get_settings()
    if not settings.upload_api_key or x_api_key != settings.upload_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not request.document_url.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        logger.info(f"Processing document ID: {request.document_id} (sync)")
        result = await rag_service.ingest_pdf(
            document_url=request.document_url,
            document_id=request.document_id,
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
