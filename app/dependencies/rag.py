from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.dependencies.db import get_db
from app.services.doclingRag.rag_ingestion_service import RagIngestionService

async def get_rag_ingestion_service(db: AsyncSession = Depends(get_db)):
    return RagIngestionService(db)