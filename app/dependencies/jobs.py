"""
Job service dependency injection.
"""

from fastapi import Depends

from app.dependencies.redis_client import get_redis_client
from app.services.jobs.ingestion_job_service import IngestionJobService


def get_ingestion_job_service(
    redis=Depends(get_redis_client),
) -> IngestionJobService:
    """Provide IngestionJobService instance."""
    return IngestionJobService(redis)
