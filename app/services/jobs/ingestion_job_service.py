"""
Redis-based job tracking service for PDF ingestion operations.
Follows the RagCacheService pattern for Redis interaction.
"""

import json
from datetime import datetime
from typing import Optional
from uuid import UUID

from redis.asyncio import Redis


class IngestionJobService:
    """
    Tracks ingestion job status in Redis.

    Key pattern: job:ingest:{document_id}
    TTL: 3600 seconds (1 hour)

    Status values: pending | processing | completed | failed
    Stage values: starting | cache_invalidation | parsing | chunking | embedding | storing | done
    """

    TTL_JOB = 3600  # 1 hour
    PREFIX_JOB = "job:ingest"

    # Stage progress percentages
    STAGE_PROGRESS = {
        "starting": 0,
        "cache_invalidation": 10,
        "parsing": 20,
        "chunking": 40,
        "embedding": 60,
        "storing": 80,
        "done": 100,
    }

    def __init__(self, redis: Redis):
        self.redis = redis

    def _job_key(self, document_id: UUID) -> str:
        """Generate Redis key for job."""
        return f"{self.PREFIX_JOB}:{document_id}"

    async def create_job(self, document_id: UUID) -> dict:
        """
        Create a new ingestion job entry.

        Returns the initial job state.
        """
        now = datetime.utcnow().isoformat()
        job_data = {
            "document_id": str(document_id),
            "status": "pending",
            "stage": "starting",
            "progress": 0,
            "chunks_count": None,
            "error": None,
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
        }

        await self.redis.set(
            self._job_key(document_id),
            json.dumps(job_data),
            ex=self.TTL_JOB,
        )

        return job_data

    async def get_job(self, document_id: UUID) -> Optional[dict]:
        """
        Retrieve job status for a document.

        Returns None if job doesn't exist or has expired.
        """
        key = self._job_key(document_id)
        cached = await self.redis.get(key)

        if cached:
            data = cached if isinstance(cached, str) else cached.decode()
            return json.loads(data)

        return None

    async def update_stage(
        self,
        document_id: UUID,
        stage: str,
        chunks_count: Optional[int] = None,
    ) -> None:
        """
        Update job to a new stage.

        Automatically calculates progress percentage based on stage.
        """
        job = await self.get_job(document_id)
        if not job:
            return

        job["stage"] = stage
        job["status"] = "processing"
        job["progress"] = self.STAGE_PROGRESS.get(stage, job["progress"])
        job["updated_at"] = datetime.utcnow().isoformat()

        if chunks_count is not None:
            job["chunks_count"] = chunks_count

        await self.redis.set(
            self._job_key(document_id),
            json.dumps(job),
            ex=self.TTL_JOB,
        )

    async def mark_completed(
        self,
        document_id: UUID,
        chunks_count: int,
    ) -> None:
        """Mark job as successfully completed."""
        job = await self.get_job(document_id)
        if not job:
            return

        now = datetime.utcnow().isoformat()
        job["status"] = "completed"
        job["stage"] = "done"
        job["progress"] = 100
        job["chunks_count"] = chunks_count
        job["updated_at"] = now
        job["completed_at"] = now

        await self.redis.set(
            self._job_key(document_id),
            json.dumps(job),
            ex=self.TTL_JOB,
        )

    async def mark_failed(
        self,
        document_id: UUID,
        error: str,
    ) -> None:
        """Mark job as failed with error message."""
        job = await self.get_job(document_id)
        if not job:
            # Create minimal job record for failed jobs
            job = {
                "document_id": str(document_id),
                "status": "failed",
                "stage": "failed",
                "progress": 0,
                "chunks_count": None,
                "error": error,
                "started_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "completed_at": None,
            }
        else:
            job["status"] = "failed"
            job["error"] = error
            job["updated_at"] = datetime.utcnow().isoformat()

        await self.redis.set(
            self._job_key(document_id),
            json.dumps(job),
            ex=self.TTL_JOB,
        )

    async def delete_job(self, document_id: UUID) -> bool:
        """Delete a job entry. Returns True if deleted."""
        result = await self.redis.delete(self._job_key(document_id))
        return result > 0
