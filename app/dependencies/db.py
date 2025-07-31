from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session, engine
from app.models.base import Base


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session"""
    async with engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        db = async_session()
        try:
            yield db
        finally:
            await db.close() 