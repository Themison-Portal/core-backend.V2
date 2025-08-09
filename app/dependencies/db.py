"""
This module contains the database dependencies.
"""
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database session.
    """

    db = async_session()
    try:
        yield db
    finally:
        await db.close()
        