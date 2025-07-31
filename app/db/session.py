# boilerplate code for the database session

from sqlalchemy.ext.asyncio import (AsyncSession, async_sessionmaker,
                                    create_async_engine)
from sqlalchemy.orm import sessionmaker

from app.config import get_settings

settings = get_settings()
engine = create_async_engine(settings.supabase_db_url)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False) 