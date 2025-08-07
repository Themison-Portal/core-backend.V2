# boilerplate code for the database session

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (AsyncSession, async_sessionmaker,
                                    create_async_engine)
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.models.base import Base

settings = get_settings()
engine = create_async_engine(
    settings.supabase_db_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=1800,
    connect_args={
        "command_timeout": 60,
        "server_settings": {
            "statement_timeout": "60000",  # 60 seconds
            "idle_in_transaction_session_timeout": "60000"
        }
    }
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def create_tables_if_not_exist():
    """Create tables with better error handling"""
    try:
        async with engine.begin() as conn:
            # Set longer timeout for table creation
            await conn.execute(text("SET statement_timeout = '120000'"))  # 2 minutes
            await conn.run_sync(Base.metadata.create_all, checkfirst=True)
    except Exception as e:
        print(f"Warning: Could not create tables: {e}")
        # Continue without table creation - they might already exist 