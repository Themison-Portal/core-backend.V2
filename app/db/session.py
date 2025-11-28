"""
This module contains the database session.
"""
import sys
import logging
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.config import get_settings

try:
    settings = get_settings()
except Exception as e:
    print("Failed to load settings:", e, file=sys.stderr)
    logging.error("Failed to load settings:", exc_info=e)
    settings = None

"""
Create an async engine for the database.
"""
if settings:
    try:
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
    except Exception as e:
        print("Failed to create database engine:", e, file=sys.stderr)
        logging.error("Failed to create database engine:", exc_info=e)
