from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
# from dotenv import load_dotenv
import os

# Load environment variables
# load_dotenv()

DATABASE_URL = os.getenv("SUPABASE_DB_URL")

# Create SQLAlchemy engine
engine = create_async_engine(DATABASE_URL, echo=False)

# Create session

AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

# Base class for models
Base = declarative_base()
 
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session