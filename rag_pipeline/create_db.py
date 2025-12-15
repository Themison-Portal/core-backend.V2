import asyncio
from database import engine, Base
from models.vector_model_dockling import Protocol, ProtocolChunk, TrialProtocol

async def create_tables():
    async with engine.begin() as conn:
        # Run the synchronous create_all on the async connection
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created successfully!")

if __name__ == "__main__":
    asyncio.run(create_tables())
