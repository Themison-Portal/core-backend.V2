import asyncio
from rag_pipeline.query_data_store_biobert import rag_query_biobert
from dotenv import load_dotenv, find_dotenv

load_dotenv()

async def main():
    result = await rag_query_biobert("what is 2+2")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())