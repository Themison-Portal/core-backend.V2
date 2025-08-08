from langchain_openai import ChatOpenAI, AsyncOpenAI

client = ChatOpenAI(model="gpt-4o-mini")

async_client = AsyncOpenAI(model="gpt-4o-mini")
