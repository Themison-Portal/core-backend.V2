"""
OpenAI client
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import get_settings

settings = get_settings()

llm = ChatOpenAI(
    api_key=settings.openai_api_key,
    model="gpt-4.1-mini",
    streaming=True,
    temperature=0.5,
)

embedding_client = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=settings.openai_api_key
)