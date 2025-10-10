"""
AI client - Using Anthropic Claude for LLM and OpenAI for embeddings
"""

from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings

settings = get_settings()

# Use Anthropic Claude for LLM (better citations)
llm = ChatAnthropic(
    api_key=settings.anthropic_api_key,
    model="claude-sonnet-4-20250514",  # Upgraded from Haiku to Sonnet 4
    temperature=0.1,  # Lower temp for consistent citation format
)

# Keep OpenAI for embeddings (faster and more efficient)
embedding_client = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=settings.openai_api_key
)