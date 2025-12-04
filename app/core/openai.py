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
    model="claude-opus-4-5-20251101",  # Upgraded from Sonnet 4 to Opus 4.5
    temperature=0,  # Deterministic for maximum consistency
)

# Keep OpenAI for embeddings (faster and more efficient)
embedding_client = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1536,  # Reduce from 3072 to 1536 for Supabase pgvector compatibility
    api_key=settings.openai_api_key
)
