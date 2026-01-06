"""
AI client - Using OpenAI for both LLM and embeddings
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from app.config import get_settings
from app.schemas.rag_docling_schema import DoclingRagStructuredResponse

settings = get_settings()

# OpenAI for embeddings
embedding_client = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536,  # Reduce from 3072 to 1536 for Supabase pgvector compatibility
    api_key=settings.openai_api_key
)

# Singleton ChatOpenAI for structured RAG generation
# Pre-configured to avoid per-request instantiation overhead (~100-200ms savings)
_chat_openai = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=2000,  # Limit response size to prevent slow generation
    api_key=settings.openai_api_key,
)

# Pre-bind structured output schema (avoids per-request binding)
structured_llm = _chat_openai.with_structured_output(DoclingRagStructuredResponse)

# LLM for general use (agentic RAG, etc.)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=settings.openai_api_key,
)