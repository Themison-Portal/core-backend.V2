"""
AI client - Using OpenAI for both LLM and embeddings

Supports configurable embedding models:
- text-embedding-3-small (1536 dims) - Default, cost-effective
- text-embedding-3-large (3072 dims) - Higher quality, 6.5x cost
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from app.config import get_settings
from app.schemas.rag_docling_schema import DoclingRagStructuredResponse

settings = get_settings()

# OpenAI for embeddings - configurable model and dimensions
embedding_client = OpenAIEmbeddings(
    model=settings.embedding_model,
    dimensions=settings.embedding_dimensions,
    api_key=settings.openai_api_key
)

# For migration period: separate clients for small and large embeddings
# Use these when backfilling or during dual-write migration
embedding_client_small = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536,
    api_key=settings.openai_api_key
)

embedding_client_large = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=2000,  # Reduced from 3072 due to HNSW index limit
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