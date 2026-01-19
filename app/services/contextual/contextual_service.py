"""
Contextual Retrieval Service

Implements Anthropic's contextual retrieval approach:
https://www.anthropic.com/news/contextual-retrieval

For each chunk, generates a contextual summary that situates it within
the overall document. This summary is prepended to the chunk before
embedding, improving retrieval accuracy by 20-35%.
"""

import logging
from typing import List, Optional

from anthropic import AsyncAnthropic

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize Anthropic client
_anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

# Prompt template for generating contextual summaries
# Based on Anthropic's recommended approach
CONTEXT_PROMPT = """<document>
{document_context}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


class ContextualService:
    """
    Service for generating contextual summaries for document chunks.

    The contextual summary helps embeddings capture the chunk's meaning
    within the broader document context, improving retrieval accuracy.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",  # Cost-effective for summaries
        max_tokens: int = 100
    ):
        self.model = model
        self.max_tokens = max_tokens

    async def generate_contextual_summary(
        self,
        chunk_content: str,
        document_context: str,
    ) -> str:
        """
        Generate a contextual summary for a chunk.

        Args:
            chunk_content: The text content of the chunk
            document_context: Surrounding chunks or document summary for context

        Returns:
            A 2-3 sentence contextual summary
        """
        try:
            response = await _anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{
                    "role": "user",
                    "content": CONTEXT_PROMPT.format(
                        document_context=document_context,
                        chunk_content=chunk_content
                    )
                }]
            )
            summary = response.content[0].text.strip()
            logger.debug(f"Generated contextual summary: {summary[:100]}...")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate contextual summary: {e}")
            # Return empty string on failure - chunk will be embedded without context
            return ""

    def create_contextualized_chunk(
        self,
        original_content: str,
        contextual_summary: str
    ) -> str:
        """
        Prepend contextual summary to chunk content.

        Args:
            original_content: The original chunk text
            contextual_summary: The generated contextual summary

        Returns:
            Contextualized chunk text ready for embedding
        """
        if not contextual_summary:
            return original_content
        return f"{contextual_summary}\n\n{original_content}"

    def get_surrounding_context(
        self,
        chunks: List[str],
        current_index: int,
        window_size: int = 3
    ) -> str:
        """
        Get surrounding chunks as context for the current chunk.

        Args:
            chunks: List of all chunk contents
            current_index: Index of the current chunk
            window_size: Number of chunks before/after to include

        Returns:
            Combined text of surrounding chunks
        """
        start = max(0, current_index - window_size)
        end = min(len(chunks), current_index + window_size + 1)

        # Get surrounding chunks, excluding the current one
        context_chunks = []
        for i in range(start, end):
            if i != current_index:
                # Truncate long chunks to avoid token limits
                chunk_text = chunks[i][:500]
                context_chunks.append(f"[Chunk {i+1}]: {chunk_text}")

        return "\n---\n".join(context_chunks)

    async def process_chunks_with_context(
        self,
        chunks: List[str],
        window_size: Optional[int] = None
    ) -> List[dict]:
        """
        Process a list of chunks, generating contextual summaries for each.

        Args:
            chunks: List of chunk content strings
            window_size: Context window (uses config default if not specified)

        Returns:
            List of dicts with 'original', 'summary', and 'contextualized' keys
        """
        if window_size is None:
            window_size = settings.contextual_context_window

        results = []
        total = len(chunks)

        for i, chunk_content in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{total} for contextual summary...")

            # Get surrounding context
            context = self.get_surrounding_context(chunks, i, window_size)

            # Generate summary
            summary = await self.generate_contextual_summary(
                chunk_content=chunk_content,
                document_context=context
            )

            # Create contextualized version
            contextualized = self.create_contextualized_chunk(
                original_content=chunk_content,
                contextual_summary=summary
            )

            results.append({
                "original": chunk_content,
                "summary": summary,
                "contextualized": contextualized
            })

        logger.info(f"Processed {len(results)} chunks with contextual summaries")
        return results
