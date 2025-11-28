"""
Query routes
"""

import re
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.api.routes.auth import get_current_user
from app.services.agenticRag.agent import RagAgent

router = APIRouter()


def normalize_markdown(text: str) -> str:
    """
    Normalize markdown formatting for proper rendering in ReactMarkdown.

    - Ensures double line breaks before/after headers
    - Ensures proper line breaks for list items
    - Cleans up excessive whitespace
    """
    # Ensure double line breaks before headers (## Header)
    text = re.sub(r'\n(#{1,6}\s+)', r'\n\n\1', text)
    text = re.sub(r'(#{1,6}\s+[^\n]+)\n(?!\n)', r'\1\n\n', text)

    # Ensure single line break after each numbered list item (1. item)
    # and double line break before starting a new list
    text = re.sub(r'(\d+\.\s+[^\n]+)\n(\d+\.)', r'\1\n\2', text)

    # Ensure single line break after each bullet list item (- item)
    text = re.sub(r'(-\s+[^\n]+)\n(-\s+)', r'\1\n\2', text)

    # Ensure double line break after lists end (before regular text)
    text = re.sub(r'(\d+\.\s+[^\n]+)\n([^-\d\n#])', r'\1\n\n\2', text)
    text = re.sub(r'(-\s+[^\n]+)\n([^-\d\n#])', r'\1\n\n\2', text)

    # Clean up multiple consecutive line breaks (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Clean up leading/trailing whitespace
    text = text.strip()

    return text

class QueryRequest(BaseModel):
    """
    Query request
    """
    message: str
    user_id: str
    limit: Optional[int] = 5
    document_ids: Optional[List[str]] = None
    
@router.post("")
async def process_query(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Process a query and return response
    """
    try:
        rag_agent_instance = RagAgent()
        compiled_graph = rag_agent_instance.create_graph(document_ids=request.document_ids)
        
        result = compiled_graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.user_id}}
        )
                
        response = "No response generated"
        tool_calls = []
        used_chunks = []

        # Extract response and used chunks from messages
        for message in result.get('messages', []):
            message_type = message.__class__.__name__
            print(f"🔍 Message type: {message_type}")

            # Check if this is a ToolMessage (result of tool execution)
            if message_type == 'ToolMessage':
                # ToolMessage.content has the generated answer
                if hasattr(message, 'content'):
                    print(f"🔧 Found ToolMessage content: {message.content[:100]}...")
                    # Normalize markdown for proper rendering
                    response = normalize_markdown(message.content)

                # ToolMessage.artifact has the structured data
                if hasattr(message, 'artifact') and message.artifact:
                    artifact = message.artifact
                    if isinstance(artifact, dict):
                        used_chunks = artifact.get('used_chunks', [])
                        print(f"📊 Found {len(used_chunks)} used chunks in artifact")

            # Check if this is an AIMessage with tool_calls (tool invocation)
            elif hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = message.tool_calls

            # Check if this is an AI response (final answer after tool)
            elif hasattr(message, 'content') and message_type == 'AIMessage' and message.content:
                # Only use if we don't have a response yet
                if response == "No response generated":
                    response = normalize_markdown(message.content)

        # Map used chunks to sources for frontend
        sources = []
        print(f"\n{'='*80}")
        print(f"🎯 MAPPING CHUNKS TO SOURCES")
        print(f"{'='*80}")

        for chunk in used_chunks:
            page_numbers = chunk.get('page_numbers', [])
            filename = chunk.get('filename', 'Unknown')
            exact_quote = chunk.get('exact_quote', '')  # LLM's exact citation
            chunk_index = chunk.get('chunk_index', 0)

            # Get first page number
            page = page_numbers[0] if page_numbers else 1

            # Format section name
            if len(page_numbers) == 1:
                section = f"Page {page}"
            elif len(page_numbers) > 1:
                section = f"Pages {'-'.join(map(str, page_numbers))}"
            else:
                section = "Page Unknown"

            # Create source object
            source = {
                "section": section,
                "page": page,
                "content": exact_quote[:200] + "..." if len(exact_quote) > 200 else exact_quote,
                "exactText": exact_quote,  # Exact quote for PDF highlighting
                "chunk_index": chunk_index,
                "filename": filename,
                "relevance": "high",
                "context": f"Referenced from {filename}",
                "highlightURL": ""
            }

            sources.append(source)
            print(f"  ✅ Source {len(sources)}: {section} - \"{exact_quote[:60]}...\"")

        print(f"{'='*80}")
        print(f"✅ TOTAL SOURCES: {len(sources)}")
        print(f"{'='*80}\n")

        final_response = {
            "response": response,
            "sources": sources,
            "tool_calls": tool_calls,
        }

        print(f"🚀 FINAL RESPONSE TO FRONTEND:")
        print(f"   📝 Response length: {len(response)}")
        print(f"   📊 Sources count: {len(sources)}")
        print(f"   🔧 Tool calls: {len(tool_calls)}")

        return final_response

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"❌ ERROR IN QUERY ENDPOINT:")
        print(error_traceback)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/get-chat-history")
async def get_chat_history(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get chat history
    """
    try:
        return RagAgent().get_chat_history(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
