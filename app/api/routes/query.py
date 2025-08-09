"""
Query routes
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.api.routes.auth import get_current_user
from app.services.agenticRag.agent import RagAgent

router = APIRouter()

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
    Process a query with streaming response
    """
    try:
        # Create streaming agent
        rag_agent = RagAgent().create_graph()
        
        # Create streaming generator
        async def generate_stream():
            async for chunk in rag_agent.astream(
                {"messages": [HumanMessage(content=request.message)]},
                config={"configurable": {"thread_id": request.user_id}}
            ):
                # Extract content from the chunk
                if hasattr(chunk, 'messages') and chunk.messages:
                    for message in chunk.messages:
                        if hasattr(message, 'content') and message.content:
                            yield f"data: {message.content}\n\n"
                elif hasattr(chunk, 'content') and chunk.content:
                    yield f"data: {chunk.content}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/get_chat_history")
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
