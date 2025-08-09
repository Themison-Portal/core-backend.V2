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
    rag_agent: RagAgent = Depends(RagAgent().create_graph()),
    current_user: dict = Depends(get_current_user)
):
    """
    Process a query
    """
    try:

        # Full RAG pipeline with streaming response
        generator = await rag_agent.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.user_id}}
        )
        return StreamingResponse(
            generator,
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))