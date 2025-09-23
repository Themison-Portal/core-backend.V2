"""
Query routes
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
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
        retrieved_documents = []
        
        # Extract response and tool results from messages
        for message in result.get('messages', []):
            # Check if this is a ToolMessage (result of tool execution)
            if hasattr(message, 'artifact') and message.artifact:
                artifact = message.artifact
                if isinstance(artifact, dict):
                    retrieved_documents = artifact.get('retrieved_documents', [])
                    
            # Check if this is an AIMessage with tool_calls (tool invocation)
            elif hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = message.tool_calls
                
            # Check if this is the final response
            elif hasattr(message, 'content') and not hasattr(message, 'tool_calls'):
                response = message.content
        
        # Extract tool_calls from the result
        if 'tool_calls' in result:
            tool_calls = result['tool_calls']
        elif result.get('messages'):
            for msg in result['messages']:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls = msg.tool_calls
                    break
         
        return {
            "response": response,
            "tool_calls": tool_calls,
            "retrieved_documents": retrieved_documents,
        }
        
    except Exception as e:
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
