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
        # Create agent
        rag_agent_instance = RagAgent()
        compiled_graph = rag_agent_instance.create_graph(document_ids=request.document_ids)
        
        # Process the query
        result = compiled_graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.user_id}}
        )
        
        # Extract comprehensive information from the result
        response = "No response generated"
        tool_calls = []
        
        if result.get('messages') and len(result['messages']) > 0:
            # Get the last message which should be the agent's response
            final_message = result['messages'][-1]
            response = final_message.content
        
        # Extract tool_calls from the result
        if 'tool_calls' in result:
            tool_calls = result['tool_calls']
        elif result.get('messages'):
            # Check if tool_calls are in any of the messages
            for msg in result['messages']:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls = msg.tool_calls
                    break
                elif hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
                    tool_calls = msg.additional_kwargs['tool_calls']
                    break
 
        return {
            "response": response,
            "tool_calls": tool_calls,
        }
        
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
