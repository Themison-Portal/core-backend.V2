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
        tool_response_with_citations = None

        # Extract response and tool results from messages
        for message in result.get('messages', []):
            message_type = message.__class__.__name__
            print(f"🔍 Message type: {message_type}")

            # Check if this is a ToolMessage (result of tool execution)
            if message_type == 'ToolMessage':
                # ToolMessage.content has the generation with citations
                if hasattr(message, 'content'):
                    print(f"🔧 Found ToolMessage content: {message.content[:100]}...")
                    tool_response_with_citations = message.content

                # ToolMessage.artifact has the retrieved_documents dict
                if hasattr(message, 'artifact') and message.artifact:
                    artifact = message.artifact
                    if isinstance(artifact, dict):
                        retrieved_documents = artifact.get('retrieved_documents', [])

            # Check if this is an AIMessage with tool_calls (tool invocation)
            elif hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = message.tool_calls

            # Check if this is an AI response (skip HumanMessage)
            elif hasattr(message, 'content') and message_type == 'AIMessage':
                response = message.content
        
        # Extract tool_calls from the result
        if 'tool_calls' in result:
            tool_calls = result['tool_calls']
        elif result.get('messages'):
            for msg in result['messages']:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls = msg.tool_calls
                    break
         
        # Extract citations from tool response (which has proper format)
        sources = []
        print(f"🔍 QUERY ENDPOINT - Starting citation extraction...")
        print(f"📝 Agent response: {response[:100] if response else 'None'}...")
        print(f"🔧 Tool response: {tool_response_with_citations[:100] if tool_response_with_citations else 'None'}...")

        # Try to extract from tool response first (has proper citations)
        citation_source = tool_response_with_citations or response
        if citation_source:
            import re
            # Extract page citations - handle both single and double quotes
            citation_pattern = r'\[Page (\d+): ["\']([^"\']+)["\']\]'
            citations = re.findall(citation_pattern, citation_source)

            print(f"🎯 REGEX PATTERN: {citation_pattern}")
            print(f"📊 CITATIONS FOUND: {len(citations)} in {'tool' if tool_response_with_citations else 'agent'} response")

            for i, (page_num, quote) in enumerate(citations):
                print(f"  📖 Citation {i+1}: Page {page_num} - '{quote[:50]}...'")
                sources.append({
                    "section": f"Page {page_num}",
                    "page": int(page_num),
                    "content": quote,
                    "exactText": quote,
                    "relevance": "high",
                    "context": f"Citation from document page {page_num}",
                    "highlightURL": ""  # Could add document URL + page anchor if needed
                })

            print(f"✅ FINAL SOURCES ARRAY: {len(sources)} sources created")

        final_response = {
            "response": response,
            "sources": sources,  # Add sources for frontend
            "tool_calls": tool_calls,
            "retrieved_documents": retrieved_documents,
        }

        print(f"🚀 FINAL RESPONSE TO FRONTEND:")
        print(f"   📝 Response length: {len(response)}")
        print(f"   📊 Sources count: {len(sources)}")
        print(f"   🔧 Tool calls: {len(tool_calls)}")
        print(f"   📚 Retrieved docs: {len(retrieved_documents)}")

        return final_response
        
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
