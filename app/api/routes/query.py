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

            # Strategy 1: Extract individual citations with quotes [Page X: "quote"]
            individual_pattern = r'\[Page (\d+): ["\']([^"\']+)["\']\]'
            individual_citations = re.findall(individual_pattern, citation_source)

            print(f"🎯 Individual citations pattern: {individual_pattern}")
            print(f"📊 Individual citations found: {len(individual_citations)}")

            for i, (page_num, quote) in enumerate(individual_citations):
                print(f"  📖 Individual {i+1}: Page {page_num} - '{quote[:50]}...'")
                sources.append({
                    "section": f"Page {page_num}",
                    "page": int(page_num),
                    "content": quote,
                    "exactText": quote,
                    "relevance": "high",
                    "context": f"Direct citation from page {page_num}",
                    "highlightURL": ""
                })

            # Strategy 2: Extract section references
            # Matches multiple formats:
            # - "Section 5.1 Inclusion Criteria (Page 34)"
            # - "Reference: Section 4.1.1 Inclusion Criteria (Pages 10, 34)"
            # - "Section 4.1.2 Exclusion Criteria (Pages 10-11, 34-35)"
            section_pattern = r'(?:Reference:\s+)?(?:Section|section)\s+([\d\.]+)\s+([^(]+?)\s*\(Pages?\s+([\d\-,\s]+)\)'
            section_citations = re.findall(section_pattern, citation_source)

            print(f"🎯 Section references pattern: {section_pattern}")
            print(f"📊 Section references found: {len(section_citations)}")

            for i, (section_num, section_name, page_range) in enumerate(section_citations):
                # Parse page range - can be "34", "10-11", "10, 34", etc.
                # Extract first page number
                page_numbers = re.findall(r'\d+', page_range)
                if page_numbers:
                    page_start = int(page_numbers[0])
                else:
                    page_start = 1  # Fallback

                section_title = f"Section {section_num} {section_name.strip()}"
                print(f"  📚 Section {i+1}: {section_title} (Pages: {page_range})")

                # Add section reference as a source for highlighting
                sources.append({
                    "section": section_title,
                    "page": page_start,
                    "content": f"See {section_title} for complete information",
                    "exactText": section_title,
                    "relevance": "high",
                    "context": f"Section reference (Pages: {page_range})",
                    "highlightURL": ""
                })

            # Strategy 3: Extract heading-style references with pages in parentheses
            # Matches formats like:
            # - "Study Drug Discontinuation (Page 61-62):"
            # - "Contraception Requirements (Section 4.1.1, Page 34):"
            # - "Pregnancy Testing Schedule (Pages 46-49, 52):"
            # - "Exclusion Criteria (Page 34):"
            heading_pattern = r'\*\*([^(]+?)\s*\(((?:Section\s+[\d\.]+,?\s*)?Pages?\s+[\d\-,\s]+)\):\*\*'
            heading_citations = re.findall(heading_pattern, citation_source)

            print(f"🎯 Heading references pattern: {heading_pattern}")
            print(f"📊 Heading references found: {len(heading_citations)}")

            for i, (heading_name, page_info) in enumerate(heading_citations):
                # Extract page numbers from the page_info string
                page_numbers = re.findall(r'\d+', page_info)
                if page_numbers:
                    page_start = int(page_numbers[0])
                else:
                    page_start = 1  # Fallback

                # Extract section number if present
                section_match = re.search(r'Section\s+([\d\.]+)', page_info)
                if section_match:
                    section_num = section_match.group(1)
                    heading_title = f"{heading_name.strip()} (Section {section_num})"
                else:
                    heading_title = heading_name.strip()

                print(f"  📋 Heading {i+1}: {heading_title} (Page info: {page_info})")

                # Generate more descriptive content based on heading name
                content_description = f"This section covers: {heading_title.lower()}. Refer to the document for complete details."

                # Add heading reference as a source for highlighting
                sources.append({
                    "section": heading_title,
                    "page": page_start,
                    "content": content_description,
                    "exactText": heading_title,
                    "relevance": "high",
                    "context": f"Referenced in response ({page_info})",
                    "highlightURL": ""
                })

            print(f"✅ TOTAL SOURCES: {len(sources)} ({len(individual_citations)} individual + {len(section_citations)} section refs + {len(heading_citations)} heading refs)")

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
