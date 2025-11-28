"""
This module contains the RAG agent.
"""

from typing import List, Literal

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.core.openai import llm
from app.services.agenticRag.tools import (
    documents_retrieval_generation_tool,
    generic_tool,
)

# usage: rag_agent = RagAgent().create_graph()
# rag_agent.invoke(
#     {"messages": [HumanMessage(content="Explain what a list is in Python")]},
#     config={"configurable": {"thread_id": 'session token from frontend'}}
# )

class RagAgent:
    """
    A class that represents a RAG agent.
    
    it temporary uses in memory chat history to store the chat history
    this is not a good practice and should be replaced with a database
    but for now it is a good way to test the agent
    """
    
    def __init__(self):
        self.tools = [
            generic_tool,
            documents_retrieval_generation_tool
        ]
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.system_message = SystemMessage(
            content="""You are a helpful agent that uses tools to search documents and provide answers.

            Available tools:
            - documents_retrieval_generation_tool: Use this to search documents and generate responses
            - generic_tool: Use this for general queries

            IMPORTANT:
            1. Call documents_retrieval_generation_tool ONCE with the user's original query
            2. The tool will return a complete answer with citations - use that as your final response
            3. Do NOT call the tool multiple times to refine or get more details
            4. Do NOT break down the query into sub-queries - pass the full query to the tool
            5. Return the tool's response exactly as provided"""
        )
        self.chat_history = InMemoryChatMessageHistory()
        self.document_ids = []
        
    def should_continue(self, state: MessagesState) -> Literal["end", "continue"]:
        """
        Determine whether to continue or not.
        Limits to maximum 3 tool calls to prevent excessive iterations.
        """
        messages = state["messages"]
        last_message = messages[-1]

        # Count how many tool calls have been made
        tool_call_count = sum(
            1 for msg in messages
            if isinstance(msg, AIMessage) and msg.tool_calls
        )

        # If we've already made 3 tool calls, stop (hard limit)
        if tool_call_count >= 2:
            print(f"⚠️ Reached maximum tool calls limit (2). Stopping.")
            return END

        # Otherwise, check if there's a tool call pending
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END
    
    def create_graph(self, document_ids: List[str] = None):
        """
        Create the graph for the RAG agent.
        """
        if document_ids:
            self.document_ids = document_ids
        
        # Update system message to include document_ids context
        if document_ids:
            self.system_message = SystemMessage(
                content=f"""You are a helpful agent specialized in analyzing and explaining information from documents.

                You have access to these tools:
                - documents_retrieval_generation_tool: Find relevant documents from the knowledge base (use document_ids: {document_ids} to search specific documents)
                - generic_tool: Handle general queries not covered by other tools

                The user has specified document IDs: {document_ids}. Use the documents_retrieval_generation_tool with these IDs when searching for information.
                Always use the tool's response directly - do not modify or summarize it further.
                Choose the most appropriate tool(s) based on the user's specific query."""
            )
        
        graph = StateGraph(MessagesState)
        
        graph.add_node("agent", self.agent)
        graph.add_node("tools", ToolNode(self.tools))
        
        graph.add_edge(START, "agent")
        # from langgraph to determine whether or not to use tools
        graph.add_conditional_edges("agent", self.should_continue, ["tools", END])

        # AGENTIC LOOP: Allow agent to refine response after tool execution
        graph.add_edge("tools", "agent")
        checkpointer = MemorySaver()

        return graph.compile(checkpointer=checkpointer)
        

    def agent(self, state: MessagesState):
        """
        The agent node that processes messages and returns state updates.
        """
        result = self.llm_with_tools.invoke([self.system_message] + state["messages"])
        # print(result)        
        tool_calls = result.tool_calls
        
        response = {
            "messages": [result],
            "tool_calls": tool_calls,
        }
        
        return response

    def get_chat_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """
        Get the chat history for a given session id.
        """
        chat_history = self.chat_history.get(session_id)
        if chat_history is None:
            chat_history = InMemoryChatMessageHistory()
            self.chat_history[session_id] = chat_history
        return chat_history