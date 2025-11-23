"""
This module contains the RAG agent.
"""

from typing import List, Literal

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from anthropic._exceptions import OverloadedError
import time

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
    
    def __init__(self, max_tool_calls: int = 2):
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
        self.max_tool_calls = max_tool_calls
        
    def should_continue(self, state: MessagesState):
        messages = state["messages"]
        last = messages[-1]

        # Case 1: LLM wants to call a tool → allow exactly one step
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"

        # Case 2: Tool result arrived → STOP (prevents recursion)
        if last.type == "tool":
            return END

        # Default: nothing special → end
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
        Includes automatic retries on LLM overload (529 error).
        """

        max_retries = 5
        delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                # Attempt LLM call
                result = self.llm_with_tools.invoke(
                    [self.system_message] + state["messages"]
                )

                tool_calls = result.tool_calls

                return {
                    "messages": [result],
                    "tool_calls": tool_calls,
                }

            except OverloadedError as e:
                # If last retry, rethrow
                if attempt == max_retries - 1:
                    raise e
                
                print(
                    f"⚠️ LLM overloaded (attempt {attempt+1}/{max_retries}). "
                    f"Retrying in {delay} seconds..."
                )
                time.sleep(delay)
                delay *= 2  # exponential backoff

            except Exception as e:
                # Any other error → rethrow
                raise e

    def get_chat_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """
        Get the chat history for a given session id.
        """
        chat_history = self.chat_history.get(session_id)
        if chat_history is None:
            chat_history = InMemoryChatMessageHistory()
            self.chat_history[session_id] = chat_history
        return chat_history