"""
This module contains the RAG agent.
"""

from typing import List, Literal

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage
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
            content="""You are a helpful agent that MUST use tools to answer questions.

            Available tools:
            - documents_retrieval_generation_tool: Use this to search documents and generate responses
            - generic_tool: Use this for general queries
            
            IMPORTANT: Always use the appropriate tool(s) to answer user questions. 
            Do not answer directly without using tools."""
        )
        self.chat_history = InMemoryChatMessageHistory()
        self.document_ids = []
        
    def should_continue(self, state: MessagesState) -> Literal["end", "continue"]:
        """
        Determine whether to continue or not.
        If there is no tool call, then we finish
        Otherwise if there is, we continue
        """
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"
    
    def create_graph(self, document_ids: List[str] = None):
        """
        Create the graph for the RAG agent.
        """
        if document_ids:
            self.document_ids = document_ids
        
        # Update system message to include document_ids context
        if document_ids:
            self.system_message = SystemMessage(
                content=f"""You are a helpful agent tasked with finding and explaining relevant information about movies.

                You have access to these tools:
                - documents_retrieval_tool: Find relevant documents from the knowledge base (use document_ids: {document_ids} to search specific documents)
                - generate_response_tool: Create comprehensive answers based on retrieved information
                - generic_tool: Handle general queries not covered by other tools
                
                The user has specified document IDs: {document_ids}. Use the documents_retrieval_tool with these IDs when searching for information.
                Choose the most appropriate tool(s) based on the user's specific query."""
            )
        
        graph = StateGraph(MessagesState)
        
        graph.add_node("agent", self.agent)
        graph.add_node("tools", ToolNode(self.tools))
        
        graph.add_edge(START, "agent")
        # from langgraph to determine whether or not to use tools
        graph.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            self.should_continue,
            # Finally we pass in a mapping.
            # The keys are strings, and the values are other nodes.
            # END is a special node marking that the graph should finish.
            # What will happen is we will call `should_continue`, and then the output of that
            # will be matched against the keys in this mapping.
            # Based on which one it matches, that node will then be called.
            {
                # If `tools`, then we call the tool node.
                "continue": "tools",
                # Otherwise we finish.
                "end": END,
            },
        )
        
        graph.add_edge("tools", "agent")
        graph.add_edge("agent", END)
        checkpointer = MemorySaver()
        
        return graph.compile(checkpointer=checkpointer)
        

    def agent(self, state: MessagesState):
        """
        The agent node that processes messages and returns state updates.
        """
        result = self.llm_with_tools.invoke([self.system_message] + state["messages"])
        
        tool_calls = result.tool_calls
        
        response = {
            "messages": [result],
            "tool_calls": tool_calls
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