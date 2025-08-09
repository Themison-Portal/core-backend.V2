"""
This module contains the RAG agent.
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.core.openai import llm
from app.services.agenticRag.tools import (
    documents_analysis_tool,
    generic_tool,
    retrieve_documents_tool,
)
from app.services.retrieval.retriever import create_retriever


class RagAgent:
    """
    A class that represents a RAG agent.
    """
    
    def __init__(self):
        self.retriever = create_retriever(match_count=5, query_chunk_size=500)
        self.tools = [retrieve_documents_tool, documents_analysis_tool, generic_tool]
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.system_message = SystemMessage(
            content="You are a helpful assistant tasked with finding and explaining relevant information about movies."
        )
    
    def create_graph(self):
        """
        Create the graph for the RAG agent.
        """
        
        # State

        graph = StateGraph(MessagesState)
        
        graph.add_node("assistant", self.assistant)
        graph.add_node("tools", ToolNode(self.tools))
        
        graph.add_edge(START, "assistant")
        graph.add_conditional_edges(
            "assistant",
            # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
            # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
            tools_condition,
        )
        
        graph.add_edge("tools", "assistant")
        graph.add_edge("assistant", END)
        
        return graph.compile()

    def assistant(self,state: MessagesState):
        """
        The assistant node.
        """
        return {"messages": [self.llm_with_tools.invoke([self.system_message] + state["messages"])]}