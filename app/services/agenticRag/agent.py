"""
This module contains the RAG agent.
"""

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.core.openai import llm
from app.services.agenticRag.tools import (
    documents_retrieval_tool,
    generate_response_tool,
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
    """
    
    def __init__(self):
        self.tools = [
            documents_retrieval_tool, 
            generic_tool,
            generate_response_tool
        ]
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.system_message = SystemMessage(
            content="""You are a helpful agent tasked with finding and explaining relevant information about movies.

            You have access to these tools:
            - documents_retrieval_tool: Find relevant documents from the knowledge base
            - generate_response_tool: Create comprehensive answers based on retrieved information
            - generic_tool: Handle general queries not covered by other tools
            
            Choose the most appropriate tool(s) based on the user's specific query. 
            Consider whether they need document retrieval, analysis, or general information.
            You can use multiple tools in sequence if needed."""
        )
    
    def create_graph(self):
        """
        Create the graph for the RAG agent.
        """
        
        graph = StateGraph(MessagesState)
        
        graph.add_node("agent", self.agent)
        graph.add_node("tools", ToolNode(self.tools))
        
        graph.add_edge(START, "agent")
        # from langgraph to determine whether or not to use tools
        graph.add_conditional_edges(
            "agent",
            # If the latest message (result) from agent is a tool call -> tools_condition routes to tools
            # If the latest message (result) from agent is a not a tool call -> tools_condition routes to END
            tools_condition,
        )
        
        graph.add_edge("tools", "agent")
        graph.add_edge("agent", END)
        checkpointer = MemorySaver()
        
        return graph.compile(checkpointer=checkpointer)
        

    def agent(self,state: MessagesState):
        """
        The agent node.
        """
        return {"messages": [self.llm_with_tools.invoke([self.system_message] + state["messages"])]}