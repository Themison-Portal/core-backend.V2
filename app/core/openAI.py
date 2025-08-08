"""
OpenAI client
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

chat_client = ChatOpenAI(
    model="gpt-4.1-mini",
    streaming=True,
    temperature=0.5
)

embedding_client = OpenAIEmbeddings(
    model="text-embedding-3-small"
)