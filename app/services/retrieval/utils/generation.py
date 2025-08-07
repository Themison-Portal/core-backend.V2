import asyncio
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from app.core.openAI import async_client

load_dotenv()

logger = logging.getLogger(__name__)

def generate_response(query, retrieved_documents: List[Dict[Any, Any]]):
    # Format retrieved documents into context
    context = ""
    print(f"retrieved_documents: {retrieved_documents}")
    for i, doc in enumerate(retrieved_documents):
        source = doc.get('metadata', {}).get('source', 'Unknown')
        page = doc.get('metadata', {}).get('page', 'Unknown')
        context += f"\nDocument {i+1} (Source: {source}, Page: {page}):\n{doc['content']}\n"
    
    # Create a structured prompt
    prompt = f"""
    User Query: {query}
    
    Retrieved Information:
    {context}
    
    Based on the above information only, provide a comprehensive answer to the user's query.
    If the information is not sufficient to answer the query, acknowledge this limitation.
    Include citations to the specific documents you reference.
    """
    
    return prompt

async def call_llm_stream(prompt):
    # Set up your API client
    client = async_client

    try:     
        # Fixed API call format
        stream = await client.chat.completions.create(
            model="gpt-4.1-nano",  # Use a valid model name
            messages=[
                {
                    "role": "developer", 
                    "content": "You are a helpful assistant, answer the user's query based on the provided information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=True,
            temperature=0.5,
        )
            
        async def generator():
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return generator()
    except Exception as e:
        logger.error(f"Error calling LLM API: {e}")
        async def error_generator():
            yield "Sorry, I encountered an error generating a response."
        return error_generator()