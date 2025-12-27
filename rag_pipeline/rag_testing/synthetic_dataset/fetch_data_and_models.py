from dotenv import load_dotenv
from sqlalchemy import text
from rag_pipeline.database import AsyncSessionLocal # Assumed to be configured
from pathlib import Path


# LLM
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# EMBEDDING
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings


LLM_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"


# Load environment variables from .env
load_dotenv()


async def fetch_data():
    print("Fetching data")
    # fetch data and sort it by page and creation, so 
    # we have the whole document in proper page-wise chunks
    async with AsyncSessionLocal() as session:
        sql = text(
            """
            SELECT dd.id, dd.document_id, dd.content, dd.page_number
            FROM document_chunks_docling dd
            ORDER BY dd.page_number ASC, dd.created_at ASC
            """
            )
        
    
        result = await session.execute(
            sql
        )

        return result.fetchall()


def page_wise_window_generator(docs):
    """
    Takes each row from the database, collects it into one page and yields a result
    for knowledge graph node's creation.
    
    :param chunks: Description
    """

    i = 0
    current_page = 1
    window = {"document_chunk_content": "",
            "metadata": {
                "document_id": "",
                "page_number": ""
            }
    }
    while i < len(docs):
        # we gather all chunks and yield each chunk
        row = docs[i]
        page = window
        page["metadata"]["document_id"] = str(row.document_id)
        page["metadata"]["page_number"] = str(current_page)
        while i < len(docs) and docs[i].page_number == current_page:
            page["document_chunk_content"] = row.content
            i += 1

        yield page

        # after processing one page go to the next one
        current_page += 1



def get_pipeline_components():
    print("Getting LLM handle")
    # LLM
    chat_model = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.1)
    generator_llm = LangchainLLMWrapper(chat_model)
    # EMBEDDING
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return generator_embeddings, generator_llm