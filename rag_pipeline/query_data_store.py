import asyncio
from functools import partial

from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from sqlalchemy import text

from rag_pipeline.database import AsyncSessionLocal # Assumed to be configured


LLM_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"


# ============================================================
# TEMPLATES (omitted for brevity)
# ============================================================

PROMPT_TEMPLATE = """ 
You are an expert clinical trial assistant. Your goal is to answer the question using ONLY the provided context. 
⚠️ CRITICAL CITATION RULES ⚠️ 
The provided context includes source metadata in this exact format: 
[Protocol_Title (p. PAGE_NUMBER, ¶PARAGRAPH_NUMBER)] or: [Protocol_Title (p. PAGE_NUMBER)] (when paragraph is not available) 

You MUST follow these rules when answering:
 1️⃣ When you use information from a source, you MUST add an inline citation immediately after the sentence, 
 using this exact format: (p. PAGE_NUMBER, ¶PARAGRAPH_NUMBER) or if paragraph is missing: (p. PAGE_NUMBER, ¶None)
 2️⃣ Inline citations MUST appear in the main answer text — NOT only in the SOURCES USED section.
 3️⃣ At the end of your answer, include a section exactly named: SOURCES USED:
 4️⃣ In that section, list each referenced source once, in this exact format: Protocol_Title (p. PAGE_NUMBER, ¶PARAGRAPH_NUMBER)
 5️⃣ If multiple sentences use the same source, do NOT duplicate it in the sources list.
 6️⃣ DO NOT fabricate page or paragraph numbers. Use ONLY those explicitly present in the context. 
 7️⃣ DO NOT include any URLs, IDs, brackets, or text not present in the context metadata. 
 ──────────────────────────────── 
 Context: {context} 
 Question: {question} 
 
 Answer: """ 

SUMMARY_TEMPLATE = """ 
You are an expert clinical protocol summarization assistant. 
The user question indicates an emergency or severe side effect. 
Using ONLY the provided context, produce a structured, concise clinical summary including: 
1. Immediate actions 
2. Step-by-step management 
3. Monitoring requirements 
4. Warnings / red flags 
5. Section references 
6. No invented information 

Context: {context} 
Question: {question} 

Provide the summary now: """


# ============================================================
# HELPERS
# ============================================================

async def run_in_thread(fn, *args, **kwargs):
    """Run blocking functions in a background thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


def get_rag_components():
    """Initializes synchronous LangChain components."""
    # Note: These components will be wrapped in threads when using .ainvoke()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    chat_model = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.1)
    return embeddings, chat_model


def embedding_to_pg_vector(emb):
    return "[" + ",".join(str(x) for x in emb) + "]"


# ============================================================
# ASYNC VECTOR SEARCH
# ============================================================

async def search_supabase_similar_chunks(query_text: str, embeddings, top_k: int = 15):
    """
    Async pgvector similarity search, wrapping the synchronous embed_query call.
    """
    # CORRECT: embeddings.embed_query is blocking, so run in thread
    query_vector = await run_in_thread(embeddings.embed_query, query_text)

    query_vector = embedding_to_pg_vector(query_vector)

    async with AsyncSessionLocal() as session:
        sql = text("""
            SELECT pc.id, pc.content, pc.protocol_id, pc.page_number, pc.paragraph_number, p.title,
                1 - (pc.embedding <=> (:v)::vector) AS similarity
            FROM protocol_chunks pc
            JOIN protocols p ON pc.protocol_id = p.id
            ORDER BY pc.embedding <=> (:v)::vector
            LIMIT :k
        """)

        
        result = await session.execute(
            sql,
            {"v": query_vector, "k": top_k}
        )
                
        rows = result.fetchall()

    docs = []
    for row in rows:
        docs.append({
            "page_content": row.content,
            "metadata": {
                "protocol_id": str(row.protocol_id),
                "title": row.title,
                "page_number": row.page_number,
                "paragraph_number": row.paragraph_number
            },
            "score": float(row.similarity),
        })

    return docs


# ============================================================
# CLASSIFICATION
# ============================================================

async def classify_query(chat_model: ChatOpenAI, query_text: str):
    """
    Classifies the query using the chat model.
    Since ChatOpenAI is synchronous but supports .ainvoke(), 
    we use .ainvoke() which handles the thread wrapping internally.
    """
    prompt = f"""
    Classify the following query into one category:
    - emergency
    - severe_side_effect
    - routine

    Query: "{query_text}"

    Only return one word.
    """
    
    
    # This automatically runs the underlying sync call in a thread pool.
    res = await chat_model.ainvoke(prompt)
    return res.content.strip().lower()


# ============================================================
# FORMATTER
# ============================================================

def format_context_with_citation(doc):
    title = doc["metadata"].get("title", "Unknown")
    page = doc["metadata"].get("page_number", "N/A")
    para = doc["metadata"].get("paragraph_number")

    if para:
        return f"[{title} (p. {page}, ¶{para})]\n{doc['page_content']}"
    return f"[{title} (p. {page})]\n{doc['page_content']}"


# ============================================================
# MAIN RAG EXECUTION
# ============================================================

async def rag_query(query_text: str):
    if not query_text:
        raise ValueError("Query text must be provided.")

    embeddings, chat_model = get_rag_components()

    # classify (uses .ainvoke internally)
    category = await classify_query(chat_model, query_text)
    is_complex = category in ["emergency", "severe_side_effect"]
    top_k = 40 if is_complex else 15

    # run similarity search
    results = await search_supabase_similar_chunks(query_text, embeddings, top_k)

    # filter
    filtered_docs = [d for d in results if d["score"] >= 0.33]

    if not filtered_docs:
        return {
            "answer": "The answer is not available in the provided documents.",
            "sources": []
        }

    merged_context = "\n\n".join(format_context_with_citation(doc) for doc in filtered_docs)

    # choose prompt
    template = SUMMARY_TEMPLATE if is_complex else PROMPT_TEMPLATE
    CHAT_PROMPT = ChatPromptTemplate.from_template(template)

    # Build chain
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | CHAT_PROMPT
        | chat_model
        | StrOutputParser()
    )

    # CORRECT: use chain.ainvoke()
    response = await chain.ainvoke({"context": merged_context, "question": query_text})

    # build sources list
    sources = []
    for doc in filtered_docs:
        title = doc["metadata"]["title"]
        page = doc["metadata"]["page_number"]
        para = doc["metadata"]["paragraph_number"]
        
        # Using the citation format specified in the prompt template
        citation = f"(p. {page})"
        if para is not None:
             citation = f"(p. {page}, ¶{para})"
             
        sources.append(f"{title} {citation}")

    sources = list(dict.fromkeys(sources))

    return {
        "answer": response.strip(),
        "sources": sources
    }


# Example usage (for testing)
# if __name__ == '__main__':
#     result = rag_query("What is the protocol for managing severe diarrhea?")
#     print(result)