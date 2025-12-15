import os
import re
import json
import hashlib
import redis
import logging
import fitz 
import uuid
import asyncio
from typing import Optional
from functools import partial
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import select, and_, text
from sqlalchemy.exc import IntegrityError
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

from rag_pipeline.database import AsyncSessionLocal
from rag_pipeline.models.vector_model_dockling import Protocol, ProtocolChunk
from rag_pipeline.schema.rag_docling_schema import DoclingRagStructuredResponse
# --- Paths (Defined relative to this file's location) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
LLM_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
TOKENIZER_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2" 
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)
# --------------------------------------------------------


def safe_basename(name: str):
    return os.path.basename(name)


def normalize_text(text: str) -> str:    
    text = text.replace('', '')           
    text = text.replace('\n', ' ')         
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('≥', '>=')
    text = re.sub(r'\s+', ' ', text)       
    return text.strip()


def chunk_text(text: str, max_len: int = 70) -> list[str]:
    words = text.split()
    chunks = []
    chunk = []
    length = 0
    for word in words:
        length += len(word) + 1
        chunk.append(word)
        if length >= max_len:
            chunks.append(' '.join(chunk))
            chunk = []
            length = 0
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks


# --- Initialization function (Called only once from main.py startup) ---
def load_pdf_blocks_into_redis(r_client: redis.Redis):
    """Loads all PDF text blocks into Redis."""
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".pdf"):
            doc_stem = fname[:-4]
            pdf_path = os.path.join(DATA_DIR, fname)
            
            try:
                doc = fitz.open(pdf_path)
                blocks = []
                for page_num, page in enumerate(doc, start=1):
                    for block in page.get_text("blocks"):
                        x0, y0, x1, y1, text, *_ = block
                        if text.strip():
                            blocks.append({
                                "page": page_num,
                                "bbox": [x0, y0, x1, y1],
                                "text": text.strip()
                            })
                doc.close()
                # Store the JSON string as bytes (since r_client is set with decode_responses=False for cache bytes)
                r_client.set(f"blocks:{doc_stem}", json.dumps(blocks).encode('utf-8'))
                logging.info(f"Loaded blocks for {doc_stem}")
            except Exception as e:
                logging.error(f"Error loading {fname}: {e}")
    logging.info("PDF block loading complete.")


# --- Getter function (Requires the client to be passed in) ---
def get_blocks_from_redis(r_client: redis.Redis, doc_stem: str):
    """Retrieves and decodes document blocks from Redis."""
    raw_bytes = r_client.get(f"blocks:{doc_stem}")
    if raw_bytes:
        return json.loads(raw_bytes.decode('utf-8')) 
    return None

# ============================================================
# Ingesion Starts Here
# ============================================================


async def process_pdf_document(document_id: uuid.UUID, pdf_path: str):
    logging.info(f"Processing document {document_id} at {pdf_path}")

    # 1. Load and chunk with Docling + HybridChunker
    loader = DoclingLoader(
        file_path=pdf_path,
        export_type=ExportType.DOC_CHUNKS,  
        chunker=HybridChunker(tokenizer=tokenizer),
    )
    docs = loader.load()  # returns list of Document objects

    # 3. prepare embedding client
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)    

    # 4. embed + store
    await save_chunks_for_document(document_id, docs, embeddings) 

    logging.info(f"Finished processing document {document_id}.")


async def run_in_thread(fn, *args, **kwargs):
    """Utility to run blocking synchronous functions in a dedicated thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


# --- Check hash function ---
async def check_hash_exists(file_hash: str) -> Optional[uuid.UUID]:
    """
    Return the Protocol.id (UUID) if a document with this hash already exists, else None.
    """
    async with AsyncSessionLocal() as session:
        stmt = select(Protocol.id).where(Protocol.protocol_hash == file_hash)
        result = await session.scalars(stmt)
        document_id = result.first()
        return document_id  # could be None or UUID


async def compute_embedding_async(embeddings, text: str):
    """Runs synchronous embedding computation in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, embeddings.embed_query, text)


async def save_chunks_for_document(document_id: uuid.UUID, docs, embeddings):
    """
    Save LangChain Docling chunks into your ProtocolChunkDocling table.
    Each `doc` is a LangChain Document with `page_content` and `metadata`.
    """
    async with AsyncSessionLocal() as session:
        for i, doc in enumerate(docs):
            
            citation_meta = extract_docling_citation_metadata(doc.metadata)
            content = doc.page_content

            # Compute embedding asynchronously
            embedding_vector = await compute_embedding_async(embeddings, content)

            # Extract Docling metadata
            
            page_number = citation_meta["page_number"]
            
            print(f"page: {page_number} ")
            chunk = ProtocolChunk(
                id=uuid.uuid4(),
                protocol_id=document_id,
                content=content,
                embedding=embedding_vector,
                source=str(document_id),
                page_number=page_number,
                paragraph_number=None,
                metadata_json=doc.metadata,
            )
            session.add(chunk)

            if i % 50 == 0:
                logging.info(f"Saved {i}/{len(docs)} chunks...")

        await session.commit()



async def create_protocol_row(filename: str, file_hash: str, trial_id: uuid.UUID, uploaded_by: Optional[uuid.UUID] = None) -> uuid.UUID:
    """
    Insert a new Protocol row and return its id.
    Handles race conditions by catching IntegrityError and returning existing id.
    Assumes Protocol has: id (UUID PK), name/text, protocol_hash (UNIQUE), trial_id, uploaded_by, uploaded_at
    """
    async with AsyncSessionLocal() as session:
        proto = Protocol(title=filename, protocol_hash=file_hash, trial_id=trial_id, uploaded_by=uploaded_by)
        session.add(proto)
        try:
            await session.commit()
            await session.refresh(proto)  # populates proto.id
            return proto.id
        except IntegrityError:
            # Another process inserted same hash concurrently; fetch existing row
            await session.rollback()
            stmt = select(Protocol.id).where(Protocol.protocol_hash == file_hash)
            result = await session.scalars(stmt)
            existing_id = result.first()
            if existing_id is not None:
                return existing_id
            # unexpected: re-raise if we still can't find it
            raise


def extract_docling_citation_metadata(metadata_json):
    """
    Returns a dict with page_number and headings for a chunk.
    """
    try:
        dl_meta = metadata_json.get("dl_meta", {})
        doc_items = dl_meta.get("doc_items", [])
        headings = dl_meta.get("headings", [])

        # Docling provides provenance info for each doc_item
        page_number = None
        if doc_items:
            prov_list = doc_items[0].get("prov", [])
            if prov_list:
                page_number = prov_list[0].get("page_no")

        return {
            "page_number": page_number,
            "headings": headings or []
        }

    except Exception:
        return {
            "page_number": None,
            "headings": []
        }


# ============================================================
# Retrieval Starts Here
# ============================================================

def embedding_to_pg_vector(emb):
    return "[" + ",".join(str(x) for x in emb) + "]"

async def search_similar_chunks_docling(
    query_text: str,
    embeddings,
    protocol_id: uuid.UUID,
    top_k: int = 20,
):
    query_vector = await run_in_thread(embeddings.embed_query, query_text)
    query_vector = embedding_to_pg_vector(query_vector)

    async with AsyncSessionLocal() as session:
        sql = text("""
            SELECT
                pc.content,
                pc.page_number,
                pc.chunk_metadata,
                p.document_name,
                1 - (pc.embedding <=> (:v)::vector) AS similarity
            FROM document_chunks_docling pc
            JOIN trial_documents p ON pc.document_id = p.id
            WHERE pc.document_id = :pid
            ORDER BY pc.embedding <=> (:v)::vector
            LIMIT :k
        """)

        result = await session.execute(
            sql,
            {"v": query_vector, "k": top_k, "pid": protocol_id}
        )

        rows = result.fetchall()

    docs = []
    for row in rows:
        docs.append({
            "page_content": row.content,
            "score": float(row.similarity),
            "metadata": {
                "title": row.document_name,
                "page": row.page_number,
                "docling": row.chunk_metadata,
            },
        })

    return docs


def format_context_docling(doc):
    title = doc["metadata"].get("title", "Unknown Protocol")
    chunk_metadata = doc["metadata"].get("docling", {})

    citation_meta = extract_docling_citation_metadata(chunk_metadata)

    page = citation_meta["page_number"]
    headings = citation_meta["headings"]

    section = headings[-1] if headings else None

    if page and section:
        header = f"[{title} (p. {page}, section: {section})]"
    elif page:
        header = f"[{title} (p. {page})]"
    else:
        header = f"[{title}]"

    return f"{header}\n{doc['page_content']}"

# ============================================================
# Generation Starts Here
# ============================================================

UNIFIED_PROMPT_TEMPLATE = """
You are an expert clinical protocol assistant.

⚠️ CRITICAL RULES ⚠️
• You MUST answer the question using ONLY the provided context.
• You are strictly forbidden from using any knowledge not in the context.
• Each context block starts with a citation header:
  [Protocol_Title (p. PAGE_NUMBER)] or
  [Protocol_Title (p. PAGE_NUMBER, section: SECTION_TITLE)]

• Inline citations:
  – Every fact in your answer MUST have an inline citation immediately after it.
  – Format: (Protocol_Title, p. PAGE_NUMBER[, section: SECTION_TITLE])
  – If section metadata is missing, omit it.

• Do NOT:
  – Invent page numbers or section titles
  – Include IDs, brackets, URLs, or any metadata not in the context

• If context does not answer the question, say:
  "The provided documents do not contain this information."

──────────────────────────────
CONTEXT:
{context}

QUESTION:
{question}

──────────────────────────────
OUTPUT:
Return ONLY valid JSON in this exact structure:

{{
  "response": "<Markdown formatted clinical answer with headings, lists, emphasis, and inline citations>",
  "sources": [
    {{
      "protocol": "<Protocol_Title>",
      "page": <PAGE_NUMBER>,
      "section": "<SECTION_TITLE or null>",
      "relevance": "<high | medium | low>",
      "exactText": "<verbatim supporting excerpt from context>"
    }}
  ],  
}}

Do NOT include explanations, extra text, or markdown code blocks.
"""

async def rag_query_docling(query_text: str, protocol_id: UUID ):

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    top_k = 40
    # run similarity search
    results = await search_similar_chunks_docling(query_text, embeddings, protocol_id, top_k)
    
    # filter
    filtered_docs = [d for d in results if d["score"] >= 0.04]

    if not filtered_docs:
        return {
            "answer": "The answer is not available in the provided documents.",
            "sources": []
        }
    
    filtered_docs.sort(key=lambda d: d["score"], reverse=True)

    merged_context = "\n\n".join(format_context_docling(doc) for doc in filtered_docs)
    
    structured_chat_model = ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=0.0
        ).with_structured_output(DoclingRagStructuredResponse)

    CHAT_PROMPT = ChatPromptTemplate.from_template(UNIFIED_PROMPT_TEMPLATE)

    chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | CHAT_PROMPT
            | structured_chat_model        
        )

    # CORRECT: use chain.ainvoke()
    result: DoclingRagStructuredResponse = await chain.ainvoke({
        "context": merged_context,
        "question": query_text
    })
    print(result)
    if isinstance(result, dict):
        response = result.get("response")
        sources = result.get("sources", [])
    else:
        response = result.response
        sources = result.sources

    return {
        "response": response,
        "sources": sources,
        "tool_calls": []
    }