import os
import uuid
import asyncio
from functools import partial
import logging

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings # NOTE: Assumed to be synchronous
# from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from database import AsyncSessionLocal
from models.vector_model import Protocol, ProtocolChunk 


# load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data")
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536 

# ============================================================
# ASYNC/SYNC BRIDGING HELPERS 
# ============================================================
def clean_text_sync(text: str) -> str:
    """Synchronous text cleaning."""
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def split_paragraphs_sync(page_content: str):
    """Synchronous paragraph splitting."""
    # returns list of paragraph strings
    return [p.strip() for p in page_content.split("\n\n") if p.strip()]


async def run_in_thread(fn, *args, **kwargs):
    """Utility to run blocking synchronous functions in a dedicated thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))

# ============================================================
# DOCUMENT LOADING & PROCESSING (I/O BOUND) 
# ============================================================
async def load_documents_async():
    """Load PDFs using PyMuPDF loader — runs blocking loader in a thread."""
    pdf_files = [
        os.path.join(DATA_PATH, f)
        for f in os.listdir(DATA_PATH)
        if f.endswith(".pdf")
    ]
    documents = []
    
    if not pdf_files:
        logging.warning(f"No PDF files found in {DATA_PATH}. Skipping loading.")
        return []

    for pdf_file in pdf_files:
        # PyMuPDFLoader.load() is blocking, run inside thread
        loader = PyMuPDFLoader(pdf_file)
        docs = await run_in_thread(loader.load)
        for doc in docs:
            doc.page_content = clean_text_sync(doc.page_content)
            # Adjust page number (PyMuPDFLoader is 0-indexed)
            if "page" in doc.metadata:
                doc.metadata["page"] = int(doc.metadata["page"]) + 1 
            doc.metadata["source"] = pdf_file
        documents.extend(docs)
    logging.info(f"Loaded {len(documents)} pages from {len(pdf_files)} PDF files.")
    return documents


def split_text_sync(documents: list[Document]):
    """Synchronous document splitting into chunks."""
    all_chunks = []
    for doc in documents:
        paragraphs = split_paragraphs_sync(doc.page_content)
        paragraph_counter = 1
        
        for para in paragraphs:
            para = para.strip()
            if not para: continue
            
            # Simple check for very large paragraphs requiring recursive splitting
            if len(para) > 2000:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=300,
                    separators=["\n", ".", "!", "?", ",", " ", ""],
                )
                sub_docs = splitter.create_documents([para])
                for sub_doc in sub_docs:
                    sub_doc.metadata["page"] = doc.metadata.get("page")
                    sub_doc.metadata["paragraph"] = paragraph_counter
                    sub_doc.metadata["source"] = doc.metadata.get("source")
                    all_chunks.append(sub_doc)
            else:
                chunk_doc = Document(
                    page_content=para,
                    metadata={
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page"),
                        "paragraph": paragraph_counter
                    }
                )
                all_chunks.append(chunk_doc)
            paragraph_counter += 1
            
    return all_chunks


async def compute_embedding_async(embeddings, text: str):
    """Runs synchronous embedding computation in a thread pool."""
    # embeddings.embed_query is blocking -> run in thread
    return await run_in_thread(embeddings.embed_query, text)

# ============================================================
# DATABASE STORAGE (I/O BOUND) 
# ============================================================

async def save_to_db_async(chunks: list[Document], embeddings: OpenAIEmbeddings):
    """
    Store paragraph-level chunks using async SQLAlchemy session.
    Embeddings client is passed as a dependency.
    """
    if not chunks:
        logging.info("No chunks to save. Exiting save_to_db_async.")
        return

    async with AsyncSessionLocal() as session:
        logging.info(f"Starting transaction to save {len(chunks)} chunks...")
        try:
            for i, chunk in enumerate(chunks):
                source_path = chunk.metadata.get("source", "unknown")
                protocol_title = os.path.basename(source_path).replace(".pdf", "")
                
                if i % 50 == 0:
                     logging.info(f"Processing chunk {i}/{len(chunks)}...")

                # 1. Check or create parent Protocol entry
                q = select(Protocol).where(Protocol.title == protocol_title)
                res = await session.execute(q)
                protocol = res.scalar_one_or_none()
                
                if not protocol:
                    protocol = Protocol(title=protocol_title)
                    session.add(protocol)
                    await session.flush()  # Ensure protocol.id available

                # 2. Skip if chunk already exists (simple content check)
                q2 = select(ProtocolChunk).where(ProtocolChunk.content == chunk.page_content)
                res2 = await session.execute(q2)
                existing = res2.scalar_one_or_none()
                if existing:
                    logging.debug(f"Chunk already exists: {protocol_title} (p. {chunk.metadata.get('page')})")
                    continue

                # 3. Compute embedding (in thread)
                embedding_vector = await compute_embedding_async(embeddings, chunk.page_content)

                # 4. Create and store chunk
                new_chunk = ProtocolChunk(
                    id=str(uuid.uuid4()),
                    content=chunk.page_content,
                    page_number=chunk.metadata.get("page", None),
                    paragraph_number=chunk.metadata.get("paragraph", None),
                    embedding=embedding_vector,
                    protocol_id=protocol.id,
                )
                session.add(new_chunk)

            await session.commit()
            logging.info(f"✅ Successfully saved {len(chunks)} paragraph-level chunks into the database.")
            
        except SQLAlchemyError as e:
            logging.error(f"DB error: {e}")
            await session.rollback()
        except Exception as e:
            logging.error(f"Unexpected error while saving: {e}")
            await session.rollback()


# --- MAIN ASYNC ENTRY POINT ---

async def generate_data_store_async():
    """Initializes clients and orchestrates the entire ingestion process."""
    
    # 1. Initialize synchronous clients ONCE (best practice)
    try:
        logging.info(f"Initializing {EMBEDDING_MODEL_NAME} embedding client...")
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        # Optional: Run a test embedding to ensure connection/API key is valid
        await compute_embedding_async(embeddings, "test initialization")
        logging.info("Embedding client initialized successfully.")
    except Exception as e:
        logging.error(f"FATAL: Error initializing embeddings client. Check API key/model name: {e}")
        return

    # 2. Run data pipeline steps
    logging.info("******* Loading documents... ********")
    docs = await load_documents_async()
    
    if not docs:
        logging.warning("No documents loaded, exiting data generation.")
        return

    logging.info("******* Splitting documents into chunks... ********")
    # Splitting is CPU-bound but potentially long for many documents, so run in thread
    chunks = await run_in_thread(split_text_sync, docs)
    logging.info(f"Split into {len(chunks)} chunks.")
    
    logging.info("******* Saving chunks to database and computing embeddings... ********")
    await save_to_db_async(chunks, embeddings)
    logging.info("Data store generation complete.")


if __name__ == "__main__":
    try:
        asyncio.run(generate_data_store_async())
    except KeyboardInterrupt:
        logging.info("Script interrupted by user.")
    except RuntimeError as e:
        logging.error(f"Script failed due to runtime error: {e}")