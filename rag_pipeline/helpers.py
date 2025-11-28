import os
import re
import json
import hashlib
import redis
import logging
import fitz 

# --- Paths (Defined relative to this file's location) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
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