import os, re, hashlib, difflib, fitz, uuid
from uuid import UUID

from fastapi import APIRouter, Request, Form, HTTPException, Response, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from urllib.parse import quote, unquote_plus

from rag_pipeline.schema.rag_res_schema import RagStructuredResponse
from rag_pipeline.schema.rag_docling_schema import DoclingRagStructuredResponse
from rag_pipeline.query_data_store import rag_query
from rag_pipeline.query_data_store_biobert import rag_query_biobert
from rag_pipeline.helpers import (
    STATIC_DIR, DATA_DIR, safe_basename,
    normalize_text, chunk_text, get_blocks_from_redis, 
    check_hash_exists, create_protocol_row, process_pdf_document, rag_query_docling
)

router = APIRouter()
templates = Jinja2Templates(directory=STATIC_DIR)

# ============================================================
#   Get Redis Client
# ============================================================
def get_redis_client(request: Request):
    if not hasattr(request.app.state, "redis_client"):
        raise HTTPException(status_code=500, detail="Redis client not initialized.")
    return request.app.state.redis_client


# ============================================================
#   Markdown Normalization
# ============================================================
def normalize_markdown(text: str) -> str:
    """
    Normalize markdown for ReactMarkdown rendering.
    """
    # Ensure double line breaks before headers
    text = re.sub(r'\n(#{1,6}\s+)', r'\n\n\1', text)
    text = re.sub(r'(#{1,6}\s+[^\n]+)\n(?!\n)', r'\1\n\n', text)

    # Numbered list formatting
    text = re.sub(r'(\d+\.\s+[^\n]+)\n(\d+\.)', r'\1\n\2', text)

    # Bullet list formatting
    text = re.sub(r'(-\s+[^\n]+)\n(-\s+)', r'\1\n\2', text)

    # Double break after list ends
    text = re.sub(r'(\d+\.\s+[^\n]+)\n([^-\d\n#])', r'\1\n\n\2', text)
    text = re.sub(r'(-\s+[^\n]+)\n([^-\d\n#])', r'\1\n\n\2', text)

    # Remove too many newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

# ============================================================
#        Home Endpoint
# ============================================================
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# ============================================================
#        PDF Viewer Endpoint
# ============================================================
@router.get("/viewer", response_class=HTMLResponse, include_in_schema=False)
async def viewer(request: Request, doc: str, page: int = 1, highlight: str = ""):
    return templates.TemplateResponse(
        "viewer.html",
        {"request": request, "doc": doc, "page": page, "highlight": highlight}
    )

# ============================================================
#        Main RAG Query Endpoint
# ============================================================
@router.post("/query", response_class=JSONResponse)
async def query_endpoint(query: str = Form(...)):
    """
    RAG query endpoint returning structured JSON with response and sources.
    """
    # Call your RAG function that uses structured model
    result: RagStructuredResponse = await rag_query(query)

    print("RAG Query Result:", result)
    # The structured model already returned 'response' and 'sources'
    return {
        "response": result['response'],  # markdown-safe answer
        "sources": result['sources'],    # list of fully structured source dicts
        "tool_calls": []              # keep for frontend compatibility
    }


# ============================================================
#        Main RAG-BioBERT Query Endpoint
# ============================================================
@router.post("/query-biobert", response_class=JSONResponse)
async def query_endpoint(query: str = Form(...)):
    """
    RAG query endpoint returning structured JSON with response and sources.
    """
    # Call your RAG function that uses structured model
    result: RagStructuredResponse = await rag_query_biobert(query)

    print("BioBERT RAG Query Result:", result)
    # The structured model already returned 'response' and 'sources'
    return {
        "response": result['response'],  # markdown-safe answer
        "sources": result['sources'],    # list of fully structured source dicts
        "tool_calls": []              # keep for frontend compatibility
    }

# ============================================================
#        PDF HIGHLIGHT
# ============================================================
@router.get("/highlighted_pdf", include_in_schema=False)
async def highlighted_pdf(request: Request, doc: str, page: int, highlight: str = ""):

    r_client = get_redis_client(request)

    # --- Clear previous highlighted PDFs for this document ---
    doc_stem = safe_basename(doc).replace(".pdf", "")
    keys_to_delete = r_client.scan_iter(f"highlighted:{doc_stem}:*")
    for key in keys_to_delete:
        r_client.delete(key)

    decoded = unquote_plus(highlight or "")
    highlight_texts = [h.strip() for h in decoded.split("|") if h.strip()]

    if not highlight_texts:
        raise HTTPException(status_code=400, detail="Missing highlight text")

    doc_stem = safe_basename(doc).replace(".pdf", "")
    cache_key = f"highlighted:{doc_stem}:p{page}:{hashlib.sha1(decoded.encode()).hexdigest()[:10]}"

    # Redis GET in thread
    cached_pdf_bytes = await run_in_threadpool(r_client.get, cache_key)
    if cached_pdf_bytes:
        return Response(content=cached_pdf_bytes, media_type="application/pdf")

    cleaned = [re.sub(r'^\s*\d+\.\s*', '', t, flags=re.MULTILINE) for t in highlight_texts]
    cleaned = [c for c in cleaned if c.strip()]

    pdf_path = os.path.join(DATA_DIR, safe_basename(doc))
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    # Load PDF in thread
    doc_pdf = await run_in_threadpool(fitz.open, pdf_path)

    try:
        page_obj = doc_pdf[page - 1]
    except:
        doc_pdf.close()
        raise HTTPException(status_code=400, detail="Invalid page number")

    # --- Highlight search (blocking, run in thread) ---
    def do_highlights():
        blocks = page_obj.get_text("blocks")

        for t in cleaned:
            norm = normalize_text(t)
            chunks = chunk_text(norm, max_len=70)

            for chunk in chunks:
                if not chunk.strip():
                    continue

                areas = page_obj.search_for(chunk)
                if areas:
                    for a in areas:
                        annot = page_obj.add_highlight_annot(a)
                        annot.update()
                    continue

                for b in blocks:
                    block_text = normalize_text(b[4]).lstrip("●-•* ")
                    ratio = difflib.SequenceMatcher(None, chunk, block_text).ratio()
                    if ratio >= 0.6:
                        rect = fitz.Rect(b[0], b[1], b[2], b[3])
                        annot = page_obj.add_highlight_annot(rect)
                        annot.update()

    await run_in_threadpool(do_highlights)

    # Redis fallback also wrapped
    def redis_fallback():
        try:
            blocks = get_blocks_from_redis(r_client, doc_stem) or []
            page_blocks = [b for b in blocks if int(b.get("page", -1)) == int(page)]

            def norm(t): return re.sub(r"\s+", " ", (t or "")).strip().lower()
            targets = [norm(t) for t in highlight_texts]

            for b in page_blocks:
                txt = norm(b.get("text", ""))
                if any(t in txt for t in targets):
                    rect = fitz.Rect(b["bbox"])
                    annot = page_obj.add_highlight_annot(rect)
                    annot.update()
        except Exception as e:
            print("Redis fallback error:", e)

    await run_in_threadpool(redis_fallback)

    # Build PDF bytes
    pdf_bytes = await run_in_threadpool(doc_pdf.tobytes, garbage=3, clean=True, deflate=True)
    doc_pdf.close()

    # Cache result in redis asynchronously
    await run_in_threadpool(r_client.set, cache_key, pdf_bytes, ex=3600)

    return Response(content=pdf_bytes, media_type="application/pdf")

@router.get("/clear_redis_cache")
def clear_redis_cache(request: Request):
    redis_client = get_redis_client(request)
    keys = redis_client.scan_iter("highlighted:*")  # <-- your prefix
    count = 0
    for key in keys:
        redis_client.delete(key)
        count += 1
    return {"deleted": count}

# ============================================================
# Endpoints for new RAG using Docling
# ============================================================

@router.post("/upload-document", response_model=None, response_class=JSONResponse)
async def upload_document(
    file: UploadFile = File(..., description="PDF document to upload"),    
    trial_id: str = Form(..., description="Trial ID associated with this protocol"),
    uploaded_by: str = Form(..., description="The protocol uploaded by"),   
):
    """
    Upload PDF -> compute hash -> dedupe by hash -> save file -> create DB row 
    -> link to trial -> process (Docling + chunk + embed).
    """
    trial_id = uuid.UUID(trial_id)
    uploaded_by = uuid.UUID(uploaded_by)    

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # --- Check hash ---
    existing_doc_id = await check_hash_exists(file_hash)
    if existing_doc_id:
        # Ensure trial → protocol link exists
        await link_trial_protocol(trial_id, existing_doc_id)

        return {
            "message": "File already exists.",
            "document_id": str(existing_doc_id),
        }

    # --- Save file locally ---
    safe_name = safe_basename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    save_path = os.path.join(DATA_DIR, unique_name)

    try:
        with open(save_path, "wb") as out_f:
            out_f.write(file_bytes)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # --- Insert protocol row ---
    try:
        document_id = await create_protocol_row(safe_name, file_hash, trial_id=trial_id, uploaded_by=uploaded_by)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Failed to create document record: {e}")    

    # --- Ingeingestion ---
    await process_pdf_document(document_id, save_path)

    return {
        "message": "File uploaded successfully.",
        "filename": unique_name,
        "document_id": str(document_id),
    }


@router.post("/query-docling")
async def query_endpoint_docling(
    query: str = Form(...),
    document_id: UUID = Form(...)
):
    print(f"document_id: {document_id}")
    
    result: DoclingRagStructuredResponse = await rag_query_docling(query, document_id)
    print(f"final result: {result}")
    return result

