import os, re, hashlib, difflib, fitz
from fastapi import APIRouter, Request, Form, HTTPException, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from urllib.parse import quote, unquote_plus

from rag_pipeline.schema.rag_res_schema import RagStructuredResponse
from rag_pipeline.query_data_store import rag_query
from rag_pipeline.helpers import (
    STATIC_DIR, DATA_DIR, safe_basename,
    normalize_text, chunk_text, get_blocks_from_redis
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
#        PDF HIGHLIGHT
# ============================================================
@router.get("/highlighted_pdf", include_in_schema=False)
async def highlighted_pdf(request: Request, doc: str, page: int, highlight: str = ""):

    r_client = get_redis_client(request)

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
