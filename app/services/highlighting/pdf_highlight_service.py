import fitz, httpx, hashlib, json

from fastapi import HTTPException
from app.services.highlighting.interfaces.pdf_highlight_service import IPDFHightlightService

class PDFHighlightService(IPDFHightlightService):
    def __init__(self, redis):
        self.redis = redis
        

    async def _get_pdf_from_url(self, url: str) -> fitz.Document:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            pdf_bytes = resp.content
            # Open PDF from bytes
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            return pdf_doc
        

    async def get_highlighted_pdf(
    self,
    doc_url: str,
    page: int,
    bboxes: list[list[float]],
    ) -> bytes:
        

        # 1️⃣ Build deterministic cache key
        bboxes_hash = hashlib.sha1(
            json.dumps(bboxes, sort_keys=True).encode()
        ).hexdigest()[:10]

        cache_key = (
            f"pdf_hl:{hashlib.sha1(doc_url.encode()).hexdigest()[:10]}"
            f":p{page}:{bboxes_hash}"
        )

        # 2️⃣ Redis cache
        cached = await self.redis.get(cache_key)
        if cached:
            return cached

        # 3️⃣ Load PDF
        doc_pdf = await self._get_pdf_from_url(doc_url)

        try:
            if page < 1 or page > len(doc_pdf):
                raise ValueError(f"Page {page} out of range")

            page_obj = doc_pdf[page - 1]
            page_height = page_obj.rect.height

            if not bboxes:
                raise ValueError("No bboxes provided for highlighting")

            # 4️⃣ Highlight ALL bboxes
            for bbox in bboxes:
                if not bbox or len(bbox) != 4:
                    continue

                x0, y0, x1, y1 = map(float, bbox)

                # Normalize bbox
                x0, x1 = sorted([x0, x1])
                y0, y1 = sorted([y0, y1])

                # Convert Docling (top-left) → PDF (bottom-left)
                target_rect = fitz.Rect(
                    x0,
                    page_height - y1,
                    x1,
                    page_height - y0,
                )

                if target_rect.is_empty or target_rect.is_infinite:
                    continue

                # Smart highlight: expand to text blocks if overlapping
                blocks = page_obj.get_text("blocks")
                intersecting_blocks = [
                    fitz.Rect(b[:4])
                    for b in blocks
                    if target_rect.intersects(fitz.Rect(b[:4]))
                ]

                if intersecting_blocks:
                    for block_rect in intersecting_blocks:
                        annot = page_obj.add_highlight_annot(block_rect)
                        annot.update()
                else:
                    annot = page_obj.add_highlight_annot(target_rect)
                    annot.update()

            # 5️⃣ Serialize and cache
            pdf_bytes = doc_pdf.tobytes(garbage=3, clean=True, deflate=True)
            await self.redis.set(cache_key, pdf_bytes, ex=3600)

            return pdf_bytes

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Error generating highlighted PDF: {str(e)}"
            )

        finally:
            doc_pdf.close()
