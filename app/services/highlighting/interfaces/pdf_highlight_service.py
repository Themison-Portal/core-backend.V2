from abc import ABC, abstractmethod

class IPDFHightlightService:
    @abstractmethod
    async def get_highlighted_pdf(self, doc_url: str, page: int, bboxes: list[list[float]] | None = None,) -> bytes:
        """Standardizes a PDF page with coordinate-based highlights."""
        pass