from pydantic import BaseModel
from typing import List

class RagSource(BaseModel):
    section: str
    page: int
    filename: str
    exactText: str
    chunk_index: int
    relevance: str
    context: str
    highlightURL: str

class RagStructuredResponse(BaseModel):
    response: str  # main answer text, markdown-safe
    sources: List[RagSource]
    class Config:
        extra = "forbid"
