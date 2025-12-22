from pydantic import BaseModel
from typing import List, Optional, Literal


class RagSource(BaseModel):
    name: str
    page: int
    section: Optional[str] = None
    exactText: str  
    bboxes: List[List[float]]
    relevance: Literal["high", "medium", "low"]

class DoclingRagStructuredResponse(BaseModel):
    response: str
    sources: List[RagSource]

    class Config:
        extra = "forbid"
