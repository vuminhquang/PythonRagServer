from pydantic import BaseModel
from typing import List, Dict, Any

class Document(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class DocumentList(BaseModel):
    documents: List[Document]

class Query(BaseModel):
    text: str
    top_k: int = 3

class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult] 