from pydantic import BaseModel
from typing import List, Dict, Any, Optional



# Request model
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 4  # number of sources to return

# Response model
class Source(BaseModel):
    content: str
    metadata: Dict[str, Any]
    similarity_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: str  # "high", "medium", "low"