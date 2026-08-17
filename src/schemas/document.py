from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime

class DocumentCreate(BaseModel):
    text: str = Field(..., min_length=5, max_length=50000, description="Raw text payload to index")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata dictionary")

class DocumentResponse(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    score: Optional[float] = None
    rrf_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=50)
    num_queries: int = Field(default=3, ge=1, le=6)

class SearchQueryResponse(BaseModel):
    query: str
    expanded_queries: list[str]
    fused_results: list[DocumentResponse]
    synthesized_context: str
