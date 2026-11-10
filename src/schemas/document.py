from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    """Payload for indexing a text document with optional metadata."""

    text: str = Field(..., min_length=5, max_length=50000, description="Raw text payload to index")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata dictionary"
    )


class DocumentResponse(BaseModel):
    """Indexed document representation with retrieval and fusion scores."""

    id: str
    text: str
    metadata: dict[str, Any]
    score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SearchQueryRequest(BaseModel):
    """Request parameters for hybrid multi-query RRF search and reranking."""

    query: str = Field(..., min_length=2, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=50)
    num_queries: int = Field(default=3, ge=1, le=6)
    filters: dict[str, Any] | None = Field(
        default=None, description="Optional exact-match metadata filter pairs"
    )
    rerank: bool = Field(
        default=True, description="Apply cross-encoder reranking after hybrid RRF fusion"
    )


class SearchQueryResponse(BaseModel):
    """Response containing query variants, fused/reranked hits, and context."""

    query: str
    expanded_queries: list[str]
    fused_results: list[DocumentResponse]
    synthesized_context: str
