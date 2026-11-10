from fastapi import APIRouter, status

from src.core.config import settings
from src.schemas.document import (
    DocumentCreate,
    DocumentResponse,
    SearchQueryRequest,
    SearchQueryResponse,
)
from src.services.fusion import DenseVectorStore, ReciprocalRankFusionEngine

router = APIRouter(prefix="/api/v1", tags=["RAG Fusion"])

store = DenseVectorStore(dim=settings.vector_dim)
store.add(
    "FastAPI provides automatic OpenAPI docs and native asyncio execution loop.",
    {"topic": "web-frameworks"},
)
store.add(
    "Reciprocal Rank Fusion (RRF) combines multi-query search results without parameter tuning.",
    {"topic": "information-retrieval"},
)
store.add(
    "Docker multi-stage builds optimize production images by pruning compile dependencies.",
    {"topic": "devops"},
)

engine = ReciprocalRankFusionEngine(store=store, rrf_k=settings.rrf_k)


@router.post("/documents", status_code=status.HTTP_201_CREATED)
def index_document(payload: DocumentCreate) -> dict[str, str]:
    """Index a document into the hybrid dense and BM25 store."""
    doc_id = store.add(payload.text, payload.metadata)
    return {"status": "indexed", "document_id": doc_id}


@router.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: str) -> DocumentResponse:
    """Fetch a single indexed document by identifier."""
    record = store.get(doc_id)
    return DocumentResponse(
        id=record["id"],
        text=record["text"],
        metadata=record["metadata"],
    )


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, str]:
    """Delete an indexed document and update corpus term statistics."""
    store.delete(doc_id)
    return {"status": "deleted", "document_id": doc_id}


@router.post("/search", response_model=SearchQueryResponse)
def execute_search(payload: SearchQueryRequest) -> dict[str, object]:
    """Execute hybrid multi-query RRF retrieval and optional reranking."""
    return engine.execute_rag(
        payload.query,
        top_k=payload.top_k,
        num_queries=payload.num_queries,
        filters=payload.filters,
        rerank=payload.rerank,
    )
