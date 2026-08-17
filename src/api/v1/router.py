from fastapi import APIRouter, status
from src.schemas.document import DocumentCreate, SearchQueryRequest, SearchQueryResponse
from src.services.fusion import DenseVectorStore, ReciprocalRankFusionEngine
from src.core.config import settings

router = APIRouter(prefix="/api/v1", tags=["RAG Fusion"])

store = DenseVectorStore(dim=settings.vector_dim)
# Seed baseline documents
store.add("FastAPI provides automatic OpenAPI docs and native asyncio execution loop.", {"topic": "web-frameworks"})
store.add("Reciprocal Rank Fusion (RRF) combines multi-query search results without parameter tuning.", {"topic": "information-retrieval"})
store.add("Docker multi-stage builds optimize production images by pruning compile dependencies.", {"topic": "devops"})

engine = ReciprocalRankFusionEngine(store=store, rrf_k=settings.rrf_k)

@router.post("/documents", status_code=status.HTTP_201_CREATED)
def index_document(payload: DocumentCreate):
    doc_id = store.add(payload.text, payload.metadata)
    return {"status": "indexed", "document_id": doc_id}

@router.post("/search", response_model=SearchQueryResponse)
def execute_search(payload: SearchQueryRequest):
    return engine.execute_rag(payload.query, top_k=payload.top_k, num_queries=payload.num_queries)
