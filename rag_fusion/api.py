from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from rag_fusion.store import VectorStore
from rag_fusion.pipeline import RAGFusionPipeline

app = FastAPI(
    title="RAG Fusion Engine",
    description="Advanced RAG with Reciprocal Rank Fusion & Multi-Query Expansion",
    version="0.1.0"
)

store = VectorStore()
# Preload knowledge
store.add_document("FastAPI provides automatic OpenAPI generation and async request handling in Python.")
store.add_document("Docker Compose simplifies local multi-container orchestration and environment reproducibility.")
store.add_document("Reciprocal Rank Fusion (RRF) aggregates ranked retrieval lists into an optimal fused ranking.")

pipeline = RAGFusionPipeline(store)


class IngestRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/api/v1/ingest")
def ingest_document(req: IngestRequest):
    doc_id = store.add_document(req.text, req.metadata)
    return {"status": "indexed", "id": doc_id}


@app.post("/api/v1/search")
def search_rag(req: SearchRequest):
    return pipeline.query(req.query, top_k=req.top_k)


@app.get("/health")
def health():
    return {"status": "healthy", "service": "rag-fusion-engine"}
