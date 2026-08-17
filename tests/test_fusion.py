import pytest
from src.services.fusion import ReciprocalRankFusionEngine, DenseVectorStore
from src.schemas.document import DocumentResponse

def test_dense_vector_search(vector_store):
    hits = vector_store.search("How to configure PostgreSQL pool?", top_k=2)
    assert len(hits) == 2
    assert hits[0].score is not None
    assert hits[0].score > 0

def test_rrf_scoring_convergence(fusion_engine):
    batch1 = [
        DocumentResponse(id="doc_a", text="Postgres", metadata={}),
        DocumentResponse(id="doc_b", text="Redis", metadata={}),
    ]
    batch2 = [
        DocumentResponse(id="doc_b", text="Redis", metadata={}),
        DocumentResponse(id="doc_c", text="K8s", metadata={}),
    ]
    fused = fusion_engine.fuse_rankings([batch1, batch2], top_k=3)
    assert len(fused) == 3
    # doc_b appeared in both queries, so its accumulated RRF score must be strictly highest
    assert fused[0].id == "doc_b"
    assert fused[0].rrf_score > fused[1].rrf_score

def test_end_to_end_rag_query(fusion_engine):
    res = fusion_engine.execute_rag("Database caching solutions", top_k=2, num_queries=3)
    assert len(res["expanded_queries"]) == 3
    assert len(res["fused_results"]) == 2
    assert len(res["synthesized_context"]) > 0
