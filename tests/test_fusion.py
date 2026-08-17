import pytest
from rag_fusion.fusion import ReciprocalRankFusion, QueryExpander
from rag_fusion.store import VectorStore
from rag_fusion.pipeline import RAGFusionPipeline


def test_rrf_ranking():
    fusion = ReciprocalRankFusion(k=60)
    
    list1 = [
        {"id": "doc1", "text": "Doc 1"},
        {"id": "doc2", "text": "Doc 2"},
    ]
    list2 = [
        {"id": "doc2", "text": "Doc 2"},
        {"id": "doc3", "text": "Doc 3"},
    ]

    fused = fusion.fuse([list1, list2], top_n=3)
    assert len(fused) == 3
    # doc2 appeared in both lists, so its accumulated RRF score must be highest
    assert fused[0]["id"] == "doc2"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_rag_pipeline_end_to_end():
    store = VectorStore()
    store.add_document("Python FastAPI is super fast for building REST APIs")
    store.add_document("Docker Compose organizes containers seamlessly")

    pipeline = RAGFusionPipeline(store)
    res = pipeline.query("How to build APIs with FastAPI?")

    assert "fused_documents" in res
    assert len(res["fused_documents"]) > 0
    assert len(res["expanded_queries"]) == 3
