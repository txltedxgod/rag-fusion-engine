import pytest
from fastapi.testclient import TestClient

from src.core.exceptions import DocumentNotFoundError, InvalidQueryError
from src.main import app
from src.schemas.document import DocumentResponse
from src.services.fusion import CrossEncoderReranker, DenseVectorStore


def test_dense_vector_search(vector_store: DenseVectorStore) -> None:
    hits = vector_store.search("How to configure PostgreSQL pool?", top_k=2)
    assert len(hits) == 2
    assert hits[0].score is not None
    assert hits[0].score > 0
    assert "PostgreSQL" in hits[0].text
    assert hits[0].score > (hits[1].score or 0.0)


def test_dense_search_with_metadata_filters(vector_store: DenseVectorStore) -> None:
    hits = vector_store.search("container or database", top_k=5, filters={"category": "cache"})
    assert len(hits) == 1
    assert hits[0].metadata["category"] == "cache"
    assert "Redis" in hits[0].text


def test_bm25_sparse_search_and_delete_updates_stats(vector_store: DenseVectorStore) -> None:
    doc_id = vector_store.add(
        "Vector quantization reduces memory footprint during HNSW index traversal.",
        {"category": "retrieval"},
    )
    hits = vector_store.bm25_search("HNSW vector quantization", top_k=2)
    assert hits
    assert hits[0].id == doc_id
    assert (hits[0].score or 0.0) > 0.0

    vector_store.delete(doc_id)
    with pytest.raises(DocumentNotFoundError):
        vector_store.get(doc_id)

    after_delete = vector_store.bm25_search("HNSW quantization", top_k=2)
    assert all(hit.id != doc_id for hit in after_delete)


def test_rrf_scoring_convergence(fusion_engine) -> None:
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


def test_cross_encoder_reranker_prefers_exact_phrase_and_proximity() -> None:
    reranker = CrossEncoderReranker()
    loose = DocumentResponse(
        id="loose",
        text="Vector engines store embeddings for various workloads, and hybrid pipelines add sparse signals later.",
        metadata={},
        rrf_score=0.032,
    )
    tight = DocumentResponse(
        id="tight",
        text="A hybrid search pipeline fuses sparse BM25 scores with dense vectors.",
        metadata={},
        rrf_score=0.032,
    )
    reranked = reranker.rerank("hybrid search pipeline", [loose, tight], top_k=2)
    assert reranked[0].id == "tight"
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score > (reranked[1].rerank_score or 0.0)


def test_end_to_end_rag_query(fusion_engine) -> None:
    res = fusion_engine.execute_rag("Database caching solutions", top_k=2, num_queries=3)
    assert len(res["expanded_queries"]) == 3
    assert len(res["fused_results"]) == 2
    assert "Redis" in res["fused_results"][0].text
    assert res["fused_results"][0].rerank_score is not None
    assert len(res["synthesized_context"]) > 0


def test_empty_query_raises_invalid_query_error(vector_store: DenseVectorStore, fusion_engine) -> None:
    with pytest.raises(InvalidQueryError):
        vector_store.search("   ")
    with pytest.raises(InvalidQueryError):
        vector_store.bm25_search("   ")
    with pytest.raises(InvalidQueryError):
        fusion_engine.execute_rag("   ")


def test_api_document_crud_and_hybrid_search() -> None:
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    create_resp = client.post(
        "/api/v1/documents",
        json={
            "text": "HNSW graphs enable sub-linear approximate nearest neighbor search at scale.",
            "metadata": {"topic": "ann-index"},
        },
    )
    assert create_resp.status_code == 201
    doc_id = create_resp.json()["document_id"]

    get_resp = client.get(f"/api/v1/documents/{doc_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["metadata"] == {"topic": "ann-index"}

    search_resp = client.post(
        "/api/v1/search",
        json={
            "query": "approximate nearest neighbor HNSW",
            "top_k": 2,
            "num_queries": 3,
            "filters": {"topic": "ann-index"},
            "rerank": True,
        },
    )
    assert search_resp.status_code == 200
    payload = search_resp.json()
    assert len(payload["fused_results"]) == 1
    assert payload["fused_results"][0]["id"] == doc_id
    assert payload["fused_results"][0]["rrf_score"] > 0
    assert payload["fused_results"][0]["rerank_score"] > 0

    del_resp = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    missing_resp = client.get(f"/api/v1/documents/{doc_id}")
    assert missing_resp.status_code == 404
    assert missing_resp.json()["error"]["type"] == "DocumentNotFoundError"
