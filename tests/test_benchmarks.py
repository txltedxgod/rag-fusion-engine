"""
RRF Benchmark & Stress Testing Suite
Evaluates throughput (QPS) and reciprocal rank fusion precision.
"""
import time
import pytest

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores = {}
    for rank_list in rankings:
        for rank, doc_id in enumerate(rank_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

def test_rrf_scoring_order():
    rankings = [
        ["docA", "docB", "docC"],
        ["docB", "docA", "docD"],
        ["docB", "docC", "docA"]
    ]
    results = reciprocal_rank_fusion(rankings, k=60)
    # docB was top in 2/3 lists and 2nd in 1 list -> highest score
    assert list(results.keys())[0] == "docB"

def test_rrf_throughput():
    sample_list = [f"doc_{i}" for i in range(100)]
    start = time.perf_counter()
    for _ in range(1000):
        reciprocal_rank_fusion([sample_list, sample_list[::-1]], k=60)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"RRF took {elapsed:.3f}s for 1000 iterations"
