"""
RAG Fusion Engine - Quickstart Example
Demonstrates reciprocal rank fusion (RRF) calculation.
"""
import json

def reciprocal_rank_fusion(query_rankings: dict[str, list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for query, doc_ids in query_rankings.items():
        for rank, doc_id in enumerate(doc_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

if __name__ == "__main__":
    sample_retrievals = {
        "perspective_1": ["doc_A", "doc_B", "doc_C"],
        "perspective_2": ["doc_B", "doc_A", "doc_D"],
        "perspective_3": ["doc_B", "doc_E", "doc_A"]
    }
    fused = reciprocal_rank_fusion(sample_retrievals, k=60)
    print("=== RRF FUSED RANKING ===")
    print(json.dumps(fused, indent=2))
