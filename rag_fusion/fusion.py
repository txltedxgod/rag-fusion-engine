from typing import List, Dict, Any
import numpy as np
from collections import defaultdict


class ReciprocalRankFusion:
    """
    Implements Reciprocal Rank Fusion (RRF) to combine multiple retrieval results
    into a single re-ranked list based on reciprocal rank scoring: RRF(d) = sum(1 / (k + rank(d))).
    """
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, ranked_result_lists: List[List[Dict[str, Any]]], top_n: int = 5) -> List[Dict[str, Any]]:
        scores = defaultdict(float)
        doc_store = {}

        for result_list in ranked_result_lists:
            for rank, doc in enumerate(result_list):
                doc_id = doc["id"]
                doc_store[doc_id] = doc
                # RRF Formula
                scores[doc_id] += 1.0 / (self.k + (rank + 1))

        # Sort by accumulated RRF score
        sorted_doc_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)

        results = []
        for doc_id in sorted_doc_ids[:top_n]:
            doc = doc_store[doc_id].copy()
            doc["rrf_score"] = round(scores[doc_id], 5)
            results.append(doc)

        return results


class QueryExpander:
    """Generates varied perspective queries from an initial prompt."""
    def expand(self, query: str, num_queries: int = 3) -> List[str]:
        perspectives = [
            f"{query} architectural overview and key concepts",
            f"how to implement {query} step by step guide",
            f"{query} best practices and common pitfalls",
        ]
        return [query] + perspectives[:num_queries - 1]
