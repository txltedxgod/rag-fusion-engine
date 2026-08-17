import numpy as np
import hashlib
from collections import defaultdict
from typing import List, Dict, Any, Optional
from src.schemas.document import DocumentResponse
from src.core.exceptions import DocumentNotFoundError

class DenseVectorStore:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self._store: Dict[str, Dict[str, Any]] = {}

    def _generate_embedding(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        doc_id = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
        emb = self._generate_embedding(text)
        self._store[doc_id] = {
            "id": doc_id,
            "text": text,
            "metadata": metadata or {},
            "embedding": emb
        }
        return doc_id

    def get(self, doc_id: str) -> Dict[str, Any]:
        if doc_id not in self._store:
            raise DocumentNotFoundError(doc_id)
        return self._store[doc_id]

    def search(self, query: str, top_k: int = 5) -> List[DocumentResponse]:
        if not self._store:
            return []
        q_emb = self._generate_embedding(query)
        scored = []
        for doc_id, item in self._store.items():
            sim = float(np.dot(q_emb, item["embedding"]))
            scored.append(DocumentResponse(
                id=doc_id,
                text=item["text"],
                metadata=item["metadata"],
                score=round(sim, 4)
            ))
        scored.sort(key=lambda x: x.score or 0.0, reverse=True)
        return scored[:top_k]

class ReciprocalRankFusionEngine:
    def __init__(self, store: DenseVectorStore, rrf_k: int = 60):
        self.store = store
        self.rrf_k = rrf_k

    def expand_queries(self, prompt: str, count: int = 3) -> List[str]:
        variants = [
            prompt,
            f"{prompt} architectural overview and key concepts",
            f"how to implement {prompt} step by step guide",
            f"{prompt} best practices and common pitfalls"
        ]
        return variants[:count]

    def fuse_rankings(self, ranked_batches: List[List[DocumentResponse]], top_k: int = 3) -> List[DocumentResponse]:
        scores: Dict[str, float] = defaultdict(float)
        docs_map: Dict[str, DocumentResponse] = {}

        for batch in ranked_batches:
            for rank, doc in enumerate(batch):
                docs_map[doc.id] = doc
                scores[doc.id] += 1.0 / (self.rrf_k + (rank + 1))

        sorted_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        out = []
        for d_id in sorted_ids[:top_k]:
            doc = docs_map[d_id].model_copy(deep=True)
            doc.rrf_score = round(scores[d_id], 6)
            out.append(doc)
        return out

    def execute_rag(self, query: str, top_k: int = 3, num_queries: int = 3) -> Dict[str, Any]:
        expanded = self.expand_queries(query, count=num_queries)
        batches = [self.store.search(q, top_k=5) for q in expanded]
        fused = self.fuse_rankings(batches, top_k=top_k)
        context = "
---
".join([f"[{d.id}] {d.text}" for d in fused])
        return {
            "query": query,
            "expanded_queries": expanded,
            "fused_results": fused,
            "synthesized_context": context
        }
