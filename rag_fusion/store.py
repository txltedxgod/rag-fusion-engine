from typing import List, Dict, Any, Optional
import numpy as np
import hashlib


class VectorStore:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.documents: List[Dict[str, Any]] = []

    def _embed(self, text: str) -> np.ndarray:
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        doc_id = f"doc_{len(self.documents) + 1}_{hashlib.sha256(text.encode()).hexdigest()[:6]}"
        self.documents.append({
            "id": doc_id,
            "text": text,
            "embedding": self._embed(text),
            "metadata": metadata or {}
        })
        return doc_id

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.documents:
            return []

        q_vec = self._embed(query)
        scored = []
        for doc in self.documents:
            sim = float(np.dot(q_vec, doc["embedding"]))
            scored.append({
                "id": doc["id"],
                "text": doc["text"],
                "score": sim,
                "metadata": doc["metadata"]
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
