import hashlib
import math
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from src.core.exceptions import DocumentNotFoundError, InvalidQueryError
from src.schemas.document import DocumentResponse

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens from text."""
    return _TOKEN_RE.findall(text.lower())


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Return True when document metadata satisfies all exact-match filter constraints."""
    if not filters:
        return True
    return all(metadata.get(key) == value for key, value in filters.items())


def _min_cover_span(doc_tokens: list[str], target_terms: set[str]) -> int:
    """Find the shortest token window covering all target terms in doc_tokens."""
    last_seen: dict[str, int] = {}
    best_span = len(doc_tokens)
    target_count = len(target_terms)

    for idx, token in enumerate(doc_tokens):
        if token not in target_terms:
            continue
        last_seen[token] = idx
        if len(last_seen) == target_count:
            span = idx - min(last_seen.values()) + 1
            if span < best_span:
                best_span = span
    return max(best_span, target_count)


class DenseVectorStore:
    """In-memory hybrid document store supporting hashed dense vectors and BM25."""

    def __init__(self, dim: int = 128) -> None:
        if dim <= 0:
            raise ValueError("Vector dimension must be positive")
        self.dim = dim
        self._store: dict[str, dict[str, Any]] = {}
        self._doc_freqs: Counter[str] = Counter()
        self._total_tokens: int = 0

    def _bucket_index(self, feature: str) -> int:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "little") % self.dim

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Build an L2-normalized dense embedding using word and char-trigram hashing."""
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _tokenize(text)

        if tokens:
            counts = Counter(tokens)
            for token, freq in counts.items():
                weight = 1.0 + math.log(freq)
                vec[self._bucket_index(f"w:{token}")] += weight
                if len(token) >= 3:
                    padded = f"^{token}$"
                    trigram_weight = 0.35 * weight
                    for i in range(len(padded) - 2):
                        trigram = padded[i : i + 3]
                        vec[self._bucket_index(f"c:{trigram}")] += trigram_weight
        else:
            cleaned = text.lower().strip()
            for i in range(max(0, len(cleaned) - 2)):
                vec[self._bucket_index(f"raw:{cleaned[i : i + 3]}")] += 1.0

        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0.0 else vec

    def _remove_doc_stats(self, doc_id: str) -> None:
        existing = self._store.get(doc_id)
        if not existing:
            return
        tokens: list[str] = existing["tokens"]
        self._total_tokens -= len(tokens)
        for term in set(tokens):
            self._doc_freqs[term] -= 1
            if self._doc_freqs[term] <= 0:
                del self._doc_freqs[term]

    def idf_map(self) -> dict[str, float]:
        """Compute BM25+ smoothed IDF weights for every indexed term."""
        n_docs = len(self._store)
        if n_docs == 0:
            return {}
        return {
            term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in self._doc_freqs.items()
        }

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Index or update a document and return its deterministic identifier."""
        doc_id = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
        if doc_id in self._store:
            self._remove_doc_stats(doc_id)

        tokens = _tokenize(text)
        self._total_tokens += len(tokens)
        for term in set(tokens):
            self._doc_freqs[term] += 1

        self._store[doc_id] = {
            "id": doc_id,
            "text": text,
            "metadata": dict(metadata) if metadata else {},
            "embedding": self._generate_embedding(text),
            "tokens": tokens,
            "term_freqs": Counter(tokens),
        }
        return doc_id

    def get(self, doc_id: str) -> dict[str, Any]:
        """Retrieve a stored document record by id or raise DocumentNotFoundError."""
        if doc_id not in self._store:
            raise DocumentNotFoundError(doc_id)
        return self._store[doc_id]

    def delete(self, doc_id: str) -> None:
        """Delete a stored document and update corpus term statistics."""
        if doc_id not in self._store:
            raise DocumentNotFoundError(doc_id)
        self._remove_doc_stats(doc_id)
        del self._store[doc_id]

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[DocumentResponse]:
        """Rank documents by cosine similarity over dense feature-hashed embeddings."""
        if not query or not query.strip():
            raise InvalidQueryError("Query must not be empty")
        if not self._store or top_k <= 0:
            return []

        q_emb = self._generate_embedding(query)
        scored: list[DocumentResponse] = []
        for doc_id, item in self._store.items():
            if not _matches_filters(item["metadata"], filters):
                continue
            sim = float(np.dot(q_emb, item["embedding"]))
            scored.append(
                DocumentResponse(
                    id=doc_id,
                    text=item["text"],
                    metadata=dict(item["metadata"]),
                    score=round(sim, 4),
                )
            )
        scored.sort(key=lambda x: x.score or 0.0, reverse=True)
        return scored[:top_k]

    def bm25_search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[DocumentResponse]:
        """Rank matching documents using Okapi BM25+ lexical relevance."""
        if not query or not query.strip():
            raise InvalidQueryError("Query must not be empty")
        if not self._store or top_k <= 0:
            return []

        query_terms = list(dict.fromkeys(_tokenize(query)))
        if not query_terms:
            return []

        n_docs = len(self._store)
        avgdl = self._total_tokens / n_docs if self._total_tokens > 0 else 1.0
        idfs = self.idf_map()

        scored: list[DocumentResponse] = []
        for doc_id, item in self._store.items():
            if not _matches_filters(item["metadata"], filters):
                continue
            term_freqs: Counter[str] = item["term_freqs"]
            doc_len = max(len(item["tokens"]), 1)
            norm_len = 1.0 - b + b * (doc_len / avgdl)

            bm25_score = 0.0
            for term in query_terms:
                tf = term_freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = idfs.get(term, 0.0)
                bm25_score += idf * (tf * (k1 + 1.0)) / (tf + k1 * norm_len)

            if bm25_score > 0.0:
                scored.append(
                    DocumentResponse(
                        id=doc_id,
                        text=item["text"],
                        metadata=dict(item["metadata"]),
                        score=round(bm25_score, 4),
                    )
                )

        scored.sort(key=lambda x: x.score or 0.0, reverse=True)
        return scored[:top_k]


class CrossEncoderReranker:
    """Cross-interaction reranker combining IDF coverage, bigram match, and span proximity."""

    def score_pair(
        self,
        query: str,
        document_text: str,
        idf_weights: dict[str, float] | None = None,
    ) -> float:
        """Compute a normalized [0, 1] relevance score for a query-document pair."""
        q_tokens = _tokenize(query)
        d_tokens = _tokenize(document_text)
        if not q_tokens or not d_tokens:
            return 0.0

        unique_q = list(dict.fromkeys(q_tokens))
        doc_token_set = set(d_tokens)
        weights = idf_weights or {}

        total_idf = sum(weights.get(t, 1.0) for t in unique_q)
        matched_terms = [t for t in unique_q if t in doc_token_set]
        if not matched_terms or total_idf <= 0.0:
            return 0.0

        coverage = sum(weights.get(t, 1.0) for t in matched_terms) / total_idf

        normalized_q = " ".join(q_tokens)
        normalized_d = " ".join(d_tokens)
        if len(q_tokens) == 1:
            phrase_signal = 1.0
        elif normalized_q in normalized_d:
            phrase_signal = 1.0
        else:
            q_bigrams = {(q_tokens[i], q_tokens[i + 1]) for i in range(len(q_tokens) - 1)}
            d_bigrams = {(d_tokens[i], d_tokens[i + 1]) for i in range(len(d_tokens) - 1)}
            phrase_signal = len(q_bigrams & d_bigrams) / len(q_bigrams) if q_bigrams else 0.0

        if len(matched_terms) == 1:
            proximity_signal = 0.5
        else:
            best_span = _min_cover_span(d_tokens, set(matched_terms))
            proximity_signal = len(matched_terms) / best_span

        return 0.50 * coverage + 0.30 * phrase_signal + 0.20 * proximity_signal

    def rerank(
        self,
        query: str,
        documents: list[DocumentResponse],
        top_k: int | None = None,
        idf_weights: dict[str, float] | None = None,
    ) -> list[DocumentResponse]:
        """Rerank fused candidates using cross-interaction scoring while retaining RRF prior."""
        if not documents:
            return []

        max_rrf = max((doc.rrf_score or 0.0) for doc in documents) or 1.0
        reranked: list[DocumentResponse] = []
        for doc in documents:
            pair_score = self.score_pair(query, doc.text, idf_weights=idf_weights)
            rrf_prior = (doc.rrf_score or 0.0) / max_rrf
            blended = 0.85 * pair_score + 0.15 * rrf_prior

            updated = doc.model_copy(deep=True)
            updated.rerank_score = round(blended, 6)
            reranked.append(updated)

        reranked.sort(key=lambda d: (d.rerank_score or 0.0, d.rrf_score or 0.0), reverse=True)
        return reranked[:top_k] if top_k is not None else reranked


class ReciprocalRankFusionEngine:
    """Multi-query hybrid retrieval engine combining dense, BM25, RRF, and reranking."""

    def __init__(
        self,
        store: DenseVectorStore,
        rrf_k: int = 60,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.store = store
        self.rrf_k = rrf_k
        self.reranker = reranker or CrossEncoderReranker()

    def expand_queries(self, prompt: str, count: int = 3) -> list[str]:
        """Generate deterministic multi-angle query expansions."""
        if count <= 0:
            return []
        variants = [
            prompt,
            f"{prompt} architectural overview and key concepts",
            f"how to implement {prompt} step by step guide",
            f"{prompt} best practices and common pitfalls",
            f"{prompt} performance tuning and scalability trade-offs",
            f"{prompt} troubleshooting failure modes and edge cases",
        ]
        return variants[:count]

    def fuse_rankings(
        self,
        ranked_batches: list[list[DocumentResponse]],
        top_k: int = 3,
    ) -> list[DocumentResponse]:
        """Combine ranked document batches using Reciprocal Rank Fusion."""
        if top_k <= 0:
            return []

        scores: dict[str, float] = defaultdict(float)
        docs_map: dict[str, DocumentResponse] = {}

        for batch in ranked_batches:
            for rank, doc in enumerate(batch):
                if doc.id not in docs_map:
                    docs_map[doc.id] = doc
                scores[doc.id] += 1.0 / (self.rrf_k + rank + 1)

        sorted_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        fused: list[DocumentResponse] = []
        for doc_id in sorted_ids[:top_k]:
            doc = docs_map[doc_id].model_copy(deep=True)
            doc.rrf_score = round(scores[doc_id], 6)
            fused.append(doc)
        return fused

    def execute_rag(
        self,
        query: str,
        top_k: int = 3,
        num_queries: int = 3,
        filters: dict[str, Any] | None = None,
        rerank: bool = True,
    ) -> dict[str, Any]:
        """Run query expansion, hybrid dense+BM25 RRF fusion, optional reranking, and synthesis."""
        if not query or not query.strip():
            raise InvalidQueryError("Query must not be empty")
        if top_k < 1:
            raise InvalidQueryError("top_k must be at least 1")
        if num_queries < 1:
            raise InvalidQueryError("num_queries must be at least 1")

        expanded = self.expand_queries(query, count=num_queries)
        per_batch_k = max(5, top_k * 2)

        dense_batches = [
            self.store.search(variant, top_k=per_batch_k, filters=filters) for variant in expanded
        ]
        bm25_batches = [
            self.store.bm25_search(variant, top_k=per_batch_k, filters=filters)
            for variant in expanded
        ]
        all_batches = [batch for batch in (*dense_batches, *bm25_batches) if batch]

        candidate_k = max(top_k * 2, top_k) if rerank else top_k
        fused = self.fuse_rankings(all_batches, top_k=candidate_k)

        if rerank and fused:
            results = self.reranker.rerank(
                query=query,
                documents=fused,
                top_k=top_k,
                idf_weights=self.store.idf_map(),
            )
        else:
            results = fused[:top_k]

        context = "\n---\n".join(f"[{doc.id}] {doc.text}" for doc in results)
        return {
            "query": query,
            "expanded_queries": expanded,
            "fused_results": results,
            "synthesized_context": context,
        }
