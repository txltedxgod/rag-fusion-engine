from typing import Dict, Any, List
from rag_fusion.fusion import ReciprocalRankFusion, QueryExpander
from rag_fusion.store import VectorStore


class RAGFusionPipeline:
    def __init__(self, store: VectorStore, k: int = 60):
        self.store = store
        self.expander = QueryExpander()
        self.fusion = ReciprocalRankFusion(k=k)

    def query(self, user_prompt: str, top_k: int = 3) -> Dict[str, Any]:
        # 1. Expand into multi-queries
        queries = self.expander.expand(user_prompt, num_queries=3)

        # 2. Retrieve per-query
        all_results = []
        for q in queries:
            results = self.store.search(q, top_k=5)
            all_results.append(results)

        # 3. Fuse rankings via RRF
        fused_docs = self.fusion.fuse(all_results, top_n=top_k)

        # 4. Construct synthesized context
        context_str = "\n---\n".join([f"[{d['id']}] {d['text']}" for d in fused_docs])

        return {
            "query": user_prompt,
            "expanded_queries": queries,
            "fused_documents": fused_docs,
            "synthesized_context": context_str
        }
