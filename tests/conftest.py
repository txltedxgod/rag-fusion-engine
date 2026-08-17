import pytest
from src.services.fusion import DenseVectorStore, ReciprocalRankFusionEngine

@pytest.fixture
def vector_store():
    store = DenseVectorStore(dim=64)
    store.add("PostgreSQL connection pooling with asyncpg and SQLAlchemy 2.0", {"category": "db"})
    store.add("Redis in-memory caching strategies with TTL and LRU eviction", {"category": "cache"})
    store.add("Kubernetes horizontal pod autoscaler based on custom Prometheus metrics", {"category": "k8s"})
    return store

@pytest.fixture
def fusion_engine(vector_store):
    return ReciprocalRankFusionEngine(store=vector_store, rrf_k=60)
