# ⚡ RAG Fusion Engine

[![CI](https://github.com/txltedxgod/rag-fusion-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/txltedxgod/rag-fusion-engine/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**RAG Fusion Engine** is a high-accuracy Retrieval-Augmented Generation system in Python with multi-query expansion and Reciprocal Rank Fusion (RRF) scoring.

```
[User Query] ──> [ Query Expander (3 Perspectives) ]
                        │
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
   [Query 1]        [Query 2]        [Query 3]
       │                │                │
       ▼                ▼                ▼
   [Top 5 Hits]    [Top 5 Hits]    [Top 5 Hits]
       └────────────────┬────────────────┘
                        ▼
       [ Reciprocal Rank Fusion (RRF) ]
                        │
                        ▼
          [ Top Re-Ranked Documents ]
```

## 🚀 Quick Start
```bash
docker compose up -d
curl -X POST "http://localhost:8000/api/v1/search" -H "Content-Type: application/json" -d '{"query": "FastAPI async performance"}'
```

## 📄 License
MIT License
