# System Architecture

The Omnibus Legal Compass is built as a modern Retrieval-Augmented Generation (RAG) system specifically optimized for the Indonesian legal landscape.

## Overview

```
Frontend (Next.js 16) → FastAPI Backend → RAG Chain → Qdrant Vector DB
                                        → Knowledge Graph (NetworkX)
                                        → Chat Session Manager
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI |
| Frontend | Next.js 16 |
| Vector Database | Qdrant |
| AI Inference | NVIDIA NIM (Kimi K2) |
| Orchestration | LangChain |
| Knowledge Graph | NetworkX |
| Styling | Tailwind CSS v4 |

## Data Flow

The system follows a sophisticated RAG pipeline to ensure high accuracy and traceability:

1.  **Query**: The user submits a legal question in Indonesian.
2.  **Hybrid Search**: The system performs a dual search in Qdrant:
    *   **BM25 (Sparse)**: For keyword matching (e.g., specific law numbers).
    *   **Dense**: For semantic meaning using embeddings.
3.  **Cross-Encoder Reranking**: Results are reranked to ensure the most relevant legal context is passed to the LLM.
4.  **LLM Generation**: The Kimi K2 model generates a response based *only* on the retrieved context.
5.  **Citations**: The system extracts precise source information (Law No, Year, Article) to provide verifiable citations.

## Embedding Model

We use the `paraphrase-multilingual-MiniLM-L12-v2` model, which provides 384-dimensional dense vectors. This model is specifically chosen for its excellent performance across multiple languages, including Indonesian.
