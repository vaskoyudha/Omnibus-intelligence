# Competitive Comparison

How does **Omnibus Legal Compass** stack up against other open-source legal AI projects?

## Overview

The legal AI landscape is dominated by Chinese-language projects (LaWGPT, Lawyer LLaMA, Fuzi.Mingcha) focused on fine-tuning LLMs on legal corpora. For other jurisdictions — particularly Southeast Asia — there are virtually no production-ready open-source alternatives. Omnibus Legal Compass fills this gap for Indonesia.

## Feature-by-Feature Comparison

| Feature | Omnibus Legal Compass | LaWGPT (China) | Lawyer LLaMA (China) | Fuzi.Mingcha (China) | AI Legal Compliance (USA) |
|---|:---:|:---:|:---:|:---:|:---:|
| **GitHub Stars** | New | 7.1k | 980 | 365 | 311 |
| **Jurisdiction** | Indonesia | China | China | China | USA |
| **Approach** | RAG + Hybrid Search | Fine-tuned LLM | Fine-tuned LLaMA | Fine-tuned ChatGLM | RAG + keyword search |
| **Frontend UI** | Next.js 16 + Tailwind | None (API only) | None (CLI) | Gradio demo | Streamlit |
| **Hybrid Search** (BM25 + Dense) | Yes | No | No | No | No |
| **CrossEncoder Reranking** | Yes | No | No | No | No |
| **Knowledge Graph** | Yes (NetworkX) | No | No | No | No |
| **Multi-Turn Chat** | Yes (session memory) | No | No | No | No |
| **Compliance Dashboard** | Yes (heat map + charts) | No | No | No | No |
| **Source Citations** | Every response | No | No | Partial | Partial |
| **Streaming Responses** | Yes (SSE) | N/A | N/A | No | No |
| **API Versioning** | `/api/v1/*` | No | No | No | No |
| **Rate Limiting** | Yes (slowapi) | No | No | No | No |
| **Test Coverage** | 294+ tests (91%) | Minimal | Minimal | None | None |
| **CI/CD** | GitHub Actions | No | No | No | No |
| **Documentation Site** | VitePress (7 pages) | Minimal | README only | README only | README only |
| **Issue/PR Templates** | Yes (3 templates) | No | No | No | No |
| **Security Policy** | Yes (SECURITY.md) | No | No | No | No |
| **License** | MIT | Apache-2.0 | GPL-3.0 | Apache-2.0 | MIT |

## Why RAG Over Fine-Tuning?

Most Chinese legal AI projects take the **fine-tuning** approach: training a base model on millions of legal documents. This requires:

- Massive datasets (100k+ documents)
- Expensive GPU training (A100 clusters)
- Regular retraining when laws change
- Risk of hallucination from memorized patterns

Omnibus uses **Retrieval-Augmented Generation (RAG)** instead:

- **Always up-to-date** — Add new regulations without retraining
- **Verifiable** — Every answer cites its source documents
- **Cost-effective** — Uses API inference (NVIDIA NIM free tier)
- **Transparent** — You can see exactly which documents were retrieved

## What Makes Omnibus Unique

### 1. Hybrid Search Pipeline

Most legal search systems use either keyword matching OR semantic search. Omnibus combines both:

1. **BM25 sparse retrieval** — Catches exact legal terminology (e.g., "Pasal 21 ayat 3")
2. **Dense vector search** — Understands semantic meaning across languages
3. **CrossEncoder reranking** — Refines results by modeling query-document interaction

This three-stage pipeline significantly outperforms single-method approaches for legal text.

### 2. Knowledge Graph

Legal documents don't exist in isolation. Our knowledge graph captures:

- **Document hierarchy** — Laws > Government Regulations > Presidential Regulations
- **Cross-references** — Which regulations cite or amend others
- **Temporal relationships** — Which regulations supersede previous ones
- **Structural breakdown** — Chapters, articles, and clauses within each document

### 3. Production Infrastructure

Unlike most academic legal AI projects, Omnibus ships with production essentials:

- Rate limiting (slowapi) to prevent abuse
- API versioning for backward compatibility
- Comprehensive test suite (294 tests, 91% coverage)
- CI/CD pipeline with automated testing
- Security policy and pre-commit secret detection
- Issue and PR templates for community contributions

### 4. Full-Stack Application

Most legal AI projects provide only a model or API. Omnibus includes a complete, production-quality frontend with:

- 6 specialized pages (Q&A, Compliance, Guidance, Chat, Knowledge Graph, Dashboard)
- Dark theme with professional UI (Tailwind CSS + Framer Motion)
- Responsive design for desktop and mobile
- Real-time streaming responses
- Interactive data visualizations (Recharts)

## Competitor Details

### LaWGPT (China, 7.1k stars)
Fine-tuned LLM for Chinese legal domain. Strong model quality but no retrieval pipeline, no frontend, no citation tracking. Academic project without production infrastructure.

### Lawyer LLaMA (China, 980 stars)
Fine-tuned LLaMA model on Chinese legal data. CLI-only interface. No search pipeline, no graph, no tests. Research-focused.

### Fuzi.Mingcha (China, 365 stars)
ChatGLM fine-tune with Gradio demo. Basic Q&A only — no compliance checking, no knowledge graph, no test suite.

### AI Legal Compliance (USA, 311 stars)
RAG-based compliance tool using keyword search. Streamlit frontend. No hybrid search, no reranking, no multi-turn chat, no tests.

## Summary

Omnibus Legal Compass is the **only Indonesian-focused legal AI** and the **only legal AI project** (in any jurisdiction) that combines:

- Hybrid search with reranking
- Knowledge graph
- Multi-turn conversational interface
- Visual compliance dashboard
- Full-stack production architecture with 294+ tests

If you're building legal AI for Southeast Asia — or want to learn how a production RAG system is built — Omnibus is the reference implementation.
