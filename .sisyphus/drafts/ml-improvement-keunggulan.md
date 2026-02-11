# Draft: ML Improvement & Keunggulan (Competitive Advantages)

## Current System Assessment

### Architecture (What Exists Today)
- **LLM**: Kimi K2 via NVIDIA NIM (moonshotai/kimi-k2-instruct)
- **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2 (384 dim) — generic multilingual, NOT legal-specific
- **Vector DB**: Qdrant with Cosine similarity
- **Retrieval**: Hybrid Search (Dense + BM25 with RRF fusion, k=60)
- **Re-ranking**: CrossEncoder (mmarco-mMiniLMv2-L12-H384-v1) — multilingual but not legal-tuned
- **Query Expansion**: Rule-based synonym groups for Indonesian legal abbreviations (~29 groups)
- **Chunking**: 500-char chunks, 100-char overlap, sentence-boundary splitting
- **Confidence**: Multi-factor scoring (retrieval scores 40%, authority hierarchy 20%, consistency 20%, count 20%)
- **Validation**: Citation accuracy checking + hallucination risk detection
- **Streaming**: SSE-based real-time answer generation
- **Data**: Only 10 sample legal documents (VERY small)

### Frontend Features
- Legal Q&A with streaming answers
- Business formation guidance
- Compliance checker (text + PDF)
- Dark mode, citation transparency, confidence scoring
- Indonesian localization

### Identified Weaknesses
1. **Tiny dataset** — only 10 documents, production needs thousands
2. **No domain fine-tuning** — generic models, not legal-specialized
3. **Basic chunking** — no semantic awareness, legal structure not preserved
4. **No temporal reasoning** — can't handle law amendments/superseded provisions
5. **No knowledge graph** — no structured legal relationships
6. **Missing compliance page** in frontend (navigation links but page doesn't exist)
7. **No evaluation framework** — can't measure quality systematically
8. **No authentication/rate limiting** — security gaps
9. **No search history** — no user context persistence

## Research Findings (In Progress)
- Competitive analysis: PENDING (librarian agent running)
- Test infrastructure: PENDING (explore agent running)

## Requirements (confirmed)
- User wants ML improvements
- User wants competitive comparison to convince others
- User wants ideas for making it better
- User wants "keunggulan" — competitive advantages / unique selling points
- **Target**: Academic/University (thesis/research prototype)
- **Priority**: ALL — ML + Killer Features + UI/UX polish
- **Context**: Research Prototype (needs metrics, benchmarks, novel contributions)

## Academic Keunggulan Strategy
For academic evaluation, "keunggulan" must be:
1. **Quantifiable** — retrieval accuracy metrics (Precision@K, NDCG, MRR)
2. **Novel** — something existing tools DON'T have
3. **Documented** — comparison tables with competitors
4. **Reproducible** — evaluation framework others can run

### Proposed Unique Contributions (Keunggulan)
1. **Hierarchical Legal Chunking** — semantic chunking that respects Bab→Pasal→Ayat structure (no other tool does this for Indonesian law)
2. **Legal Authority-Weighted Confidence** — the existing multi-factor scoring is unique, needs to be formalized with metrics
3. **Hybrid Search + CrossEncoder + Query Expansion** pipeline — 3-stage retrieval is genuinely advanced
4. **Indonesian Legal NLP** — Sastrawi integration, legal NER, abbreviation expansion
5. **Regulatory Harmonization Check** — cross-reference regulations for conflicts/alignment (the "harmonization" in the project name)
6. **Evaluation Framework** — RAGAS-based metrics with legal expert validation dataset

## Confirmed Decisions
- **Timeline**: Extended (2+ months) — can do comprehensive overhaul
- **Purpose**: Personal portfolio project — wants to be portfolio-impressive, not thesis-graded
- **Keunggulan Focus**: All Three Combined — novel ML + metrics + unique features
- **Test Infra**: pytest exists with 4 test files (ingestion, config, retriever, data_loader)
- **Dataset**: Actually expanded to 200 documents (not just 10 as initially appeared)
- **Deployment**: Vercel (frontend) + Render (backend) + Qdrant Cloud

## Key Insight
The system is more mature than it first appears. The 200-document expanded dataset, 
existing CrossEncoder reranking, and hybrid search pipeline give us a strong base.
What's missing for "keunggulan" are the NOVEL contributions that make this stand out.

## Proposed Keunggulan Architecture

### 1. HIERARCHICAL LEGAL CHUNKING (Novel Contribution #1)
- Respect Bab → Pasal → Ayat structure 
- Context-aware: each chunk carries parent context (which law, which chapter)
- No other Indonesian legal AI tool does this

### 2. REGULATORY HARMONIZATION ENGINE (Novel Contribution #2) — THE NAMESAKE FEATURE
- Cross-reference regulations to detect conflicts/overlaps
- Example: "Does PP 5/2021 conflict with UU 11/2020?"
- Build a regulation relationship graph
- This is literally what the project is named for but doesn't exist yet!

### 3. RAGAS EVALUATION FRAMEWORK (Novel Contribution #3)
- Quantitative metrics: Precision@K, NDCG, MRR, Faithfulness, Answer Relevancy
- Create a gold-standard Q&A dataset for Indonesian legal domain
- Enable reproducible benchmarking

### 4. ADVANCED INDONESIAN LEGAL NLP (Novel Contribution #4)
- Sastrawi stemmer integration for better BM25
- Legal Named Entity Recognition (law numbers, article refs, dates)
- Automatic legal hierarchy extraction from text

### 5. COMPETITIVE COMPARISON DASHBOARD (Presentation Feature)
- Side-by-side comparison with JDIH, Hukumonline, ChatGPT
- Show retrieval quality differences with metrics
- Auto-generated comparison table for portfolio/presentation
