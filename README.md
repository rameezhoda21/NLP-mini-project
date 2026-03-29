---
title: Sindh Criminal Law RAG
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
---

# Sindh Criminal Prosecution Law — RAG System

A Retrieval-Augmented Generation (RAG) system for querying Sindh Criminal Prosecution Law documents, built as part of Mini-Project 1: NLP with Deep Learning.

## Architecture

```
Query → [BM25 + Semantic Search] → RRF Fusion → Cross-Encoder Reranking → LLM Answer → Evaluation
```

### Components

| Component | Implementation |
|-----------|---------------|
| **Chunking** | Legal-aware chunking (respects section/rule boundaries, 250-450 words) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, pre-computed locally) |
| **Vector DB** | Pinecone (cloud-hosted, cosine similarity) |
| **Retrieval** | Hybrid: BM25 keyword search + Pinecone semantic search, fused via Reciprocal Rank Fusion (k=60) |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` with confidence gating |
| **LLM** | Llama-3.1-8B-Instruct via Groq API (free-tier inference) |
| **Evaluation** | LLM-as-a-Judge: Faithfulness (claim extraction + verification) and Relevancy (alternate query generation + cosine similarity) |
| **UI** | Streamlit with retrieval mode selector, live evaluation toggle |

## Data

7 Sindh Criminal Law documents (PDFs) covering:
- Sindh Criminal Court Rules 2012
- Sindh Criminal Prosecution Service Act 2009
- 4 Amendment Acts (2011, 2014, 2015, 2025)
- Consolidated Act

**Corpus**: 6,877 legal-aware chunks (522 retrievable after filtering), 340K characters of cleaned text.

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Required Environment Variables

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | LLM inference (Llama-3.1-8B via Groq) |
| `PINECONE_API_KEY` | Vector database for semantic search |
| `HF_API_TOKEN` | HuggingFace token (for model downloads) |

## Evaluation

Run the automated LLM-as-a-Judge evaluation pipeline:
```bash
python evaluation/evaluate_rag.py
```

Run the ablation study (compares 2 chunking strategies x 2 retrieval modes):
```bash
python evaluation/ablation_study.py
```

## Pipeline Scripts (notebooks/)

| Script | Purpose |
|--------|---------|
| `process_pdfs.py` | PDF extraction and text cleaning |
| `chunk_text.py` | Legal-aware chunking strategy |
| `chunk_text_fixed.py` | Fixed-size chunking (ablation baseline) |
| `filter_retrievable_chunks.py` | Tag and filter low-value chunks |
| `embed_chunks.py` | Generate sentence-transformer embeddings |
| `upload_to_pinecone.py` | Batch upsert embeddings to Pinecone |
| `query_bm25.py` | Standalone BM25 search |
| `query_pinecone.py` | Standalone Pinecone semantic search |
| `rrf_fusion.py` | Hybrid retrieval with RRF fusion |
| `rerank_crossencoder.py` | Cross-encoder reranking + answer generation |
