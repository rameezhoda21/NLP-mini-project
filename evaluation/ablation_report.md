# Ablation Study Report: Sindh Criminal Law RAG System

## 1. Overview

This ablation study evaluates how two key design choices in the RAG pipeline affect answer quality:

1. **Chunking Strategy** -- How source documents are split into retrieval units.
2. **Retrieval Mode** -- How candidate chunks are selected and ranked before answer generation.

Each configuration was evaluated on 15 standardised test queries covering definitions, procedures, amendments, and powers from the Sindh Criminal Prosecution Service Act and Sindh Criminal Court Rules.

Scores were produced by an **LLM-as-a-Judge** approach (Qwen2.5-72B-Instruct via HF Inference API):

- **Faithfulness**: Fraction of atomic claims in the generated answer that are supported by the retrieved context.
- **Relevancy**: Mean cosine similarity between the original query and alternate queries that the generated answer would satisfy.

---

## 2. Chunking Strategies

### Legal-Aware Chunking (`chunk_text.py`)

- Splits documents at legal section/rule/article boundaries using regex patterns.
- Preserves the integrity of legal provisions: a single rule is never split across chunks unless it exceeds the maximum size.
- Filters low-value content (tables of contents, form lists, appendices).
- Target range: 250-450 words per chunk, with 40-word overlap for split rules.
- **Output**: 549 retrievable chunks (from 6,877 raw chunks after filtering).

### Fixed-Size Chunking (`chunk_text_fixed.py`)

- Splits documents into fixed windows of 300 words with 50-word overlap.
- No awareness of document structure -- cuts can fall mid-sentence or mid-section.
- No content filtering -- all text is chunked including tables of contents and form lists.
- **Output**: 221 chunks.

---

## 3. Retrieval Modes

### Hybrid + Re-ranking

1. Semantic search (top-20 by cosine similarity of MiniLM-L6-v2 embeddings).
2. BM25 keyword search (top-20 using legal-aware tokeniser).
3. Reciprocal Rank Fusion (RRF, k=60) to merge both lists into top-10 candidates.
4. Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2) to select the final top-3 answer chunks.

### Semantic-Only

1. Semantic search (top-20 by cosine similarity).
2. Top-3 chunks passed directly to the LLM -- no BM25, no fusion, no re-ranking.

---

## 4. Results

### Performance Comparison Table

| Chunking Strategy | Retrieval Mode       | Queries (n) | Mean Faithfulness | Mean Relevancy |
|-------------------|----------------------|:-----------:|:-----------------:|:--------------:|
| Legal-Aware       | Hybrid + Re-ranking  |     15      |     **0.800**     |   **0.573**    |
| Legal-Aware       | Semantic-Only        |     15      |       0.571       |     0.533      |
| Fixed-Size        | Hybrid + Re-ranking  |     13*     |       0.641       |     0.595      |

\* Fixed-Size + Semantic-Only configuration could not be fully evaluated due to HF API rate-limit exhaustion (only 1 of 15 queries completed); it is excluded from the table to avoid misleading conclusions.

### Per-Query Breakdown (Legal-Aware + Hybrid vs Semantic-Only)

| Query | Category    | Hybrid Faith | Hybrid Rel | Semantic Faith | Semantic Rel |
|-------|-------------|:------------:|:----------:|:--------------:|:------------:|
| Q01   | definition  |     1.00     |    0.92    |      0.00      |     0.07     |
| Q02   | appointment |     1.00     |    0.76    |      1.00      |     0.76     |
| Q03   | duties      |     1.00     |    0.95    |      1.00      |     0.95     |
| Q04   | procedure   |     1.00     |    0.70    |      0.29      |     0.70     |
| Q05   | establish.  |     1.00     |    0.86    |      1.00      |     0.90     |
| Q06   | procedure   |     1.00     |    0.93    |      1.00      |     0.95     |
| Q07   | procedure   |     1.00     |    0.87    |      0.40      |     0.82     |
| Q08   | sentencing  |     1.00     |    0.43    |      0.88      |     0.45     |
| Q09   | appeals     |     1.00     |    0.61    |      1.00      |     0.62     |
| Q10   | procedure   |     0.00     |    0.06    |      0.00      |     0.06     |
| Q11   | amendment   |     1.00     |    0.06    |      0.00      |     0.04     |
| Q12   | amendment   |     0.00     |    0.03    |      0.00      |     0.01     |
| Q13   | property    |     1.00     |    0.70    |      1.00      |     0.70     |
| Q14   | powers      |     0.00     |    0.10    |      1.00      |     0.80     |
| Q15   | evidence    |     1.00     |    0.62    |      0.00      |     0.16     |

---

## 5. Analysis

### Effect of Chunking Strategy

Comparing the two Hybrid + Re-ranking rows:

| Metric          | Legal-Aware | Fixed-Size | Delta   |
|-----------------|:-----------:|:----------:|:-------:|
| Faithfulness    |    0.800    |   0.641    | +0.159  |
| Relevancy       |    0.573    |   0.595    | -0.022  |

- **Faithfulness is substantially higher with Legal-Aware chunking** (+0.159). Because chunks align with legal sections, the context passed to the LLM contains complete, self-contained provisions. Fixed-size chunks frequently cut mid-rule, producing fragments that the LLM either hallucinates around or cannot verify claims against.
- **Relevancy is comparable** (within 0.02). Relevancy measures whether the answer addresses the right topic, which depends more on retrieval recall than on chunk boundaries. Both strategies surface topically relevant chunks; the difference is in how faithfully the LLM can use them.

### Effect of Retrieval Mode

Comparing the two Legal-Aware rows:

| Metric          | Hybrid + Re-rank | Semantic-Only | Delta   |
|-----------------|:----------------:|:-------------:|:-------:|
| Faithfulness    |      0.800       |     0.571     | +0.229  |
| Relevancy       |      0.573       |     0.533     | +0.040  |

- **Hybrid retrieval with re-ranking improves Faithfulness by 0.229**. BM25 catches exact legal section numbers and terms that pure semantic embeddings may miss (e.g., "Section 302(a)", "Rule 4.7"). Re-ranking then promotes the chunks that are most contextually relevant to the query, yielding higher-quality context for the LLM.
- **Relevancy also improves slightly** (+0.04). The cross-encoder re-ranker ensures the final context is tightly scoped to the query, producing more focused answers.
- Looking at per-query results, the biggest Faithfulness gaps appear on queries involving specific section references (Q01, Q04, Q07, Q11, Q15), where BM25's keyword matching gives Hybrid retrieval a decisive advantage.

### Failure Modes

- **Amendment queries (Q10-Q12)** score poorly across all configurations. The source documents for these amendments (DOC03, DOC06) had OCR extraction failures (0 chars extracted), meaning the relevant text simply is not in the corpus.
- **Q14** is an anomaly where Semantic-Only outperforms Hybrid. The query about "Government powers to make rules" is semantically clear but uses generic terms that BM25 dilutes with irrelevant matches. The cross-encoder partially corrects this, but not fully.

---

## 6. Conclusions

1. **Legal-Aware chunking outperforms Fixed-Size chunking** on Faithfulness (+0.159) while maintaining comparable Relevancy. Respecting document structure prevents the LLM from generating claims it cannot verify against fragmented context.

2. **Hybrid retrieval with re-ranking outperforms Semantic-Only** on both Faithfulness (+0.229) and Relevancy (+0.040). The combination of BM25 keyword matching and cross-encoder re-ranking provides higher-quality context, especially for queries referencing specific legal sections.

3. **The best configuration is Legal-Aware chunking + Hybrid retrieval with re-ranking**, achieving 0.800 Faithfulness and 0.573 Relevancy across 15 test queries.

4. **Data quality remains the primary bottleneck**: OCR failures on amendment documents (DOC03, DOC06) cause systematic failures regardless of pipeline configuration. Improving source document extraction would likely yield larger gains than further pipeline tuning.

---

## 7. Experimental Setup

| Parameter                | Value                                      |
|--------------------------|-------------------------------------------|
| Embedding model          | sentence-transformers/all-MiniLM-L6-v2    |
| Cross-encoder            | cross-encoder/ms-marco-MiniLM-L-6-v2     |
| LLM (generation + eval)  | Qwen/Qwen2.5-72B-Instruct (HF Inference) |
| Semantic top-k           | 20                                         |
| BM25 top-k               | 20                                         |
| RRF fused top-k          | 10                                         |
| Re-rank top-k            | 5                                          |
| Answer top-k             | 3                                          |
| Test queries             | 15 (from evaluation/test_queries.json)    |
| Evaluation method        | LLM-as-a-Judge (claim extraction + cosine similarity) |
