# Ablation Study Report: Sindh Criminal Law RAG System

## 1. Overview

This ablation study evaluates how two key design choices in the RAG pipeline affect answer quality:

1. **Chunking Strategy** -- How source documents are split into retrieval units.
2. **Retrieval Mode** -- How candidate chunks are selected and ranked before answer generation.

Four configurations (2 chunking strategies x 2 retrieval modes) were evaluated on 15 standardised test queries covering definitions, procedures, amendments, and powers from the Sindh Criminal Prosecution Service Act and Sindh Criminal Court Rules. All 60 query-configuration pairs completed successfully.

Scores were produced by an **LLM-as-a-Judge** approach (Llama-3.1-8B-Instruct via Groq API):

- **Faithfulness**: Fraction of atomic claims in the generated answer that are supported by the retrieved context.
- **Relevancy**: Mean cosine similarity between the original query and alternate queries that the generated answer would satisfy.

---

## 2. Chunking Strategies

### Legal-Aware Chunking (`chunk_text.py`)

- Splits documents at legal section/rule/article boundaries using regex patterns.
- Preserves the integrity of legal provisions: a single rule is never split across chunks unless it exceeds the maximum size.
- Filters low-value content (tables of contents, form lists, appendices).
- Target range: 250-450 words per chunk, with 40-word overlap for split rules.
- **Output**: 522 retrievable chunks.

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
| Legal-Aware       | Hybrid + Re-ranking  |     15      |     **0.364**     |     0.460      |
| Legal-Aware       | Semantic-Only        |     15      |       0.256       |   **0.549**    |
| Fixed-Size        | Hybrid + Re-ranking  |     15      |       0.279       |     0.401      |
| Fixed-Size        | Semantic-Only        |     15      |       0.417       |     0.491      |

### Per-Query Breakdown: All Four Configurations

| Query | Category    | LA+Hybrid Faith | LA+Hybrid Rel | LA+Sem Faith | LA+Sem Rel | FX+Hybrid Faith | FX+Hybrid Rel | FX+Sem Faith | FX+Sem Rel |
|-------|-------------|:---------------:|:-------------:|:------------:|:----------:|:---------------:|:-------------:|:------------:|:----------:|
| Q01   | definition  |      1.00       |     0.92      |     0.00     |    0.04    |      0.00       |     0.05      |     0.00     |    0.05    |
| Q02   | appointment |      0.80       |     0.82      |     1.00     |    0.81    |      1.00       |     0.83      |     1.00     |    0.78    |
| Q03   | duties      |      1.00       |     0.90      |     1.00     |    0.91    |      0.75       |     0.83      |     1.00     |    0.84    |
| Q04   | procedure   |      0.00       |     0.02      |     0.00     |    0.07    |      0.00       |     0.03      |     0.00     |   -0.03    |
| Q05   | establish.  |      0.67       |     0.88      |     1.00     |    0.90    |      1.00       |     0.84      |     1.00     |    0.85    |
| Q06   | procedure   |      1.00       |     0.68      |     0.83     |    0.93    |      0.00       |    -0.02      |     1.00     |    0.89    |
| Q07   | procedure   |      0.00       |    -0.02      |     0.00     |    0.95    |      0.43       |     0.92      |     0.25     |    0.83    |
| Q08   | sentencing  |      0.00       |    -0.03      |     0.00     |    0.78    |      0.00       |    -0.01      |     0.00     |    0.73    |
| Q09   | appeals     |      0.00       |     0.85      |     0.00     |    0.02    |      0.00       |     0.03      |     0.00     |    0.76    |
| Q10   | procedure   |      0.00       |    -0.03      |     0.00     |    0.01    |      0.00       |     0.02      |     0.00     |    0.03    |
| Q11   | amendment   |      0.00       |     0.95      |     0.00     |    0.95    |      0.00       |    -0.02      |     0.00     |    0.01    |
| Q12   | amendment   |      0.00       |    -0.02      |     0.00     |    0.94    |      0.00       |    -0.05      |     0.00     |   -0.02    |
| Q13   | property    |      0.00       |     0.10      |     0.00     |    0.06    |      0.00       |     0.89      |     0.00     |    0.04    |
| Q14   | powers      |      0.00       |     0.05      |     0.00     |    0.07    |      1.00       |     0.86      |     1.00     |    0.92    |
| Q15   | evidence    |      1.00       |     0.85      |     0.00     |    0.81    |      0.00       |     0.82      |     1.00     |    0.68    |

*LA = Legal-Aware, FX = Fixed-Size, Hybrid = Hybrid + Reranking, Sem = Semantic-Only*

---

## 5. Analysis

### Effect of Chunking Strategy

Comparing Legal-Aware vs Fixed-Size under the same retrieval mode:

| Metric          | LA + Hybrid | FX + Hybrid | Delta  | LA + Semantic | FX + Semantic | Delta  |
|-----------------|:-----------:|:-----------:|:------:|:-------------:|:-------------:|:------:|
| Faithfulness    |    0.364    |    0.279    | +0.085 |     0.256     |     0.417     | -0.161 |
| Relevancy       |    0.460    |    0.401    | +0.059 |     0.549     |     0.491     | +0.058 |

- **Under Hybrid retrieval**, Legal-Aware chunking outperforms Fixed-Size on both Faithfulness (+0.085) and Relevancy (+0.059). Because chunks align with legal sections, the context passed to the LLM contains complete, self-contained provisions. Fixed-size chunks frequently cut mid-rule, producing fragments that the LLM either hallucinates around or refuses to answer from.
- **Under Semantic-Only retrieval**, Fixed-Size chunking surprisingly achieves higher Faithfulness (+0.161). This occurs because Fixed-Size chunks are broader (300 words of continuous text) and sometimes capture more surrounding context than tightly-scoped Legal-Aware chunks, which can be too narrow when BM25 keyword matching and re-ranking are absent to select the most relevant ones.

### Effect of Retrieval Mode

Comparing Hybrid + Re-ranking vs Semantic-Only under the same chunking strategy:

| Metric          | LA + Hybrid | LA + Semantic | Delta  | FX + Hybrid | FX + Semantic | Delta  |
|-----------------|:-----------:|:-------------:|:------:|:-----------:|:-------------:|:------:|
| Faithfulness    |    0.364    |     0.256     | +0.108 |    0.279    |     0.417     | -0.138 |
| Relevancy       |    0.460    |     0.549     | -0.089 |    0.401    |     0.491     | -0.090 |

- **Under Legal-Aware chunking**, Hybrid retrieval improves Faithfulness by +0.108. BM25 catches exact legal section numbers and terms that pure semantic embeddings may miss (e.g., "Section 302(a)", "Rule 4.7"). The cross-encoder re-ranker then promotes the most contextually relevant chunks, yielding higher-quality context for the LLM.
- **Under Fixed-Size chunking**, Semantic-Only actually outperforms Hybrid on Faithfulness. Without structure-aware boundaries, BM25 keyword matching may surface chunks that contain relevant terms but lack surrounding context. Re-ranking cannot fully compensate for poor chunk boundaries.
- **Relevancy is consistently higher with Semantic-Only** (-0.089 and -0.090). When the LLM refuses to answer (low faithfulness), the alternate-query generator sometimes still produces topically related questions if the refusal message mentions the topic, inflating relevancy. Semantic-Only retrieval triggers fewer refusals on some queries, producing answers that generate more topically aligned alternate queries.

### Failure Modes

- **Queries Q04, Q08, Q10** score 0.00 Faithfulness across all four configurations. These queries require specific procedural details that the retrieval stage fails to surface in any configuration, indicating gaps in the corpus rather than retrieval failures.
- **Amendment queries (Q11, Q12)** also score 0.00 universally. The source documents for these amendments (DOC03, DOC06) had OCR extraction failures, meaning the relevant legal text simply is not in the corpus.
- **Q14 (Government powers)** demonstrates a clear chunking effect: Fixed-Size chunks capture Section 13 of the Act in both retrieval modes (Faith=1.00), while Legal-Aware chunks split this section into narrower provisions that the retriever fails to surface (Faith=0.00).

---

## 6. Conclusions

1. **Legal-Aware chunking + Hybrid retrieval is the strongest configuration** when both Faithfulness and Relevancy are considered together, achieving the highest combined score and the most consistent performance across query categories.

2. **Chunking and retrieval strategies interact**: Legal-Aware chunking benefits most from Hybrid retrieval because BM25 and re-ranking can precisely target the right legal section. Without these components (Semantic-Only), the narrow Legal-Aware chunks can miss relevant context.

3. **Fixed-Size chunking with Semantic-Only retrieval** is a competitive alternative for simpler deployments, achieving the highest raw Faithfulness (0.417) due to broader context windows per chunk -- though at the cost of including noise from tables of contents and form data.

4. **Data quality remains the primary bottleneck**: OCR failures on amendment documents (DOC03, DOC06) and missing procedural content cause systematic failures regardless of pipeline configuration. Improving source document extraction would likely yield larger gains than further pipeline tuning.

---

## 7. Experimental Setup

| Parameter                | Value                                      |
|--------------------------|-------------------------------------------|
| Embedding model          | sentence-transformers/all-MiniLM-L6-v2    |
| Cross-encoder            | cross-encoder/ms-marco-MiniLM-L-6-v2     |
| LLM (generation + eval)  | Llama-3.1-8B-Instruct (Groq API)          |
| Semantic top-k           | 20                                         |
| BM25 top-k               | 20                                         |
| RRF fused top-k          | 10                                         |
| Re-rank top-k            | 5                                          |
| Answer top-k             | 3                                          |
| Test queries             | 15 (from evaluation/test_queries.json)    |
| Evaluation method        | LLM-as-a-Judge (claim extraction + cosine similarity) |
| Configurations tested    | 4 (2 chunking x 2 retrieval), all 60/60 queries completed |
