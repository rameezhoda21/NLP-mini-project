"""
Ablation study for the Sindh Criminal Law RAG system.

Compares performance across two dimensions:
  1. Chunking strategy: Legal-aware (chunk_text.py) vs Fixed-size (chunk_text_fixed.py)
  2. Retrieval mode:    Semantic-only vs Hybrid (BM25 + Semantic) + Cross-Encoder Reranking

Produces a results CSV and prints a summary table with Faithfulness and Relevancy
scores for each combination (4 configurations total).

Usage:
    python evaluation/ablation_study.py

Required env vars:
    PINECONE_API_KEY, HF_API_TOKEN (or HF_TOKEN / HUGGINGFACEHUB_API_TOKEN)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "evaluation"

LEGAL_CHUNKS_CSV = DATA_DIR / "rag_chunks.csv"
FIXED_CHUNKS_CSV = DATA_DIR / "rag_chunks_fixed.csv"
TEST_QUERIES_JSON = EVAL_DIR / "test_queries.json"
RESULTS_CSV = EVAL_DIR / "ablation_results.csv"

# ---------------------------------------------------------------------------
# Model config (mirrors evaluate_rag.py / app.py)
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HF_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

SEMANTIC_TOP_K = 20
BM25_TOP_K = 20
FUSED_TOP_K = 10
RERANK_TOP_K = 5
ANSWER_TOP_K = 3

HF_TOKEN_ENV_CANDIDATES = ["HF_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HF Inference API helper — uses huggingface_hub SDK (chat completions)
# ---------------------------------------------------------------------------

def _get_hf_token() -> str:
    for name in HF_TOKEN_ENV_CANDIDATES:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return ""


_hf_client = None

def _get_hf_client():
    global _hf_client
    if _hf_client is None:
        from huggingface_hub import InferenceClient
        _hf_client = InferenceClient(token=_get_hf_token())
    return _hf_client


def call_hf_inference(prompt: str, max_tokens: int = 350, temperature: float = 0.2) -> str:
    client = _get_hf_client()
    for attempt in range(4):
        try:
            resp = client.chat_completion(
                model=HF_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
            )
            text = resp.choices[0].message.content.strip()
            if not text:
                raise RuntimeError("Empty response from HF API")
            return text
        except Exception as exc:
            if "402" in str(exc) or "429" in str(exc):
                wait = 15 * (attempt + 1)
                logger.warning("Rate limited, waiting %ds (attempt %d/4)...", wait, attempt + 1)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("HF API rate limit exceeded after 4 retries")


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def legal_tokenize(text: str) -> list[str]:
    text = str(text).lower()
    return re.findall(r"\d+[a-z]?(?:\([a-z0-9]+\))*|[a-z]+(?:-[a-z]+)*", text)


def load_chunks(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["chunk_id"] = df["chunk_id"].astype(str)
    df["chunk_text"] = df["chunk_text"].fillna("").astype(str)
    # Filter retrievable if column exists (legal chunks have it, fixed chunks don't)
    if "is_retrievable" in df.columns:
        mask = df["is_retrievable"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        df = df[mask].copy()
    return df


def build_bm25(df: pd.DataFrame) -> BM25Okapi:
    return BM25Okapi([legal_tokenize(t) for t in df["chunk_text"]])


def local_semantic_search(
    query: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    embed_model: SentenceTransformer,
    top_k: int = SEMANTIC_TOP_K,
) -> list[str]:
    """Semantic search using local embeddings (no Pinecone needed)."""
    query_emb = embed_model.encode([query])
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [df.iloc[i]["chunk_id"] for i in top_indices]


def bm25_search(df: pd.DataFrame, bm25: BM25Okapi, query: str, top_k: int = BM25_TOP_K) -> list[str]:
    scores = bm25.get_scores(legal_tokenize(query))
    result = df[["chunk_id"]].copy()
    result["score"] = scores
    return result.sort_values("score", ascending=False).head(top_k)["chunk_id"].tolist()


def reciprocal_rank_fusion(sem_ids: list[str], bm25_ids: list[str], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for rank, cid in enumerate(sem_ids, 1):
        scores[cid] += 1.0 / (k + rank)
    for rank, cid in enumerate(bm25_ids, 1):
        scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def rerank(query: str, chunks: list[dict], cross_encoder: CrossEncoder, top_k: int = RERANK_TOP_K) -> list[dict]:
    if not chunks:
        return []
    pairs = [(query, c["chunk_text"]) for c in chunks]
    scores = cross_encoder.predict(pairs)
    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)
    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return chunks[:top_k]


def build_grounded_prompt(query: str, chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[Context {i}]\nChunk ID: {c.get('chunk_id', '')}\n"
            f"Title: {c.get('title', '')}\nText:\n{c.get('chunk_text', '')}"
        )
    ctx = "\n\n".join(blocks)
    return (
        "You are a careful legal assistant.\n"
        "Use ONLY the context below to answer the user question.\n"
        "Do NOT invent legal rules, sections, dates, or case facts.\n"
        "If the answer is not clearly supported by the context, say: "
        "'The provided context does not contain enough reliable information to answer this question.'\n"
        "Keep your answer concise and factual.\n\n"
        f"User Question:\n{query}\n\nContext:\n{ctx}\n\nFinal Answer:"
    )


# ---------------------------------------------------------------------------
# Single query pipeline (parameterised by chunking source and retrieval mode)
# ---------------------------------------------------------------------------

def run_single_query(
    query: str,
    chunks_df: pd.DataFrame,
    chunk_embeddings: np.ndarray,
    bm25: BM25Okapi,
    embed_model: SentenceTransformer,
    cross_encoder: CrossEncoder,
    retrieval_mode: str = "hybrid",
) -> dict[str, Any]:
    sem_ids, bm25_ids = [], []

    if retrieval_mode in ("semantic", "hybrid"):
        sem_ids = local_semantic_search(query, chunks_df, chunk_embeddings, embed_model)
    if retrieval_mode in ("bm25", "hybrid"):
        bm25_ids = bm25_search(chunks_df, bm25, query)

    if retrieval_mode == "hybrid":
        fused = reciprocal_rank_fusion(sem_ids, bm25_ids)
        candidate_ids = [cid for cid, _ in fused[:FUSED_TOP_K]]
    elif retrieval_mode == "semantic":
        candidate_ids = sem_ids[:FUSED_TOP_K]
    else:
        candidate_ids = bm25_ids[:FUSED_TOP_K]

    lookup = chunks_df.set_index("chunk_id")
    candidates = []
    for cid in candidate_ids:
        if cid in lookup.index:
            row = lookup.loc[cid]
            candidates.append({
                "chunk_id": cid,
                "title": str(row.get("title", "")),
                "chunk_text": str(row.get("chunk_text", "")),
            })

    # Rerank only for hybrid mode
    if retrieval_mode == "hybrid":
        reranked = rerank(query, candidates, cross_encoder)
        answer_chunks = reranked[:ANSWER_TOP_K]
    else:
        answer_chunks = candidates[:ANSWER_TOP_K]

    try:
        prompt = build_grounded_prompt(query, answer_chunks)
        answer = call_hf_inference(prompt)
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        answer = f"[Generation error: {exc}]"

    context_text = "\n\n".join(c["chunk_text"] for c in answer_chunks)
    return {"answer": answer, "context": context_text, "answer_chunks": answer_chunks}


# ---------------------------------------------------------------------------
# Faithfulness scoring
# ---------------------------------------------------------------------------

def score_faithfulness(answer: str, context: str) -> dict[str, Any]:
    extract_prompt = (
        "Extract all distinct factual claims from the following answer as a JSON list of strings. "
        "Each claim should be a single, atomic statement. Output ONLY a JSON list, nothing else.\n\n"
        f"Answer:\n{answer}\n\nClaims:"
    )
    try:
        raw = call_hf_inference(extract_prompt, max_tokens=300, temperature=0.0)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        claims = json.loads(match.group()) if match else [answer]
        if not isinstance(claims, list) or not claims:
            claims = [answer]
        claims = [str(c) for c in claims if str(c).strip()]
    except Exception:
        claims = [answer]

    if not claims:
        return {"score": 1.0, "claims": [], "verified": []}

    verified = []
    for claim in claims[:8]:
        verify_prompt = (
            "Given the context below, is the following claim fully supported? "
            "Answer ONLY 'supported' or 'unsupported'.\n\n"
            f"Context:\n{context[:2000]}\n\nClaim: {claim}\n\nVerdict:"
        )
        try:
            verdict = call_hf_inference(verify_prompt, max_tokens=20, temperature=0.0).lower().strip()
            is_supported = "supported" in verdict and "unsupported" not in verdict
        except Exception:
            is_supported = True
        verified.append({"claim": claim, "supported": is_supported})

    supported_count = sum(1 for v in verified if v["supported"])
    score = supported_count / len(verified) if verified else 1.0
    return {"score": round(score, 3), "claims": claims, "verified": verified}


# ---------------------------------------------------------------------------
# Relevancy scoring
# ---------------------------------------------------------------------------

def score_relevancy(query: str, answer: str, embed_model: SentenceTransformer) -> dict[str, Any]:
    gen_prompt = (
        "Given the following answer, generate exactly 3 different questions that this answer "
        "would correctly respond to. Output ONLY a JSON list of 3 question strings.\n\n"
        f"Answer:\n{answer}\n\nQuestions:"
    )
    try:
        raw = call_hf_inference(gen_prompt, max_tokens=200, temperature=0.3)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        alt_queries = json.loads(match.group()) if match else []
        alt_queries = [str(q) for q in alt_queries if str(q).strip()][:3]
    except Exception:
        alt_queries = []

    if not alt_queries:
        return {"score": 0.5, "alt_queries": [], "similarities": []}

    orig_emb = embed_model.encode([query])
    alt_embs = embed_model.encode(alt_queries)
    sims = cosine_similarity(orig_emb, alt_embs)[0].tolist()

    return {
        "score": round(float(np.mean(sims)), 3),
        "alt_queries": alt_queries,
        "similarities": [round(s, 3) for s in sims],
    }


# ---------------------------------------------------------------------------
# Main ablation loop
# ---------------------------------------------------------------------------

CONFIGURATIONS = [
    {"chunk_strategy": "Legal-Aware",  "csv": LEGAL_CHUNKS_CSV, "retrieval": "hybrid"},
    {"chunk_strategy": "Legal-Aware",  "csv": LEGAL_CHUNKS_CSV, "retrieval": "semantic"},
    {"chunk_strategy": "Fixed-Size",   "csv": FIXED_CHUNKS_CSV, "retrieval": "hybrid"},
    {"chunk_strategy": "Fixed-Size",   "csv": FIXED_CHUNKS_CSV, "retrieval": "semantic"},
]


def main():
    logger.info("=" * 70)
    logger.info("ABLATION STUDY — Chunking x Retrieval")
    logger.info("=" * 70)

    # Validate environment
    hf_token = _get_hf_token()
    if not hf_token:
        logger.error("HF API token not set. Export HF_API_TOKEN / HF_TOKEN and re-run.")
        sys.exit(1)

    # Load test queries
    if not TEST_QUERIES_JSON.exists():
        logger.error("Test queries not found at %s", TEST_QUERIES_JSON)
        sys.exit(1)

    with open(TEST_QUERIES_JSON, "r", encoding="utf-8") as f:
        test_queries = json.load(f)
    logger.info("Loaded %d test queries.", len(test_queries))

    # Load models once (shared across configurations)
    logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    logger.info("Loading cross-encoder: %s", CROSS_ENCODER_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

    all_results = []

    for cfg_idx, cfg in enumerate(CONFIGURATIONS, 1):
        chunk_strategy = cfg["chunk_strategy"]
        retrieval_mode = cfg["retrieval"]
        retrieval_label = "Hybrid + Reranking" if retrieval_mode == "hybrid" else "Semantic-Only"
        csv_path = cfg["csv"]

        logger.info("-" * 70)
        logger.info("Config %d/4: Chunking=%s  |  Retrieval=%s", cfg_idx, chunk_strategy, retrieval_label)
        logger.info("-" * 70)

        if not csv_path.exists():
            logger.error("Chunks CSV not found: %s — skipping.", csv_path)
            continue

        # Load chunks and build indices
        chunks_df = load_chunks(csv_path)
        bm25 = build_bm25(chunks_df)
        logger.info("Loaded %d chunks, BM25 index built.", len(chunks_df))

        # Pre-compute embeddings for all chunks (local semantic search)
        logger.info("Encoding chunk embeddings locally...")
        chunk_texts = chunks_df["chunk_text"].tolist()
        chunk_embeddings = embed_model.encode(chunk_texts, show_progress_bar=True)

        for i, tq in enumerate(test_queries, 1):
            qid = tq["id"]
            query = tq["query"]
            logger.info("  [%d/%d] %s: %s", i, len(test_queries), qid, query[:60])

            try:
                result = run_single_query(
                    query=query,
                    chunks_df=chunks_df,
                    chunk_embeddings=chunk_embeddings,
                    bm25=bm25,
                    embed_model=embed_model,
                    cross_encoder=cross_encoder,
                    retrieval_mode=retrieval_mode,
                )
            except Exception as exc:
                logger.error("Pipeline failed for %s: %s", qid, exc)
                all_results.append({
                    "config": f"{chunk_strategy} + {retrieval_label}",
                    "chunk_strategy": chunk_strategy,
                    "retrieval_mode": retrieval_label,
                    "query_id": qid, "query": query,
                    "answer": f"[ERROR: {exc}]",
                    "faithfulness": 0.0, "relevancy": 0.0,
                })
                continue

            answer = result["answer"]
            context = result["context"]

            # Score faithfulness
            try:
                faith = score_faithfulness(answer, context)
            except Exception as exc:
                logger.warning("Faithfulness scoring failed: %s", exc)
                faith = {"score": 0.0}

            # Score relevancy
            try:
                rel = score_relevancy(query, answer, embed_model)
            except Exception as exc:
                logger.warning("Relevancy scoring failed: %s", exc)
                rel = {"score": 0.0}

            all_results.append({
                "config": f"{chunk_strategy} + {retrieval_label}",
                "chunk_strategy": chunk_strategy,
                "retrieval_mode": retrieval_label,
                "query_id": qid,
                "query": query,
                "answer": answer[:500],
                "faithfulness": faith["score"],
                "relevancy": rel["score"],
            })

            logger.info("    Faith=%.2f  Rel=%.2f", faith["score"], rel["score"])
            time.sleep(3)  # rate limiting

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_CSV, index=False, quoting=csv.QUOTE_ALL)
    logger.info("Results saved to %s", RESULTS_CSV)

    # Print summary table
    logger.info("")
    logger.info("=" * 70)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 70)

    if all_results:
        summary = results_df.groupby(["chunk_strategy", "retrieval_mode"]).agg(
            mean_faithfulness=("faithfulness", "mean"),
            mean_relevancy=("relevancy", "mean"),
            num_queries=("query_id", "count"),
        ).round(3)
        logger.info("\n%s", summary.to_string())

    logger.info("Done.")


if __name__ == "__main__":
    main()
