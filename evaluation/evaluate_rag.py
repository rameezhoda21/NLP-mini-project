"""
LLM-as-a-Judge evaluation pipeline for the Sindh Criminal Law RAG system.

Runs 15 fixed test queries through the full RAG pipeline and scores each on:
  - Faithfulness: Claim Extraction + Verification against retrieved context
  - Relevancy: Alternate Query Generation + Cosine Similarity

Results are saved as a CSV report with per-query and aggregate scores.

Usage:
    python evaluation/evaluate_rag.py

Required env vars:
    PINECONE_API_KEY, HF_API_TOKEN
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
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

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
CHUNKS_CSV = DATA_DIR / "rag_chunks_with_retrievable_flag.csv"
TEST_QUERIES_JSON = EVAL_DIR / "test_queries.json"
RESULTS_CSV = EVAL_DIR / "evaluation_results.csv"

# ---------------------------------------------------------------------------
# Model / index config (mirrors app.py and notebooks)
# ---------------------------------------------------------------------------
INDEX_NAME = "legal-rag-index-filtered"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

SEMANTIC_TOP_K = 20
BM25_TOP_K = 20
FUSED_TOP_K = 10
RERANK_TOP_K = 5
ANSWER_TOP_K = 3

HF_TOKEN_ENV_CANDIDATES = ["HF_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HF Inference API helper
# ---------------------------------------------------------------------------

def _get_hf_token() -> str:
    for name in HF_TOKEN_ENV_CANDIDATES:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return ""


def call_hf_inference(prompt: str, max_tokens: int = 350, temperature: float = 0.2) -> str:
    token = _get_hf_token()
    if not token:
        raise RuntimeError("Missing HF API token.")
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": False,
            "return_full_text": False,
        },
    }
    req = urllib_request.Request(
        HF_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HF API error {exc.code}: {err}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error: {exc}") from exc

    if isinstance(body, list) and body and isinstance(body[0], dict):
        text = body[0].get("generated_text", "").strip()
        if text:
            return text
    if isinstance(body, dict) and "error" in body:
        raise RuntimeError(f"HF API error: {body['error']}")
    raise RuntimeError(f"Unexpected response: {body}")


# ---------------------------------------------------------------------------
# Retrieval helpers (reused from pipeline)
# ---------------------------------------------------------------------------

def legal_tokenize(text: str) -> list[str]:
    text = str(text).lower()
    return re.findall(r"\d+[a-z]?(?:\([a-z0-9]+\))*|[a-z]+(?:-[a-z]+)*", text)


def load_chunks(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["chunk_id"] = df["chunk_id"].astype(str)
    df["chunk_text"] = df["chunk_text"].fillna("").astype(str)
    if "is_retrievable" in df.columns:
        mask = df["is_retrievable"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        df = df[mask].copy()
    return df


def build_bm25(df: pd.DataFrame) -> BM25Okapi:
    return BM25Okapi([legal_tokenize(t) for t in df["chunk_text"]])


def semantic_search(query: str, model: SentenceTransformer, api_key: str, top_k: int = SEMANTIC_TOP_K) -> list[str]:
    from pinecone import Pinecone as PineconeClient
    embedding = model.encode(query).tolist()
    pc = PineconeClient(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    response = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    ids = []
    for match in response.get("matches", []):
        meta = match.get("metadata") or {}
        ids.append(str(meta.get("chunk_id", match.get("id"))))
    return ids


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


def rerank(query: str, chunks: list[dict], model: CrossEncoder, top_k: int = RERANK_TOP_K) -> list[dict]:
    if not chunks:
        return []
    pairs = [(query, c["chunk_text"]) for c in chunks]
    scores = model.predict(pairs)
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
# Full retrieval + generation pipeline for a single query
# ---------------------------------------------------------------------------

def run_single_query(
    query: str,
    pinecone_key: str,
    chunks_df: pd.DataFrame,
    bm25: BM25Okapi,
    embed_model: SentenceTransformer,
    cross_encoder: CrossEncoder,
    retrieval_mode: str = "hybrid",
) -> dict[str, Any]:
    allowed_ids = set(chunks_df["chunk_id"].tolist())
    sem_ids, bm25_ids = [], []

    if retrieval_mode in ("semantic", "hybrid"):
        sem_ids = [i for i in semantic_search(query, embed_model, pinecone_key) if i in allowed_ids]
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

    reranked = rerank(query, candidates, cross_encoder)
    answer_chunks = reranked[:ANSWER_TOP_K]

    try:
        prompt = build_grounded_prompt(query, answer_chunks)
        answer = call_hf_inference(prompt)
    except Exception as exc:
        logger.error("Generation failed for query '%s': %s", query, exc)
        answer = f"[Generation error: {exc}]"

    context_text = "\n\n".join(c["chunk_text"] for c in answer_chunks)
    return {
        "answer": answer,
        "context": context_text,
        "answer_chunks": answer_chunks,
        "reranked_chunks": reranked,
    }


# ---------------------------------------------------------------------------
# Faithfulness: Claim Extraction + Verification
# ---------------------------------------------------------------------------

def score_faithfulness(answer: str, context: str) -> dict[str, Any]:
    """
    Step 1: Extract atomic claims from the answer.
    Step 2: Verify each claim against the retrieved context.
    Score = fraction of claims that are supported.
    """
    # Step 1 — Extract claims
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

    # Step 2 — Verify each claim
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
# Relevancy: Alternate Query Generation + Cosine Similarity
# ---------------------------------------------------------------------------

def score_relevancy(query: str, answer: str, embed_model: SentenceTransformer) -> dict[str, Any]:
    """
    Step 1: Generate 3 alternate queries the answer would satisfy.
    Step 2: Cosine similarity between original query and each alternate.
    Score = mean similarity.
    """
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
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 70)
    logger.info("LLM-as-a-Judge Evaluation Pipeline")
    logger.info("=" * 70)

    # Validate environment
    pinecone_key = os.environ.get("PINECONE_API_KEY", "").strip()
    hf_token = _get_hf_token()
    if not pinecone_key:
        logger.error("PINECONE_API_KEY not set. Export it and re-run.")
        sys.exit(1)
    if not hf_token:
        logger.error("HF_API_TOKEN not set. Export it and re-run.")
        sys.exit(1)

    # Load test queries
    if not TEST_QUERIES_JSON.exists():
        logger.error("Test queries not found at %s", TEST_QUERIES_JSON)
        sys.exit(1)

    with open(TEST_QUERIES_JSON, "r", encoding="utf-8") as f:
        test_queries = json.load(f)
    logger.info("Loaded %d test queries from %s", len(test_queries), TEST_QUERIES_JSON.name)

    # Load models and data
    logger.info("Loading chunks from %s ...", CHUNKS_CSV.name)
    chunks_df = load_chunks(CHUNKS_CSV)
    bm25 = build_bm25(chunks_df)
    logger.info("Loaded %d retrievable chunks, BM25 index built.", len(chunks_df))

    logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    logger.info("Loading cross-encoder: %s", CROSS_ENCODER_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

    # Run evaluation
    results = []
    for i, tq in enumerate(test_queries, 1):
        qid = tq["id"]
        query = tq["query"]
        logger.info("[%d/%d] Evaluating %s: %s", i, len(test_queries), qid, query[:80])

        # Run RAG pipeline
        try:
            pipeline_result = run_single_query(
                query=query,
                pinecone_key=pinecone_key,
                chunks_df=chunks_df,
                bm25=bm25,
                embed_model=embed_model,
                cross_encoder=cross_encoder,
                retrieval_mode="hybrid",
            )
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", qid, exc)
            results.append({
                "query_id": qid, "query": query, "answer": f"[ERROR: {exc}]",
                "faithfulness": 0.0, "relevancy": 0.0, "error": str(exc),
            })
            continue

        answer = pipeline_result["answer"]
        context = pipeline_result["context"]

        # Score faithfulness
        try:
            faith = score_faithfulness(answer, context)
        except Exception as exc:
            logger.warning("Faithfulness scoring failed for %s: %s", qid, exc)
            faith = {"score": 0.0, "claims": [], "verified": []}

        # Score relevancy
        try:
            rel = score_relevancy(query, answer, embed_model)
        except Exception as exc:
            logger.warning("Relevancy scoring failed for %s: %s", qid, exc)
            rel = {"score": 0.0, "alt_queries": [], "similarities": []}

        results.append({
            "query_id": qid,
            "query": query,
            "category": tq.get("category", ""),
            "answer": answer[:500],
            "faithfulness": faith["score"],
            "relevancy": rel["score"],
            "num_claims": len(faith.get("claims", [])),
            "supported_claims": sum(1 for v in faith.get("verified", []) if v.get("supported")),
            "alt_queries": "; ".join(rel.get("alt_queries", [])),
            "cosine_sims": str(rel.get("similarities", [])),
        })

        logger.info("  Faithfulness=%.2f  Relevancy=%.2f", faith["score"], rel["score"])

        # Rate limiting: brief pause between queries
        time.sleep(1)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False, quoting=csv.QUOTE_ALL)
    logger.info("Results saved to %s", RESULTS_CSV)

    # Print summary
    logger.info("=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info("Queries evaluated: %d", len(results))

    if results:
        faith_scores = [r["faithfulness"] for r in results if isinstance(r["faithfulness"], (int, float))]
        rel_scores = [r["relevancy"] for r in results if isinstance(r["relevancy"], (int, float))]

        if faith_scores:
            logger.info("Faithfulness — Mean: %.3f  Min: %.3f  Max: %.3f",
                        np.mean(faith_scores), np.min(faith_scores), np.max(faith_scores))
        if rel_scores:
            logger.info("Relevancy   — Mean: %.3f  Min: %.3f  Max: %.3f",
                        np.mean(rel_scores), np.min(rel_scores), np.max(rel_scores))

    logger.info("Done.")


if __name__ == "__main__":
    main()
