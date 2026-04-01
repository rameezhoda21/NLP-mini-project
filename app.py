"""
Streamlit RAG app for Sindh Criminal Prosecution Law.

Integrates the full pipeline:
  1) Hybrid retrieval  (BM25 + Pinecone semantic) with RRF fusion
  2) Cross-encoder reranking + confidence gate
  3) LLM answer generation via HF Inference API (Mistral-7B)
  4) Live Faithfulness & Relevancy scoring (LLM-as-a-Judge)

Designed for deployment on Hugging Face Spaces.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from groq import Groq

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — mirror the values used in the existing pipeline scripts
# ---------------------------------------------------------------------------
INDEX_NAME = "legal-rag-index-filtered"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL_ID = "llama-3.3-70b-versatile"

DATA_DIR = Path(__file__).resolve().parent / "data"
CHUNKS_CSV = DATA_DIR / "rag_chunks_with_retrievable_flag.csv"

SEMANTIC_TOP_K = 20
BM25_TOP_K = 20
FUSED_TOP_K = 10
RERANK_TOP_K = 5
ANSWER_TOP_K = 3

# Confidence-gate thresholds (from rerank_crossencoder.py)
TOP_SCORE_THRESHOLD = 1.0
LOW_SCORE_THRESHOLD = 0.0
MIN_KEYWORD_COVERAGE = 0.30

FALLBACK_RESPONSE = (
    "I could not find a reliable answer to this question in the retrieved documents."
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with",
}

HF_TOKEN_ENV_CANDIDATES = ["HF_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"]

# ---------------------------------------------------------------------------
# Helpers — reused verbatim from the existing notebooks/ scripts
# ---------------------------------------------------------------------------

def legal_tokenize(text: str) -> list[str]:
    """Legal-aware BM25 tokenizer (mirrors notebooks/query_bm25.py)."""
    text = str(text).lower()
    return re.findall(r"\d+[a-z]?(?:\([a-z0-9]+\))*|[a-z]+(?:-[a-z]+)*", text)


def _get_hf_token() -> str:
    for name in HF_TOKEN_ENV_CANDIDATES:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return ""


_groq_client = None

def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return _groq_client


def call_hf_inference(prompt: str, max_tokens: int = 350, temperature: float = 0.2) -> str:
    """Call Groq API for LLM inference (Llama-3.1-8B). Retries on rate limits."""
    client = _get_groq_client()
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = resp.choices[0].message.content.strip()
            if not text:
                raise RuntimeError("Empty response from Groq API")
            return text
        except Exception as exc:
            if "429" in str(exc) or "rate" in str(exc).lower():
                wait = 10 * (attempt + 1)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Groq API rate limit exceeded after 4 retries")


# ---------------------------------------------------------------------------
# Cached resource loaders (run once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner="Loading cross-encoder model...")
def load_cross_encoder():
    return CrossEncoder(CROSS_ENCODER_NAME)


@st.cache_data(show_spinner="Loading legal chunks & building BM25 index...")
def load_chunks_and_bm25(csv_path: str):
    df = pd.read_csv(csv_path)
    df["chunk_id"] = df["chunk_id"].astype(str)
    df["chunk_text"] = df["chunk_text"].fillna("").astype(str)

    if "is_retrievable" in df.columns:
        mask = df["is_retrievable"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        df = df[mask].copy()

    tokenized = [legal_tokenize(t) for t in df["chunk_text"]]
    bm25 = BM25Okapi(tokenized)
    return df, bm25


# ---------------------------------------------------------------------------
# Retrieval stages
# ---------------------------------------------------------------------------

def semantic_search(query: str, model: SentenceTransformer, api_key: str, top_k: int = SEMANTIC_TOP_K) -> list[str]:
    """Retrieve top-k chunk_ids from Pinecone."""
    from pinecone import Pinecone as PineconeClient

    embedding = model.encode(query).tolist()
    pc = PineconeClient(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    response = index.query(vector=embedding, top_k=top_k, include_metadata=True)

    ids: list[str] = []
    for match in response.get("matches", []):
        meta = match.get("metadata") or {}
        ids.append(str(meta.get("chunk_id", match.get("id"))))
    return ids


def bm25_search(df: pd.DataFrame, bm25: BM25Okapi, query: str, top_k: int = BM25_TOP_K) -> list[str]:
    """Retrieve top-k chunk_ids via BM25."""
    scores = bm25.get_scores(legal_tokenize(query))
    result = df[["chunk_id"]].copy()
    result["score"] = scores
    return result.sort_values("score", ascending=False).head(top_k)["chunk_id"].tolist()


def reciprocal_rank_fusion(sem_ids: list[str], bm25_ids: list[str], k: int = 60) -> list[tuple[str, float]]:
    """RRF over two ranked lists (mirrors notebooks/rrf_fusion.py)."""
    scores: dict[str, float] = defaultdict(float)
    for rank, cid in enumerate(sem_ids, 1):
        scores[cid] += 1.0 / (k + rank)
    for rank, cid in enumerate(bm25_ids, 1):
        scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def rerank(query: str, chunks: list[dict], model: CrossEncoder, top_k: int = RERANK_TOP_K) -> list[dict]:
    """Cross-encoder reranking (mirrors notebooks/rerank_crossencoder.py)."""
    if not chunks:
        return []
    pairs = [(query, c["chunk_text"]) for c in chunks]
    scores = model.predict(pairs)
    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)
    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return chunks[:top_k]


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------

def assess_confidence(query: str, top_chunks: list[dict]) -> dict[str, Any]:
    if not top_chunks:
        return {"is_in_scope": False, "reason": "no_chunks", "top_score": float("-inf"),
                "keyword_coverage": 0.0}

    scores = [c.get("rerank_score", 0.0) for c in top_chunks]
    top_score = max(scores)
    all_low = all(s <= LOW_SCORE_THRESHOLD for s in scores)

    words = re.findall(r"[a-zA-Z]+", query.lower())
    keywords = [w for w in words if len(w) >= 3 and w not in STOPWORDS]
    combined = " ".join(c.get("chunk_text", "") for c in top_chunks).lower()
    matched = [kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", combined)]
    coverage = len(matched) / len(keywords) if keywords else 1.0

    in_scope = True
    reason = "passed"
    if top_score < TOP_SCORE_THRESHOLD:
        in_scope, reason = False, "top_score_below_threshold"
    elif all_low:
        in_scope, reason = False, "all_scores_low"
    elif coverage < MIN_KEYWORD_COVERAGE:
        in_scope, reason = False, "low_keyword_coverage"

    return {"is_in_scope": in_scope, "reason": reason, "top_score": top_score,
            "keyword_coverage": coverage, "matched_keywords": matched}


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def build_grounded_prompt(query: str, chunks: list[dict]) -> str:
    """Build the legal QA prompt (mirrors rerank_crossencoder.py)."""
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


def generate_answer(query: str, chunks: list[dict]) -> str:
    prompt = build_grounded_prompt(query, chunks)
    return call_hf_inference(prompt)


# ---------------------------------------------------------------------------
# LLM-as-a-Judge: Faithfulness & Relevancy (lightweight, per-query)
# ---------------------------------------------------------------------------

def score_faithfulness(answer: str, context: str) -> dict[str, Any]:
    """
    Faithfulness via Claim Extraction + Verification.

    Step 1: Ask LLM to extract atomic claims from the answer.
    Step 2: For each claim, ask LLM whether it is supported by the context.
    Returns score in [0, 1] and claim-level details.
    """
    # Step 1 — Extract claims
    extract_prompt = (
        "Extract all distinct factual claims from the following answer as a JSON list of strings. "
        "Each claim should be a single, atomic statement. Output ONLY a JSON list, nothing else.\n\n"
        f"Answer:\n{answer}\n\nClaims:"
    )
    try:
        raw = call_hf_inference(extract_prompt, max_tokens=300, temperature=0.0)
        # Parse JSON list from LLM output
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        claims = json.loads(match.group()) if match else [answer]
        if not isinstance(claims, list) or not claims:
            claims = [answer]
        claims = [str(c) for c in claims if str(c).strip()]
    except Exception:
        claims = [answer]

    if not claims:
        return {"score": 1.0, "claims": [], "details": []}

    # Step 2 — Verify each claim
    verified = []
    for claim in claims[:8]:  # cap at 8 claims to limit API calls
        verify_prompt = (
            "Given the context below, is the following claim fully supported? "
            "Answer ONLY 'supported' or 'unsupported'.\n\n"
            f"Context:\n{context[:2000]}\n\nClaim: {claim}\n\nVerdict:"
        )
        try:
            verdict = call_hf_inference(verify_prompt, max_tokens=20, temperature=0.0).lower().strip()
            is_supported = "supported" in verdict and "unsupported" not in verdict
        except Exception:
            is_supported = True  # default to supported if API fails
        verified.append({"claim": claim, "supported": is_supported})

    supported_count = sum(1 for v in verified if v["supported"])
    score = supported_count / len(verified) if verified else 1.0

    return {"score": round(score, 3), "claims": claims, "details": verified}


def score_relevancy(query: str, answer: str, embed_model: SentenceTransformer) -> dict[str, Any]:
    """
    Relevancy via Alternate Query Generation + Cosine Similarity.

    Step 1: Ask LLM to generate 3 alternate queries the answer would satisfy.
    Step 2: Compute cosine similarity between original query and each alternate.
    Returns mean similarity as the relevancy score.
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

    # Cosine similarity between original query and each alternate
    orig_emb = embed_model.encode([query])
    alt_embs = embed_model.encode(alt_queries)
    sims = cosine_similarity(orig_emb, alt_embs)[0].tolist()

    return {
        "score": round(float(np.mean(sims)), 3),
        "alt_queries": alt_queries,
        "similarities": [round(s, 3) for s in sims],
    }


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------

def run_rag_pipeline(
    query: str,
    pinecone_key: str,
    chunks_df: pd.DataFrame,
    bm25: BM25Okapi,
    embed_model: SentenceTransformer,
    cross_encoder: CrossEncoder,
    retrieval_mode: str = "hybrid",
) -> dict[str, Any]:
    """
    End-to-end RAG pipeline.

    retrieval_mode: "semantic", "bm25", "hybrid"
    """
    t0 = time.time()
    allowed_ids = set(chunks_df["chunk_id"].tolist())

    # --- Retrieval ---
    sem_ids, bm25_ids, fused = [], [], []

    if retrieval_mode in ("semantic", "hybrid"):
        sem_ids = semantic_search(query, embed_model, pinecone_key)
        sem_ids = [i for i in sem_ids if i in allowed_ids]

    if retrieval_mode in ("bm25", "hybrid"):
        bm25_ids = bm25_search(chunks_df, bm25, query)

    if retrieval_mode == "hybrid":
        fused = reciprocal_rank_fusion(sem_ids, bm25_ids)
        candidate_ids = [cid for cid, _ in fused[:FUSED_TOP_K]]
    elif retrieval_mode == "semantic":
        candidate_ids = sem_ids[:FUSED_TOP_K]
    else:
        candidate_ids = bm25_ids[:FUSED_TOP_K]

    # Build candidate chunk dicts for reranking
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

    # --- Rerank ---
    reranked = rerank(query, candidates, cross_encoder, top_k=RERANK_TOP_K)

    # --- Confidence gate ---
    answer_chunks = reranked[:ANSWER_TOP_K]
    confidence = assess_confidence(query, answer_chunks)

    # --- Answer generation ---
    if not confidence["is_in_scope"]:
        answer = FALLBACK_RESPONSE
    else:
        try:
            answer = generate_answer(query, answer_chunks)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            answer = f"Answer generation failed: {exc}"

    retrieval_time = round(time.time() - t0, 2)

    return {
        "query": query,
        "answer": answer,
        "retrieval_mode": retrieval_mode,
        "reranked_chunks": reranked,
        "answer_chunks": answer_chunks,
        "confidence": confidence,
        "semantic_ids": sem_ids[:10],
        "bm25_ids": bm25_ids[:10],
        "retrieval_time_s": retrieval_time,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Sindh Criminal Law RAG",
        page_icon="balance_scale",
        layout="wide",
    )

    st.title("Sindh Criminal Prosecution Law — RAG System")
    st.caption(
        "Hybrid retrieval (BM25 + Pinecone) with RRF fusion, cross-encoder reranking, "
        "and Llama-3.1-8B answer generation via Groq API."
    )

    # --- Sidebar: configuration ---
    with st.sidebar:
        st.header("Configuration")

        pinecone_key = os.environ.get("PINECONE_API_KEY", "")
        if not pinecone_key:
            pinecone_key = st.text_input("Pinecone API Key", type="password")

        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            groq_key = st.text_input("Groq API Key", type="password")
            if groq_key:
                os.environ["GROQ_API_KEY"] = groq_key

        st.divider()
        retrieval_mode = st.selectbox(
            "Retrieval Mode",
            ["hybrid", "semantic", "bm25"],
            index=0,
            help="hybrid = BM25 + Semantic with RRF fusion (recommended)",
        )

        run_eval = st.checkbox("Run Faithfulness & Relevancy scoring", value=False,
                               help="Adds ~30s per query via LLM-as-a-Judge calls")

        st.divider()
        st.markdown(
            "**Pipeline:** Hybrid Retrieval → RRF → Cross-Encoder → Confidence Gate → Llama-3.1-8B"
        )

    # --- Load models & data ---
    if not CHUNKS_CSV.exists():
        st.error(f"Chunk CSV not found at `{CHUNKS_CSV}`. Run the data pipeline first.")
        return

    chunks_df, bm25 = load_chunks_and_bm25(str(CHUNKS_CSV))
    embed_model = load_embed_model()
    cross_encoder = load_cross_encoder()

    st.info(f"Loaded **{len(chunks_df)}** retrievable legal chunks from {CHUNKS_CSV.name}")

    # --- Query input ---
    query = st.text_area(
        "Enter your legal question:",
        placeholder="e.g., What are the powers of the Prosecutor General under the Sindh Act?",
        height=80,
    )

    if st.button("Search & Generate Answer", type="primary", disabled=not query.strip()):
        if not pinecone_key and retrieval_mode in ("semantic", "hybrid"):
            st.error("Pinecone API key is required for semantic/hybrid retrieval.")
            return

        with st.spinner("Running RAG pipeline..."):
            result = run_rag_pipeline(
                query=query.strip(),
                pinecone_key=pinecone_key,
                chunks_df=chunks_df,
                bm25=bm25,
                embed_model=embed_model,
                cross_encoder=cross_encoder,
                retrieval_mode=retrieval_mode,
            )

        # ----- Display results -----
        st.divider()

        # Generated answer
        st.subheader("Generated Answer")
        confidence = result["confidence"]
        if confidence["is_in_scope"]:
            st.success(f"Confidence: HIGH  |  Top rerank score: {confidence['top_score']:.2f}  |  "
                       f"Keyword coverage: {confidence['keyword_coverage']:.0%}")
        else:
            st.warning(f"Confidence: LOW ({confidence['reason']})  |  "
                       f"Top rerank score: {confidence['top_score']:.2f}")

        st.markdown(result["answer"])

        # Retrieved context
        st.subheader("Retrieved Context")
        for i, chunk in enumerate(result["reranked_chunks"], 1):
            with st.expander(
                f"#{i}  |  {chunk['chunk_id']}  |  score: {chunk.get('rerank_score', 0):.3f}  |  {chunk['title']}",
                expanded=(i <= 3),
            ):
                st.text(chunk["chunk_text"])

        # Retrieval metadata
        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieval Mode", result["retrieval_mode"].upper())
        col2.metric("Chunks Retrieved", len(result["reranked_chunks"]))
        col3.metric("Pipeline Time", f"{result['retrieval_time_s']}s")

        # ----- Faithfulness & Relevancy (optional) -----
        if run_eval:
            st.divider()
            st.subheader("LLM-as-a-Judge Evaluation")

            context_text = "\n\n".join(c["chunk_text"] for c in result["answer_chunks"])

            eval_col1, eval_col2 = st.columns(2)

            with eval_col1:
                with st.spinner("Scoring Faithfulness (claim extraction & verification)..."):
                    faith = score_faithfulness(result["answer"], context_text)
                st.metric("Faithfulness Score", f"{faith['score']:.1%}")
                if faith["details"]:
                    for d in faith["details"]:
                        icon = "white_check_mark" if d["supported"] else "x"
                        st.markdown(f"- :{icon}: {d['claim']}")

            with eval_col2:
                with st.spinner("Scoring Relevancy (alternate query gen + cosine sim)..."):
                    rel = score_relevancy(query.strip(), result["answer"], embed_model)
                st.metric("Relevancy Score", f"{rel['score']:.1%}")
                if rel["alt_queries"]:
                    for q, s in zip(rel["alt_queries"], rel["similarities"]):
                        st.markdown(f"- *\"{q}\"* — sim: {s:.3f}")


if __name__ == "__main__":
    main()
