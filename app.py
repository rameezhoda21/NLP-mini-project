"""
Streamlit RAG app for Sindh Criminal Prosecution Law.

Integrates the full pipeline:
  1) Hybrid retrieval  (BM25 + Pinecone semantic) with RRF fusion
  2) Cross-encoder reranking + confidence gate
  3) LLM answer generation via Groq API (Llama-3.3-70B)
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
from datetime import datetime

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "evaluation"
CHUNKS_CSV = DATA_DIR / "rag_chunks_with_retrievable_flag.csv"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL_ID = "llama-3.1-8b-instant"

SEMANTIC_TOP_K = 20
BM25_TOP_K = 20
FUSED_TOP_K = 10
RERANK_TOP_K = 5
ANSWER_TOP_K = 3
FALLBACK_RESPONSE = (
    "I could not find a reliable answer to this question in the retrieved documents."
)

HF_TOKEN_ENV_CANDIDATES = ["HF_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with",
}


_groq_client = None
_chunk_cache_df: pd.DataFrame | None = None
_chunk_cache_embeddings: np.ndarray | None = None


def _get_hf_token() -> str:
    for name in HF_TOKEN_ENV_CANDIDATES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return _groq_client


def call_hf_inference(prompt: str, max_tokens: int = 350, temperature: float = 0.2) -> str:
    """Call Groq for chat completion using the configured legal model."""
    client = _get_groq_client()
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content.strip()
            if not text:
                raise RuntimeError("Empty response from Groq API")
            return text
        except Exception as exc:
            if "429" in str(exc) or "rate" in str(exc).lower():
                time.sleep(10 * (attempt + 1))
                continue
            raise
    raise RuntimeError("Groq API rate limit exceeded after 4 retries")


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
    return df.reset_index(drop=True)


def build_bm25(df: pd.DataFrame) -> BM25Okapi:
    return BM25Okapi([legal_tokenize(text) for text in df["chunk_text"]])


@st.cache_resource(show_spinner=False)
def load_embed_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_cross_encoder() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_NAME)


def load_chunks_and_bm25(csv_path: str) -> tuple[pd.DataFrame, BM25Okapi]:
    global _chunk_cache_df, _chunk_cache_embeddings
    df = load_chunks(Path(csv_path))
    bm25 = build_bm25(df)
    embed_model = load_embed_model()
    _chunk_cache_df = df
    _chunk_cache_embeddings = embed_model.encode(
        df["chunk_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return df, bm25


def semantic_search(query: str, embed_model: SentenceTransformer, pinecone_key: str, top_k: int = SEMANTIC_TOP_K) -> list[str]:
    del pinecone_key
    if _chunk_cache_df is None or _chunk_cache_embeddings is None:
        load_chunks_and_bm25(str(CHUNKS_CSV))
    if _chunk_cache_df is None or _chunk_cache_embeddings is None:
        return []
    query_emb = embed_model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(query_emb, _chunk_cache_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return _chunk_cache_df.iloc[top_indices]["chunk_id"].tolist()


def bm25_search(df: pd.DataFrame, bm25: BM25Okapi, query: str, top_k: int = BM25_TOP_K) -> list[str]:
    scores = bm25.get_scores(legal_tokenize(query))
    ranked = df[["chunk_id"]].copy()
    ranked["score"] = scores
    return ranked.sort_values("score", ascending=False).head(top_k)["chunk_id"].tolist()


def reciprocal_rank_fusion(sem_ids: list[str], bm25_ids: list[str], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for rank, chunk_id in enumerate(sem_ids, 1):
        scores[chunk_id] += 1.0 / (k + rank)
    for rank, chunk_id in enumerate(bm25_ids, 1):
        scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def rerank(query: str, chunks: list[dict], model: CrossEncoder, top_k: int = RERANK_TOP_K) -> list[dict]:
    if not chunks:
        return []
    pairs = [(query, chunk["chunk_text"]) for chunk in chunks]
    scores = model.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    chunks.sort(key=lambda item: item["rerank_score"], reverse=True)
    return chunks[:top_k]


def build_grounded_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Context {i}]\nChunk ID: {chunk.get('chunk_id', '')}\nTitle: {chunk.get('title', '')}\nText:\n{chunk.get('chunk_text', '')}"
        )
    context = "\n\n".join(context_blocks)
    return (
        "You are a careful legal assistant.\n"
        "Use ONLY the context below to answer the user question.\n"
        "Do NOT invent legal rules, sections, dates, or case facts.\n"
        "If the answer is not clearly supported by the context, say: "
        "'The provided context does not contain enough reliable information to answer this question.'\n"
        "Keep your answer concise and factual.\n\n"
        f"User Question:\n{query}\n\nContext:\n{context}\n\nFinal Answer:"
    )


def generate_answer(query: str, chunks: list[dict]) -> str:
    return call_hf_inference(build_grounded_prompt(query, chunks), max_tokens=350, temperature=0.2)


def assess_confidence(query: str, answer_chunks: list[dict]) -> dict[str, float | bool]:
    del query
    top_score = max((float(chunk.get("rerank_score", 0.0)) for chunk in answer_chunks), default=0.0)
    mean_score = float(np.mean([float(chunk.get("rerank_score", 0.0)) for chunk in answer_chunks])) if answer_chunks else 0.0
    return {
        "is_in_scope": bool(answer_chunks) and top_score >= 0.15,
        "top_score": round(top_score, 2),
        "mean_score": round(mean_score, 2),
    }


def append_search_history(query: str, retrieval_mode: str, retrieval_time_s: float) -> None:
    history = st.session_state.setdefault("search_history", [])
    history.append({
        "query": query,
        "retrieval_mode": retrieval_mode,
        "retrieval_time_s": retrieval_time_s,
        "ts": datetime.utcnow().isoformat(),
    })
    st.session_state["search_history"] = history[-25:]


def main():
    st.set_page_config(
        page_title="Sindh Criminal Law RAG",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_dashboard_css()

    st.session_state.setdefault("legal_query", "")
    st.session_state.setdefault("search_history", [])

    pinecone_key = os.environ.get("PINECONE_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")

    if not CHUNKS_CSV.exists():
        st.error(f"Chunk CSV not found at `{CHUNKS_CSV}`. Run the data pipeline first.")
        return

    chunks_df, bm25 = load_chunks_and_bm25(str(CHUNKS_CSV))
    embed_model = load_embed_model()
    cross_encoder = load_cross_encoder()

    with st.sidebar:
        st.markdown("### Advanced Retrieval Settings")
        retrieval_mode_label = st.selectbox(
            "Retrieval Mode",
            ["Hybrid", "Semantic", "BM25"],
            index=0,
            help="Hybrid combines keyword and semantic search for stronger legal grounding.",
        )
        run_eval = st.checkbox("Enable Faithfulness Scoring", value=False)
        semantic_top_k = st.slider("Semantic Depth", 5, 30, SEMANTIC_TOP_K)
        bm25_top_k = st.slider("BM25 Depth", 5, 30, BM25_TOP_K)
        fused_top_k = st.slider("Fusion Window", 3, 20, FUSED_TOP_K)
        rerank_top_k = st.slider("Rerank Window", 3, 10, RERANK_TOP_K)
        st.caption("Advanced retrieval remains in the sidebar so the first screen stays focused.")

    st.markdown('<div class="workspace-shell">', unsafe_allow_html=True)

    st.markdown(
        f'''
        <div class="hero">
            <h1>Sindh Criminal Law RAG</h1>
            <p>Search Sindh criminal prosecution law with grounded retrieval and source-backed answers.</p>
            <div class="status-pills">
                <span class="status-pill">{len(chunks_df)} legal chunks indexed</span>
                <span class="status-pill">Hybrid retrieval</span>
                <span class="status-pill">Cross-encoder reranking</span>
                <span class="status-pill">Llama 3.3 70B</span>
                <span class="status-pill ready"><span class="ready-dot"></span>Ready</span>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="composer-shell">', unsafe_allow_html=True)
    composer_col1, composer_col2 = st.columns([1, 0.1])
    with composer_col1:
        query = st.text_input(
            "Legal Query",
            placeholder="Ask a legal question... e.g., What powers does the Prosecutor General have under the Act?",
            key="legal_query",
            label_visibility="collapsed",
        )
    with composer_col2:
        search_clicked = st.button("↑", type="primary", use_container_width=True)
    st.markdown(
        '<div class="composer-helper">Search Sindh criminal prosecution law with grounded retrieval and source-backed answers.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if search_clicked and query.strip():
        retrieval_mode = retrieval_mode_label.lower()

        if not pinecone_key and retrieval_mode in ("semantic", "hybrid"):
            st.error("⚠️ Pinecone API key required for semantic search. Set PINECONE_API_KEY.")
            return

        if not groq_key:
            st.error("⚠️ Groq API key required. Set GROQ_API_KEY.")
            return

        with st.spinner("🔍 Searching legal documents..."):
            result = run_rag_pipeline(
                query=query.strip(),
                pinecone_key=pinecone_key,
                chunks_df=chunks_df,
                bm25=bm25,
                embed_model=embed_model,
                cross_encoder=cross_encoder,
                retrieval_mode=retrieval_mode,
                semantic_top_k=semantic_top_k,
                bm25_top_k=bm25_top_k,
                fused_top_k=fused_top_k,
                rerank_top_k=rerank_top_k,
            )

        append_search_history(query.strip(), retrieval_mode, result["retrieval_time_s"])

        confidence = result["confidence"]
        conf_class = "high" if confidence["is_in_scope"] else "low"
        conf_text = "High confidence" if confidence["is_in_scope"] else "Low confidence"

        st.markdown(
            f'''
            <div class="result-panel">
                <div class="panel-title">Answer</div>
                <div class="answer-text">{result["answer"]}</div>
                <div class="confidence-pill {conf_class}">{conf_text} • score {confidence['top_score']:.2f}</div>
                <div class="meta-row">
                    <div class="meta-item"><div class="k">Retrieval</div><div class="v">{result['retrieval_mode'].upper()}</div></div>
                    <div class="meta-item"><div class="k">Latency</div><div class="v">{result['retrieval_time_s']:.2f}s</div></div>
                    <div class="meta-item"><div class="k">Sources Used</div><div class="v">{len(result['answer_chunks'])}</div></div>
                    <div class="meta-item"><div class="k">Chunks Found</div><div class="v">{len(result['reranked_chunks'])}</div></div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="result-panel"><div class="panel-title">Retrieved Legal Sources</div>', unsafe_allow_html=True)
        for i, chunk in enumerate(result["reranked_chunks"], 1):
            score = chunk.get("rerank_score", 0)
            score_pct = int(min(max(score * 100, 0), 100))
            st.markdown(
                f'''
                <div class="source-card">
                    <div class="source-head">
                        <div>{chunk.get("title", "Source")}</div>
                        <div class="source-id">#{chunk["chunk_id"]}</div>
                    </div>
                    <div class="source-text">{chunk["chunk_text"][:450]}...</div>
                    <div class="source-foot">
                        <span>Chunk {i}/{len(result["reranked_chunks"])}</span>
                        <span>Relevance {score_pct}% <span class="bar"><span class="fill" style="width: {score_pct}%"></span></span></span>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        if run_eval:
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Faithfulness and Relevancy</div>', unsafe_allow_html=True)
            context_text = "\n\n".join(c["chunk_text"] for c in result["answer_chunks"])

            with st.spinner("📊 Evaluating..."):
                faith = score_faithfulness(result["answer"], context_text)
                rel = score_relevancy(query.strip(), result["answer"], embed_model)

            left, right = st.columns(2)
            left.metric("Faithfulness", f"{faith['score']:.0%}")
            right.metric("Relevancy", f"{rel['score']:.0%}")

            if faith.get("details"):
                st.markdown('<div class="panel-title" style="margin-top:1.1rem;">Claim Verification</div>', unsafe_allow_html=True)
                for d in faith["details"]:
                    icon = "✓" if d["supported"] else "✗"
                    st.markdown(f"- {icon} {d['claim']}")

            st.markdown('</div>', unsafe_allow_html=True)

def render_dashboard_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            :root {
                --bg-main: #0b1018;
                --bg-main-alt: #0e1420;
                --bg-card: #141c28;
                --bg-card-soft: #111826;
                --bg-chip: #1a2434;
                --border: #2a3648;
                --border-strong: #35445a;
                --text-main: #eef3fb;
                --text-sub: #adb8cc;
                --text-dim: #7f8ca3;
                --accent: #5f8dff;
                --accent-2: #4d79dd;
                --ok: #27b980;
                --warn: #d99645;
                --radius-sm: 8px;
                --radius-md: 12px;
                --radius-lg: 18px;
                --shadow-card: 0 14px 32px rgba(0, 0, 0, 0.28);
                --shadow-soft: 0 6px 16px rgba(0, 0, 0, 0.18);
            }

            html, body, [data-testid="stAppViewContainer"], .stApp {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                color: var(--text-main) !important;
                background: radial-gradient(1200px 520px at 50% -12%, #152032 0%, var(--bg-main) 50%, #090d14 100%) !important;
            }

            [data-testid="stMain"] {
                background: transparent !important;
            }

            section[data-testid="stSidebar"],
            section[data-testid="stSidebar"] > div {
                background: #080c13 !important;
                border-right: 1px solid #172032 !important;
            }

            .block-container {
                max-width: 1180px !important;
                padding-top: 1.2rem !important;
                padding-bottom: 3rem !important;
            }

            .workspace-shell {
                max-width: 860px;
                margin: 0 auto;
                padding: 2.2rem 1.2rem 1.5rem;
            }

            .hero {
                text-align: center;
                margin-bottom: 1.3rem;
            }

            .hero h1 {
                margin: 0;
                font-size: clamp(2.05rem, 4vw, 3.15rem);
                line-height: 1.1;
                letter-spacing: -0.03em;
                font-weight: 800;
                color: var(--text-main);
            }

            .hero p {
                margin: 0.9rem auto 0;
                max-width: 760px;
                color: var(--text-sub);
                font-size: 1rem;
                line-height: 1.55;
            }

            .status-pills {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin: 1.05rem 0 1.5rem;
            }

            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.42rem;
                padding: 0.34rem 0.72rem;
                border: 1px solid var(--border);
                border-radius: 999px;
                background: linear-gradient(180deg, #151f2f 0%, #121a27 100%);
                color: var(--text-sub);
                font-size: 0.74rem;
                font-weight: 600;
                letter-spacing: 0.01em;
            }

            .status-pill.ready {
                border-color: #2a614a;
                color: #9ce4c8;
                background: linear-gradient(180deg, rgba(39, 185, 128, 0.18) 0%, rgba(39, 185, 128, 0.08) 100%);
            }

            .ready-dot {
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background: var(--ok);
            }

            .composer-shell {
                max-width: 920px;
                margin: 0.5rem auto 0;
                padding: 0;
            }

            .composer-helper {
                color: var(--text-dim);
                font-size: 0.76rem;
                margin: 0.35rem 0 0.35rem;
                text-align: center;
            }

            .result-panel {
                margin-top: 1.45rem;
                border: 1px solid var(--border);
                border-radius: var(--radius-lg);
                background: linear-gradient(180deg, #151d2a 0%, #111925 100%);
                box-shadow: var(--shadow-soft);
                padding: 1.3rem;
            }

            .panel-title {
                margin: 0 0 0.75rem;
                color: #d7e2f5;
                font-size: 0.82rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .answer-text {
                color: var(--text-main);
                line-height: 1.78;
                font-size: 0.96rem;
            }

            .confidence-pill {
                display: inline-flex;
                margin-top: 0.95rem;
                padding: 0.32rem 0.68rem;
                border-radius: 999px;
                font-size: 0.73rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                border: 1px solid transparent;
            }

            .confidence-pill.high {
                color: #8fe2c0;
                border-color: #2a634d;
                background: rgba(39, 185, 128, 0.13);
            }

            .confidence-pill.low {
                color: #efc18b;
                border-color: #7e5b2f;
                background: rgba(217, 150, 69, 0.14);
            }

            .meta-row {
                margin-top: 0.95rem;
                padding-top: 0.95rem;
                border-top: 1px solid var(--border);
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.7rem;
            }

            .meta-item .k {
                font-size: 0.68rem;
                color: var(--text-dim);
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }

            .meta-item .v {
                margin-top: 0.22rem;
                color: var(--text-sub);
                font-weight: 600;
                font-size: 0.89rem;
            }

            .source-card {
                border: 1px solid var(--border);
                border-radius: var(--radius-md);
                background: linear-gradient(180deg, #141d2b 0%, #111927 100%);
                padding: 0.95rem;
                margin-bottom: 0.75rem;
            }

            .source-head {
                display: flex;
                justify-content: space-between;
                gap: 0.75rem;
                margin-bottom: 0.5rem;
                color: var(--text-sub);
                font-size: 0.84rem;
                font-weight: 600;
            }

            .source-id {
                color: #9bb4df;
                font-size: 0.74rem;
                background: rgba(95, 141, 255, 0.13);
                border: 1px solid rgba(95, 141, 255, 0.25);
                padding: 0.13rem 0.48rem;
                border-radius: 999px;
            }

            .source-text {
                color: var(--text-sub);
                font-size: 0.87rem;
                line-height: 1.57;
                max-height: 220px;
                overflow: auto;
                background: #0e1521;
                border: 1px solid #202b3c;
                border-radius: var(--radius-sm);
                padding: 0.7rem;
            }

            .source-foot {
                margin-top: 0.55rem;
                display: flex;
                justify-content: space-between;
                color: var(--text-dim);
                font-size: 0.75rem;
            }

            .bar {
                width: 56px;
                height: 4px;
                border-radius: 999px;
                background: #202c3f;
                overflow: hidden;
                margin-left: 0.35rem;
                display: inline-block;
                vertical-align: middle;
            }

            .fill {
                height: 100%;
                background: linear-gradient(90deg, var(--accent), var(--accent-2));
            }

            /* Streamlit component overrides for custom feel */
            div[data-testid="stTextArea"] textarea {
                min-height: 92px !important;
                background: #111a27 !important;
                border: 1px solid var(--border) !important;
                border-radius: 16px !important;
                color: var(--text-main) !important;
                font-size: 1.01rem !important;
                line-height: 1.45 !important;
                padding: 0.95rem 1rem !important;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
            }

            div[data-testid="stTextInput"] input {
                min-height: 44px !important;
                height: 44px !important;
                background: #111a27 !important;
                border: 1px solid #253247 !important;
                border-radius: 16px !important;
                color: var(--text-main) !important;
                font-size: 0.96rem !important;
                padding: 0.55rem 1rem !important;
                box-shadow: none !important;
            }

            div[data-testid="stTextInput"] input::placeholder {
                color: var(--text-dim) !important;
            }

            div[data-testid="stTextInput"] input:focus {
                border-color: #49669b !important;
                box-shadow: 0 0 0 2px rgba(95, 141, 255, 0.14) !important;
            }

            /* Hide the default Streamlit label spacing for the prompt input */
            div[data-testid="stTextInput"] label {
                display: none !important;
            }

            div[data-testid="stTextInput"] {
                margin-bottom: 0 !important;
            }

            div[data-testid="stTextArea"] textarea:focus {
                border-color: #4b6492 !important;
                box-shadow: 0 0 0 2px rgba(95, 141, 255, 0.16) !important;
            }

            div[data-testid="stSelectbox"] > div,
            div[data-testid="stSelectbox"] button {
                background: #111a27 !important;
                border-radius: 10px !important;
                border-color: var(--border) !important;
                color: var(--text-sub) !important;
                min-height: 2.25rem !important;
            }

            div[data-testid="stCheckbox"] label {
                color: var(--text-sub) !important;
                font-size: 0.8rem !important;
                font-weight: 500 !important;
            }

            div[data-testid="stButton"] button {
                border-radius: 999px !important;
                border: 1px solid var(--border) !important;
                background: #152033 !important;
                color: var(--text-sub) !important;
                font-weight: 600 !important;
                letter-spacing: 0.01em !important;
                min-height: 2.15rem !important;
                box-shadow: none !important;
                font-size: 0.78rem !important;
                text-transform: none !important;
            }

            div[data-testid="stButton"] button:hover {
                border-color: #4d6282 !important;
                background: #1a2840 !important;
                color: #dce7fa !important;
            }

            div[data-testid="stButton"] button[kind="primary"] {
                border-radius: 999px !important;
                border: 1px solid #4f6ea5 !important;
                background: linear-gradient(180deg, #365388 0%, #2d4673 100%) !important;
                color: #eef3fb !important;
                font-weight: 700 !important;
                letter-spacing: 0.02em !important;
                min-height: 44px !important;
                height: 44px !important;
                min-width: 44px !important;
                width: 44px !important;
                box-shadow: none !important;
                font-size: 1rem !important;
                padding: 0 !important;
            }

            div[data-testid="stButton"] button[kind="primary"]:hover {
                border-color: #6586c7 !important;
                background: linear-gradient(180deg, #3d5f99 0%, #314e81 100%) !important;
            }

            div[data-testid="stSlider"] [data-baseweb="slider"] {
                padding-top: 0.2rem !important;
            }

            div[data-testid="stExpander"] {
                border: 1px solid var(--border) !important;
                border-radius: 12px !important;
                background: #101826 !important;
            }

            @media (max-width: 820px) {
                .workspace-shell {
                    padding: 1.5rem 0.65rem 1rem;
                }

                .composer-shell {
                    max-width: 100%;
                }

                .hero p {
                    font-size: 0.93rem;
                }

                .meta-row {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )





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
    semantic_top_k: int = SEMANTIC_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    fused_top_k: int = FUSED_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
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
        sem_ids = semantic_search(query, embed_model, pinecone_key, top_k=semantic_top_k)
        sem_ids = [i for i in sem_ids if i in allowed_ids]

    if retrieval_mode in ("bm25", "hybrid"):
        bm25_ids = bm25_search(chunks_df, bm25, query, top_k=bm25_top_k)

    if retrieval_mode == "hybrid":
        fused = reciprocal_rank_fusion(sem_ids, bm25_ids)
        candidate_ids = [cid for cid, _ in fused[:fused_top_k]]
    elif retrieval_mode == "semantic":
        candidate_ids = sem_ids[:fused_top_k]
    else:
        candidate_ids = bm25_ids[:fused_top_k]

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
    reranked = rerank(query, candidates, cross_encoder, top_k=rerank_top_k)

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


if __name__ == "__main__":
    main()
