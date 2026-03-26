"""
End-to-end hybrid retrieval with Reciprocal Rank Fusion (RRF).

What this script does:
1) Loads your local chunk CSV for BM25
2) Runs semantic retrieval from Pinecone
3) Runs BM25 retrieval locally
4) Fuses both ranked lists with RRF
5) Prints top 10 fused chunk_ids with scores

Dependencies:
    pip install pandas rank-bm25 pinecone sentence-transformers
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
INDEX_NAME = "legal-rag-index-filtered"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "rag_chunks_with_retrievable_flag.csv"

# How many candidates each retriever returns before fusion.
SEMANTIC_TOP_K = 20
BM25_TOP_K = 20

# How many final fused results to print.
FUSED_TOP_K = 10


def legal_tokenize(text: str) -> list[str]:
    """
    Simple legal-friendly tokenizer for BM25.

    Keeps:
    - lowercased words
    - hyphenated legal terms
    - section-like numbers such as 302(a), 4(2), 10(1)(b)
    """
    text = str(text).lower()
    pattern = r"\d+[a-z]?(?:\([a-z0-9]+\))*|[a-z]+(?:-[a-z]+)*"
    return re.findall(pattern, text)


def load_chunks(csv_path: Path) -> pd.DataFrame:
    """
    Load chunk CSV and validate required columns.

    If an 'is_retrievable' column exists, keep only rows where it is True.
    This avoids retrieving appendix/statistical chunks in both BM25 and fusion.
    """
    df = pd.read_csv(csv_path)

    required_cols = ["chunk_id", "chunk_text", "title", "doc_id", "source_filename"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df["chunk_id"] = df["chunk_id"].astype(str)
    df["chunk_text"] = df["chunk_text"].fillna("").astype(str)

    if "is_retrievable" in df.columns:
        before = len(df)
        retrievable_mask = (
            df["is_retrievable"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        df = df[retrievable_mask].copy()
        after = len(df)
        print(f"Filtered by is_retrievable=True: kept {after} of {before} chunks")
    else:
        print("Column 'is_retrievable' not found. Using all chunks.")

    return df


def build_bm25(df: pd.DataFrame) -> BM25Okapi:
    """Build BM25 index from chunk_text column."""
    tokenized_corpus = [legal_tokenize(text) for text in df["chunk_text"]]
    return BM25Okapi(tokenized_corpus)


def get_bm25_ranked_ids(df: pd.DataFrame, bm25: BM25Okapi, query: str, top_k: int) -> list[str]:
    """Return top BM25 chunk_ids for the query."""
    query_tokens = legal_tokenize(query)
    scores = bm25.get_scores(query_tokens)

    result_df = df[["chunk_id"]].copy()
    result_df["bm25_score"] = scores
    top_df = result_df.sort_values("bm25_score", ascending=False).head(top_k)
    return top_df["chunk_id"].tolist()


def get_semantic_ranked_ids(
    query: str,
    api_key: str,
    index_name: str,
    model_name: str,
    top_k: int,
) -> list[str]:
    """
    Query Pinecone and return ranked chunk_ids.

    It tries metadata['chunk_id'] first, then falls back to Pinecone match id.
    """
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query).tolist()

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    ranked_ids: list[str] = []
    for match in response.get("matches", []):
        metadata = match.get("metadata", {}) or {}
        chunk_id = metadata.get("chunk_id", match.get("id"))
        if chunk_id is not None:
            ranked_ids.append(str(chunk_id))

    return ranked_ids


def filter_ids_by_allowed_set(ranked_ids: list[str], allowed_ids: set[str]) -> list[str]:
    """Keep ranked IDs that exist in allowed_ids, preserving order and removing duplicates."""
    filtered: list[str] = []
    seen: set[str] = set()
    for chunk_id in ranked_ids:
        if chunk_id in allowed_ids and chunk_id not in seen:
            filtered.append(chunk_id)
            seen.add(chunk_id)
    return filtered


def reciprocal_rank_fusion(
    semantic_ranked_ids: list[str],
    bm25_ranked_ids: list[str],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Combine two ranked chunk_id lists using Reciprocal Rank Fusion (RRF).

    Args:
        semantic_ranked_ids: Ranked chunk_ids from semantic retrieval (best first).
        bm25_ranked_ids: Ranked chunk_ids from BM25 retrieval (best first).
        k: RRF constant. 60 is a common default.

    Returns:
        A list of (chunk_id, fused_score), sorted by fused_score (highest first).
    """
    fused_scores: dict[str, float] = defaultdict(float)

    # Add contributions from the semantic ranking.
    for rank, chunk_id in enumerate(semantic_ranked_ids, start=1):
        fused_scores[chunk_id] += 1.0 / (k + rank)

    # Add contributions from the BM25 ranking.
    for rank, chunk_id in enumerate(bm25_ranked_ids, start=1):
        fused_scores[chunk_id] += 1.0 / (k + rank)

    # Sort by score descending; tie-break by chunk_id for stable output.
    fused_ranking = sorted(
        fused_scores.items(),
        key=lambda x: (-x[1], x[0]),
    )
    return fused_ranking


def print_top_results(fused_ranking: list[tuple[str, float]], top_k: int = 10) -> None:
    """Print top-k fused results in a readable format."""
    print("\nTop fused results (RRF):")
    print("=" * 50)

    for i, (chunk_id, score) in enumerate(fused_ranking[:top_k], start=1):
        print(f"{i:>2}. chunk_id={chunk_id:<20} fused_score={score:.6f}")


def print_top_results_with_metadata(
    fused_ranking: list[tuple[str, float]],
    chunks_df: pd.DataFrame,
    top_k: int = 10,
) -> None:
    """Print top-k fused chunk_ids with score + short title/preview."""
    lookup = chunks_df.set_index("chunk_id")

    print("\nTop fused results (with metadata):")
    print("=" * 90)
    for i, (chunk_id, score) in enumerate(fused_ranking[:top_k], start=1):
        if chunk_id in lookup.index:
            row = lookup.loc[chunk_id]
            title = str(row.get("title", "Unknown"))
            preview = str(row.get("chunk_text", ""))[:140].replace("\n", " ")
        else:
            title = "Not found in CSV"
            preview = ""

        print(f"{i:>2}. chunk_id={chunk_id:<20} fused_score={score:.6f}")
        print(f"    title: {title}")
        if preview:
            print(f"    preview: {preview}")


def main() -> None:
    print("Hybrid Retrieval + RRF")

    csv_input = input(f"Chunk CSV path (Enter for default: {DEFAULT_CSV_PATH}): ").strip()
    csv_path = Path(csv_input) if csv_input else DEFAULT_CSV_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read Pinecone key from environment for safer usage.
    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not api_key:
        api_key = input("Enter Pinecone API key: ").strip()
    if not api_key:
        print("Pinecone API key is required. Exiting.")
        return

    query = input("\nEnter legal query: ").strip()
    if not query:
        print("Empty query. Exiting.")
        return

    print("\nLoading chunks and building BM25 index...")
    chunks_df = load_chunks(csv_path)
    bm25 = build_bm25(chunks_df)
    allowed_chunk_ids = set(chunks_df["chunk_id"].tolist())

    print("Running semantic retrieval (Pinecone)...")
    semantic_ranked_ids = get_semantic_ranked_ids(
        query=query,
        api_key=api_key,
        index_name=INDEX_NAME,
        model_name=MODEL_NAME,
        top_k=SEMANTIC_TOP_K,
    )
    semantic_ranked_ids = filter_ids_by_allowed_set(semantic_ranked_ids, allowed_chunk_ids)

    print("Running BM25 retrieval (local index)...")
    bm25_ranked_ids = get_bm25_ranked_ids(
        df=chunks_df,
        bm25=bm25,
        query=query,
        top_k=BM25_TOP_K,
    )

    print("Fusing semantic + BM25 results with RRF...")
    fused_ranking = reciprocal_rank_fusion(
        semantic_ranked_ids=semantic_ranked_ids,
        bm25_ranked_ids=bm25_ranked_ids,
        k=60,
    )

    # Requirement output: top 10 chunk_ids with fused scores.
    print_top_results(fused_ranking, top_k=FUSED_TOP_K)

    # Extra beginner-friendly view (can help inspect the result quality).
    print_top_results_with_metadata(fused_ranking, chunks_df, top_k=FUSED_TOP_K)


if __name__ == "__main__":
    main()
