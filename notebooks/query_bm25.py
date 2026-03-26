"""
Beginner-friendly BM25 search for legal RAG chunks.

What this script does:
1) Loads a chunk CSV (must include chunk_id, chunk_text, title, doc_id, source_filename)
2) Preprocesses text with a simple legal-friendly tokenizer
3) Builds a BM25 index using rank-bm25
4) Accepts a user query
5) Prints top 5 matches with chunk_id, BM25 score, title, and chunk preview

Install dependencies first:
    pip install pandas rank-bm25
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi


# Change this if your CSV is elsewhere.
# By default we use the flagged file produced by filter_retrievable_chunks.py.
DEFAULT_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "rag_chunks_with_retrievable_flag.csv"
TOP_K = 5


def legal_tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 in a way that is simple and legal-text friendly.

    Rules:
    - lowercase everything
    - keep words (including hyphenated forms like "cross-examination")
    - keep section-like numbers where possible (e.g., "302", "302(a)", "4(2)")

    This is intentionally simple so beginners can understand and adjust it.
    """
    text = str(text).lower()

    # Pattern parts:
    # 1) section-like numeric tokens: 302, 302(a), 4(2), 10(1)(b)
    # 2) word tokens, including hyphenated words
    pattern = r"\d+[a-z]?(?:\([a-z0-9]+\))*|[a-z]+(?:-[a-z]+)*"
    tokens = re.findall(pattern, text)

    return tokens


def load_chunks(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV and validate required columns.

    If 'is_retrievable' exists, keep only rows where it is True.
    This excludes appendix/statistical chunks marked as low value.
    """
    df = pd.read_csv(csv_path)

    required_cols = ["chunk_id", "chunk_text", "title", "doc_id", "source_filename"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Replace missing chunk text with empty strings to avoid tokenization errors.
    df["chunk_text"] = df["chunk_text"].fillna("").astype(str)

    if "is_retrievable" in df.columns:
        before = len(df)
        # Accept bool True as well as string/number representations.
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


def build_bm25(df: pd.DataFrame) -> tuple[BM25Okapi, list[list[str]]]:
    """Tokenize all chunks and build BM25 index."""
    tokenized_corpus = [legal_tokenize(text) for text in df["chunk_text"]]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def search_bm25(df: pd.DataFrame, bm25: BM25Okapi, query: str, top_k: int = TOP_K) -> pd.DataFrame:
    """Return top-k BM25 matches as a DataFrame."""
    query_tokens = legal_tokenize(query)
    scores = bm25.get_scores(query_tokens)

    result_df = df.copy()
    result_df["bm25_score"] = scores

    # Keep only top-k rows by score.
    top_df = result_df.sort_values("bm25_score", ascending=False).head(top_k)

    # Keep only requested output columns.
    return top_df[["chunk_id", "bm25_score", "title", "chunk_text"]]


def print_results(query: str, results: pd.DataFrame) -> None:
    """Pretty-print top matches with chunk preview."""
    print("\n" + "=" * 80)
    print(f"Top {len(results)} BM25 results for query: {query}")
    print("=" * 80)

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        preview = row["chunk_text"][:300].replace("\n", " ").strip()

        print(f"\nResult #{rank}")
        print(f"chunk_id   : {row['chunk_id']}")
        print(f"BM25 score : {row['bm25_score']:.4f}")
        print(f"title      : {row['title']}")
        print(f"preview    : {preview}")


def main() -> None:
    csv_input = input(f"Enter chunk CSV path (press Enter for default: {DEFAULT_CSV_PATH}): ").strip()
    csv_path = Path(csv_input) if csv_input else DEFAULT_CSV_PATH

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print("\nLoading chunks...")
    df = load_chunks(csv_path)

    print("Building BM25 index...")
    bm25, _ = build_bm25(df)

    print("Index ready. Type your legal query.")
    query = input("\nQuery: ").strip()
    if not query:
        print("Empty query. Exiting.")
        return

    results = search_bm25(df, bm25, query, top_k=TOP_K)
    print_results(query, results)


if __name__ == "__main__":
    main()
