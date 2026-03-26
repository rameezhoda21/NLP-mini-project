"""
Cross-encoder re-ranking for candidate RAG chunks.

What this script does:
1) Loads a cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
2) Takes a user query and a list of candidate chunks
3) Scores each (query, chunk_text) pair
4) Re-ranks chunks by cross-encoder score
5) Prints top 5 re-ranked chunks

Expected candidate chunk format (list of dictionaries):
[
    {
        "chunk_id": "chunk_001",
        "title": "Some law title",
        "chunk_text": "Text of the legal chunk..."
    },
    ...
]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder


# Cross-encoder model requested by the user.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5

# -----------------------------
# Confidence gate settings
# -----------------------------
# Easy to tune: if top rerank score is below this, treat as low confidence.
TOP_SCORE_THRESHOLD = 1.0

# If ALL top chunks are <= this value, retrieval is likely weak/off-topic.
LOW_SCORE_THRESHOLD = 0.0

# Minimum fraction of important query keywords that should appear
# in retrieved text to trust the retrieval.
MIN_KEYWORD_COVERAGE = 0.30

FALLBACK_RESPONSE = (
    "I could not find a reliable answer to this question in the retrieved documents."
)


# Very small stopword list so we focus on important query terms.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with",
}


def load_candidate_chunks(json_path: Path) -> list[dict[str, Any]]:
    """
    Load candidate chunks from a JSON file.

    The JSON file must contain a list of objects with:
    - chunk_id
    - title
    - chunk_text
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Candidate JSON must be a list of chunk objects.")

    required_fields = {"chunk_id", "title", "chunk_text"}
    cleaned: list[dict[str, Any]] = []

    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item #{i} is not a JSON object.")

        missing = required_fields - set(item.keys())
        if missing:
            raise ValueError(f"Item #{i} is missing required fields: {sorted(missing)}")

        cleaned.append(
            {
                "chunk_id": str(item["chunk_id"]),
                "title": str(item["title"]),
                "chunk_text": str(item["chunk_text"]),
            }
        )

    return cleaned


def rerank_chunks(
    query: str,
    candidate_chunks: list[dict[str, Any]],
    model: CrossEncoder,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """
    Score and re-rank candidate chunks using a cross-encoder.

    Steps:
    - Build pairs: (query, chunk_text)
    - Predict a relevance score for each pair
    - Attach score to each chunk
    - Sort by score descending
    - Return top-k
    """
    if not query.strip():
        raise ValueError("Query is empty.")

    if not candidate_chunks:
        return []

    # Build pairs that the model expects.
    pairs = [(query, chunk["chunk_text"]) for chunk in candidate_chunks]

    # Predict relevance scores (higher usually means more relevant).
    scores = model.predict(pairs)

    # Attach each score to its chunk.
    scored_chunks: list[dict[str, Any]] = []
    for chunk, score in zip(candidate_chunks, scores):
        scored_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "chunk_text": chunk["chunk_text"],
                "rerank_score": float(score),
            }
        )

    # Sort from highest score to lowest score.
    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_chunks[:top_k]


def extract_important_query_keywords(query: str) -> list[str]:
    """
    Extract simple important keywords from a query.

    We keep alphabetical words with length >= 3 and remove common stopwords.
    This is intentionally simple and beginner-friendly.
    """
    words = re.findall(r"[a-zA-Z]+", query.lower())
    keywords = [w for w in words if len(w) >= 3 and w not in STOPWORDS]

    # Keep order while removing duplicates.
    seen: set[str] = set()
    unique_keywords: list[str] = []
    for kw in keywords:
        if kw not in seen:
            unique_keywords.append(kw)
            seen.add(kw)

    return unique_keywords


def keyword_coverage_in_chunks(keywords: list[str], chunks: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """
    Check how many important query keywords are found in top chunk texts.

    Returns:
    - coverage ratio in [0, 1]
    - list of matched keywords
    """
    if not keywords:
        return 1.0, []

    combined_text = " ".join(chunk.get("chunk_text", "") for chunk in chunks).lower()

    matched = [kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", combined_text)]
    coverage = len(matched) / len(keywords)
    return coverage, matched


def assess_retrieval_confidence(
    query: str,
    top_chunks: list[dict[str, Any]],
    top_score_threshold: float = TOP_SCORE_THRESHOLD,
    low_score_threshold: float = LOW_SCORE_THRESHOLD,
    min_keyword_coverage: float = MIN_KEYWORD_COVERAGE,
) -> dict[str, Any]:
    """
    Decide whether retrieval is in-scope (high confidence) or out-of-scope (low confidence).

    Checks requested by user:
    - top rerank score
    - whether all top results are low/negative
    - keyword presence in retrieved chunks
    """
    if not top_chunks:
        return {
            "is_in_scope": False,
            "reason": "no_retrieved_chunks",
            "top_score": float("-inf"),
            "all_low_or_negative": True,
            "keyword_coverage": 0.0,
            "matched_keywords": [],
        }

    scores = [float(chunk.get("rerank_score", 0.0)) for chunk in top_chunks]
    top_score = max(scores)
    all_low_or_negative = all(score <= low_score_threshold for score in scores)

    keywords = extract_important_query_keywords(query)
    coverage, matched_keywords = keyword_coverage_in_chunks(keywords, top_chunks)

    is_in_scope = True
    reason = "passed_confidence_checks"

    if top_score < top_score_threshold:
        is_in_scope = False
        reason = "top_score_below_threshold"
    elif all_low_or_negative:
        is_in_scope = False
        reason = "all_top_scores_low_or_negative"
    elif coverage < min_keyword_coverage:
        is_in_scope = False
        reason = "insufficient_keyword_match"

    return {
        "is_in_scope": is_in_scope,
        "reason": reason,
        "top_score": top_score,
        "all_low_or_negative": all_low_or_negative,
        "keyword_coverage": coverage,
        "matched_keywords": matched_keywords,
        "keywords": keywords,
    }


def generate_answer_stub(query: str, top_chunks: list[dict[str, Any]]) -> str:
    """
    Placeholder for your real answer generation step.

    Replace this with your LLM call using top_chunks as context.
    """
    top_chunk_ids = [chunk["chunk_id"] for chunk in top_chunks]
    return (
        "Proceeding to answer generation with retrieved context. "
        f"Top chunk_ids: {top_chunk_ids}"
    )


def run_confidence_gate_and_respond(query: str, top_chunks: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """
    Apply confidence gate:
    - out-of-scope/low confidence -> fallback response
    - in-scope -> continue to answer generation
    """
    diagnostics = assess_retrieval_confidence(query, top_chunks)

    if not diagnostics["is_in_scope"]:
        response = FALLBACK_RESPONSE
    else:
        response = generate_answer_stub(query, top_chunks)

    return response, diagnostics


def print_top_results(query: str, top_chunks: list[dict[str, Any]]) -> None:
    """Print top re-ranked chunks in a beginner-friendly format."""
    print("\n" + "=" * 85)
    print(f"Top {len(top_chunks)} re-ranked chunks for query: {query}")
    print("=" * 85)

    if not top_chunks:
        print("No candidate chunks found.")
        return

    for rank, chunk in enumerate(top_chunks, start=1):
        preview = chunk["chunk_text"][:300].replace("\n", " ").strip()
        print(f"\nResult #{rank}")
        print(f"chunk_id      : {chunk['chunk_id']}")
        print(f"rerank score  : {chunk['rerank_score']:.4f}")
        print(f"title         : {chunk['title']}")
        print(f"preview       : {preview}")


def main() -> None:
    """
    Simple CLI flow:
    - Ask user for query
    - Ask path to candidate JSON list
    - Load model and re-rank
    """
    print("Cross-Encoder Re-Ranker")

    query = input("\nEnter your legal query: ").strip()
    if not query:
        print("Empty query. Exiting.")
        return

    json_input = input("Enter candidate chunk JSON path: ").strip()
    if not json_input:
        print("Candidate JSON path is required. Exiting.")
        return

    json_path = Path(json_input)
    if not json_path.exists():
        raise FileNotFoundError(f"Candidate JSON not found: {json_path}")

    print("\nLoading candidate chunks...")
    candidate_chunks = load_candidate_chunks(json_path)

    print(f"Loading model: {MODEL_NAME}")
    model = CrossEncoder(MODEL_NAME)

    print("Scoring and re-ranking chunks...")
    top_chunks = rerank_chunks(query, candidate_chunks, model, top_k=TOP_K)

    print_top_results(query, top_chunks)

    # Low-confidence / out-of-scope detection step.
    response, diagnostics = run_confidence_gate_and_respond(query, top_chunks)

    print("\n" + "=" * 85)
    print("Confidence Gate Diagnostics")
    print("=" * 85)
    print(f"Top rerank score          : {diagnostics['top_score']:.4f}")
    print(f"All top scores low/neg    : {diagnostics['all_low_or_negative']}")
    print(f"Keyword coverage          : {diagnostics['keyword_coverage']:.2f}")
    print(f"Matched keywords          : {diagnostics['matched_keywords']}")
    print(f"Decision                  : {'IN-SCOPE' if diagnostics['is_in_scope'] else 'OUT-OF-SCOPE'}")
    print(f"Reason                    : {diagnostics['reason']}")

    print("\nFinal pipeline output:")
    print(response)


if __name__ == "__main__":
    main()
