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
import os
import re
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from sentence_transformers import CrossEncoder


# Cross-encoder model requested by the user.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5
ANSWER_TOP_K = 3

# Hugging Face Inference API settings.
HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Supported env var names for Hugging Face token.
HF_TOKEN_ENV_CANDIDATES = [
    "HF_API_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HF_TOKEN",
]

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
    """Backward-compatible wrapper. Kept for minimal code breakage."""
    result = generate_grounded_answer_with_hf_api(query, top_chunks)
    return result["final_answer"]


def build_grounded_prompt(query: str, context_chunks: list[dict[str, Any]]) -> str:
    """
    Build a strict prompt that forces grounded legal QA behavior.

    The model is explicitly told to:
    - answer only from provided context
    - never invent legal facts
    - clearly say when evidence is insufficient
    """
    context_blocks: list[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        title = chunk.get("title", "Untitled")
        text = chunk.get("chunk_text", "")
        context_blocks.append(
            f"[Context {i}]\n"
            f"Chunk ID: {chunk_id}\n"
            f"Title: {title}\n"
            f"Text:\n{text}"
        )

    context_section = "\n\n".join(context_blocks)

    return (
        "You are a careful legal assistant.\n"
        "Use ONLY the context below to answer the user question.\n"
        "Do NOT invent legal rules, sections, dates, or case facts.\n"
        "If the answer is not clearly supported by the context, say: "
        "'The provided context does not contain enough reliable information to answer this question.'\n"
        "Keep your answer concise and factual.\n\n"
        f"User Question:\n{query}\n\n"
        f"Context:\n{context_section}\n\n"
        "Final Answer:"
    )


def call_huggingface_inference_api(prompt: str) -> str:
    """
    Send a prompt to Hugging Face Inference API and return model text.

    Requires one of these environment variables:
    - HF_API_TOKEN
    - HUGGINGFACEHUB_API_TOKEN
    - HF_TOKEN
    """
    hf_api_token = ""
    for env_name in HF_TOKEN_ENV_CANDIDATES:
        value = os.environ.get(env_name, "").strip()
        if value:
            hf_api_token = value
            break

    if not hf_api_token:
        raise RuntimeError(
            "Missing Hugging Face API token. Set one of: "
            "HF_API_TOKEN, HUGGINGFACEHUB_API_TOKEN, HF_TOKEN"
        )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 350,
            "temperature": 0.2,
            "do_sample": False,
            "return_full_text": False,
        }
    }

    req = urllib_request.Request(
        HF_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {hf_api_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=120) as resp:
            response_text = resp.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Hugging Face API HTTP error {exc.code}: {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error calling Hugging Face API: {exc}") from exc

    parsed = json.loads(response_text)

    # Common response shape: [{"generated_text": "..."}]
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        generated = parsed[0].get("generated_text", "").strip()
        if generated:
            return generated

    # Some endpoints return dict with error details.
    if isinstance(parsed, dict) and "error" in parsed:
        raise RuntimeError(f"Hugging Face API error: {parsed['error']}")

    raise RuntimeError(f"Unexpected Hugging Face API response: {parsed}")


def generate_grounded_answer_with_hf_api(query: str, top_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build grounded prompt from top chunks and get a final answer from HF API.

    Returns a dictionary with:
    - final_answer
    - chunk_ids_used
    """
    chunk_ids_used = [str(chunk.get("chunk_id", "")) for chunk in top_chunks]
    prompt = build_grounded_prompt(query, top_chunks)
    final_answer = call_huggingface_inference_api(prompt)

    return {
        "final_answer": final_answer,
        "chunk_ids_used": chunk_ids_used,
    }


def run_confidence_gate_and_respond(query: str, top_chunks: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Apply confidence gate:
    - out-of-scope/low confidence -> fallback response
    - in-scope -> continue to answer generation
    """
    top_chunks_for_answer = top_chunks[:ANSWER_TOP_K]
    diagnostics = assess_retrieval_confidence(query, top_chunks_for_answer)

    if not diagnostics["is_in_scope"]:
        response = {
            "final_answer": FALLBACK_RESPONSE,
            "chunk_ids_used": [str(chunk.get("chunk_id", "")) for chunk in top_chunks_for_answer],
            "confidence_status": "LOW_CONFIDENCE",
        }
    else:
        try:
            answer_payload = generate_grounded_answer_with_hf_api(query, top_chunks_for_answer)
            response = {
                "final_answer": answer_payload["final_answer"],
                "chunk_ids_used": answer_payload["chunk_ids_used"],
                "confidence_status": "HIGH_CONFIDENCE",
            }
        except Exception as exc:
            # Safe fallback in case API call fails.
            response = {
                "final_answer": (
                    "I could not generate a reliable answer right now because the answer model "
                    f"request failed: {exc}"
                ),
                "chunk_ids_used": [str(chunk.get("chunk_id", "")) for chunk in top_chunks_for_answer],
                "confidence_status": "HIGH_CONFIDENCE_MODEL_ERROR",
            }

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
    print("=" * 85)
    print(f"Confidence status : {response['confidence_status']}")
    print(f"Chunk IDs used    : {response['chunk_ids_used']}")
    print("Final answer:")
    print(response["final_answer"])


if __name__ == "__main__":
    main()
