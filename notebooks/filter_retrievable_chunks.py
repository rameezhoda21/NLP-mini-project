"""
Filter/tag low-value legal chunks before indexing or retrieval.

This script keeps the original chunk data and adds:
- is_retrievable (True/False)
- filter_reason  (why a chunk was marked non-retrievable)

Goal:
- Keep substantive legal text (rules, sections, powers, duties, procedures)
- Mark appendix/statistical/reporting chunks as non-retrievable
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# Default input/output locations (change if needed)
DEFAULT_INPUT_CSV = Path("data/rag_chunks.csv")
DEFAULT_OUTPUT_CSV = Path("data/rag_chunks_with_retrievable_flag.csv")
DEFAULT_RETRIEVABLE_ONLY_CSV = Path("data/rag_chunks_retrievable_only.csv")


# Phrases strongly associated with appendix/statistical reporting content.
STRONG_LOW_VALUE_PATTERNS = [
    r"\bpending at the end of the previous year\b",
    r"\bdisposed of\b",
    r"\bended in acquittal\b",
    r"\bended in conviction\b",
    r"\bannual statement\b",
    r"\bquarterly statement\b",
]


# Broad admin/reporting words. On their own, these are not always enough to exclude a chunk.
ADMIN_KEYWORDS = [
    r"\bstatement\b",
    r"\bappendix\b",
    r"\bform\b",
    r"\bregister\b",
    r"\bstatistical\b",
    r"\breport\b",
]


# Terms that usually indicate substantive legal content.
SUBSTANTIVE_KEYWORDS = [
    r"\bsection\b",
    r"\bchapter\b",
    r"\brule\b",
    r"\bclause\b",
    r"\barticle\b",
    r"\bprocedure\b",
    r"\bpower\b",
    r"\bduty\b",
    r"\bjurisdiction\b",
    r"\bmagistrate\b",
    r"\bcourt\b",
    r"\bpublic prosecutor\b",
    r"\bshall\b",
    r"\bmay\b",
]


def count_pattern_hits(text: str, patterns: list[str]) -> int:
    """Count how many patterns appear at least once in text."""
    hits = 0
    for pattern in patterns:
        if re.search(pattern, text):
            hits += 1
    return hits


def is_mainly_tabular_or_listing(text: str) -> bool:
    """
    Heuristic for appendix/statistical blocks.

    We consider text likely low-value if many lines look like list/table rows,
    especially with words like form/register/statement and short line lengths.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False

    numbered_line_count = 0
    short_line_count = 0
    admin_line_count = 0

    for line in lines:
        if re.match(r"^\d+[\).\-\s]", line):
            numbered_line_count += 1
        if len(line.split()) <= 8:
            short_line_count += 1
        if re.search(r"\b(form|register|statement|appendix|disposed|acquittal|conviction)\b", line):
            admin_line_count += 1

    numbered_ratio = numbered_line_count / len(lines)
    short_ratio = short_line_count / len(lines)
    admin_ratio = admin_line_count / len(lines)

    return (numbered_ratio > 0.45 and short_ratio > 0.45) or admin_ratio > 0.45


def classify_chunk(chunk_text: str) -> tuple[bool, str]:
    """
    Return (is_retrievable, reason).

    Decision logic:
    1) Strong reporting phrases -> mark as non-retrievable
    2) Tabular/listing-heavy appendix style -> mark as non-retrievable
    3) If substantive legal signals are strong, keep retrievable
    4) If admin/reporting signals dominate, mark non-retrievable
    5) Otherwise keep retrievable (safer default to avoid losing legal content)
    """
    text = str(chunk_text or "").lower()

    if not text.strip():
        return False, "empty_chunk_text"

    strong_hits = count_pattern_hits(text, STRONG_LOW_VALUE_PATTERNS)
    admin_hits = count_pattern_hits(text, ADMIN_KEYWORDS)
    substantive_hits = count_pattern_hits(text, SUBSTANTIVE_KEYWORDS)

    if strong_hits >= 2:
        return False, "strong_reporting_phrases"

    if is_mainly_tabular_or_listing(text) and admin_hits >= 1 and substantive_hits <= 1:
        return False, "appendix_or_tabular_pattern"

    # Protect substantive legal content when legal signals are clear.
    if substantive_hits >= 3:
        return True, "substantive_legal_content"

    # If admin/reporting language dominates and legal signals are weak, exclude.
    if admin_hits >= 2 and substantive_hits <= 1:
        return False, "admin_terms_dominate"

    return True, "default_keep"


def add_retrievable_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_retrievable and filter_reason columns without removing original columns."""
    if "chunk_text" not in df.columns:
        raise ValueError("Input CSV must include a 'chunk_text' column.")

    decisions = df["chunk_text"].apply(classify_chunk)
    df_out = df.copy()
    df_out["is_retrievable"] = decisions.apply(lambda x: x[0])
    df_out["filter_reason"] = decisions.apply(lambda x: x[1])

    return df_out


def main() -> None:
    print("Chunk Filtering: Add is_retrievable Flag")

    input_raw = input(f"Input CSV path (Enter for default: {DEFAULT_INPUT_CSV}): ").strip()
    output_raw = input(f"Output CSV path (Enter for default: {DEFAULT_OUTPUT_CSV}): ").strip()

    input_csv = Path(input_raw) if input_raw else DEFAULT_INPUT_CSV
    output_csv = Path(output_raw) if output_raw else DEFAULT_OUTPUT_CSV

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print("\nLoading CSV...")
    df = pd.read_csv(input_csv)

    print("Applying chunk filter rules...")
    df_flagged = add_retrievable_flag(df)

    # Save full dataset + flags (preserves original rows and columns).
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_flagged.to_csv(output_csv, index=False)

    # Also save a convenience file for retrievable chunks only.
    retrievable_only = df_flagged[df_flagged["is_retrievable"]].copy()
    retrievable_only.to_csv(DEFAULT_RETRIEVABLE_ONLY_CSV, index=False)

    total = len(df_flagged)
    keep = int(df_flagged["is_retrievable"].sum())
    drop = total - keep

    print("\n" + "=" * 70)
    print("Filtering complete")
    print("=" * 70)
    print(f"Total chunks              : {total}")
    print(f"Retrievable (keep)        : {keep}")
    print(f"Non-retrievable (marked)  : {drop}")
    print(f"Full output               : {output_csv}")
    print(f"Retrievable-only output   : {DEFAULT_RETRIEVABLE_ONLY_CSV}")


if __name__ == "__main__":
    main()
