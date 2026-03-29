"""
Fixed-size chunking strategy for ablation study comparison.

Splits cleaned text files into fixed-size chunks based on word count,
with optional overlap. Unlike the legal-aware chunker (chunk_text.py),
this strategy has NO awareness of document structure — it simply splits
text at word boundaries every N words.

Output: data/rag_chunks_fixed.csv  (same schema as rag_chunks.csv)
"""

import os
import re
import csv
from pathlib import Path

# ==========================================
# Configuration
# ==========================================
BASE_DIR = Path("data")
CLEANED_DIR = BASE_DIR / "cleaned_text"
OUTPUT_CSV_PATH = BASE_DIR / "rag_chunks_fixed.csv"

CHUNK_SIZE_WORDS = 300      # fixed number of words per chunk
OVERLAP_WORDS = 50          # overlap between consecutive chunks


def get_metadata_from_filename(filename: str) -> tuple:
    """
    Infers the document ID and title from the filename.
    Same logic as chunk_text.py to keep IDs consistent.
    """
    match = re.match(r'^(\d+)[_-](.*)\.txt$', filename)
    if match:
        doc_num = match.group(1)
        name_part = match.group(2)
        doc_id = f"DOC{int(doc_num):02d}"
        title = name_part.replace('_', ' ').title()
    else:
        doc_id = "DOCXX"
        title = filename.replace('.txt', '').replace('_', ' ').title()
    return doc_id, title


def chunk_text_fixed(text: str) -> list[str]:
    """
    Splits text into fixed-size chunks of CHUNK_SIZE_WORDS words
    with OVERLAP_WORDS overlap between consecutive chunks.
    """
    # Normalize whitespace — collapse runs of whitespace into single spaces
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0
    step = CHUNK_SIZE_WORDS - OVERLAP_WORDS  # how far we advance each time

    while start < len(words):
        end = start + CHUNK_SIZE_WORDS
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        chunks.append(chunk_text)

        if end >= len(words):
            break
        start += step

    return chunks


def main():
    print(f"Starting FIXED-SIZE chunking (size={CHUNK_SIZE_WORDS}, overlap={OVERLAP_WORDS})...")

    if not CLEANED_DIR.exists():
        print(f"Error: Directory {CLEANED_DIR} does not exist.")
        return

    txt_files = sorted(CLEANED_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No text files found in {CLEANED_DIR}")
        return

    all_chunks_data = []

    for filepath in txt_files:
        filename = filepath.name
        print(f"Chunking: {filename}")

        doc_id, title = get_metadata_from_filename(filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Skip very short files (e.g. OCR failures)
        if len(text.strip()) < 50:
            print(f"  [Skipping] {filename} is too short.")
            continue

        chunks = chunk_text_fixed(text)

        for i, chunk_txt in enumerate(chunks, start=1):
            chunk_word_count = len(chunk_txt.split())

            # Skip tiny trailing fragments
            if chunk_word_count < 10:
                continue

            chunk_id = f"{doc_id}_FX{i:03d}"

            all_chunks_data.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "source_filename": filename,
                "title": title,
                "chunk_index": i,
                "chunk_text": chunk_txt,
                "word_count": chunk_word_count
            })

    print(f"\nWriting {len(all_chunks_data)} total chunks to {OUTPUT_CSV_PATH}...")

    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "chunk_id", "doc_id", "source_filename", "title",
            "chunk_index", "chunk_text", "word_count"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_chunks_data)

    print(f"Fixed-size chunking complete! {len(all_chunks_data)} chunks written.")


if __name__ == "__main__":
    main()
