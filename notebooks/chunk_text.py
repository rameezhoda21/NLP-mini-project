import os
import re
import csv
from pathlib import Path

# ==========================================
# 1. Setup Paths and Configuration
# ==========================================
BASE_DIR = Path("data")
CLEANED_DIR = BASE_DIR / "cleaned_text"
OUTPUT_CSV_PATH = BASE_DIR / "rag_chunks.csv"

# Chunking targets
TARGET_MIN_WORDS = 250
TARGET_MAX_WORDS = 450
OVERLAP_WORDS_TARGET = 40

def get_metadata_from_filename(filename: str) -> tuple:
    """
    Infers the document ID and title from the filename.
    Example: '07_consolidated_sindh_act.txt' -> ('DOC07', 'Consolidated Sindh Act')
    """
    # Look for leading numbers in the filename
    match = re.match(r'^(\d+)[_-](.*)\.txt$', filename)
    if match:
        doc_num = match.group(1)
        name_part = match.group(2)
        # Format doc_id as DOC01, DOC02, etc.
        doc_id = f"DOC{int(doc_num):02d}"
        
        # Clean up the title (replace underscores with spaces, title case)
        title = name_part.replace('_', ' ').title()
    else:
        # Fallback if the pattern doesn't match
        doc_id = "DOCXX"
        title = filename.replace('.txt', '').replace('_', ' ').title()
        
    return doc_id, title

def is_low_value(para):
    para_lower = para.lower()
    
    # Check for direct matches of headers we don't care about
    if re.search(r'^\s*(contents|table of contents|index|appendices)\s*$', para_lower, re.MULTILINE):
        return True
        
    lines = [L.strip() for L in para.split('\n') if L.strip()]
    if not lines: return True
    
    toc_line_count = 0
    form_list_count = 0
    
    # Check for "Form" and "Page Nos", "Register" columns
    if re.search(r'page nos\.?', para_lower):
        return True
        
    for line in lines:
        line_l = line.lower()
        # line ends with a number (like page number) but has words before
        if re.search(r'\s+\d+$', line_l):
            toc_line_count += 1
        if 'form no.' in line_l or 'form of' in line_l or re.search(r'^\d+\s+form', line_l) or 'register of' in line_l:
            form_list_count += 1
            
    # If a good chunk of lines are ToC-like (trailing numbers) or forms/registers
    if len(lines) > 2:
        if (toc_line_count / len(lines)) > 0.3:
            return True
        if (form_list_count / len(lines)) > 0.3:
            return True
            
    # Forms and registers tabular data usually starts with a sequence number
    tabular_lines = 0
    for line in lines:
        if re.match(r'^\d+\s+', line):
            tabular_lines += 1
    if len(lines) >= 3 and (tabular_lines / len(lines)) > 0.5:
        # Check if it lacks normal sentence structure (e.g. lots of short items)
        avg_words = sum(len(L.split()) for L in lines) / len(lines)
        if avg_words < 15:
            return True

    return False

def chunk_text_legally(text: str) -> list:
    """
    Chunks text with a preference for keeping legal sections/paragraphs together.
    Attempts to stay within the TARGET_MIN_WORDS and TARGET_MAX_WORDS bounds.
    """
    keywords = r'(?:Section|SECTION|Article|ARTICLE|Chapter|CHAPTER|Part|PART|Rule|RULE|Clause|CLAUSE)'
    top_level_pattern_str = (
        r'^' + keywords + r'\s+[-A-Za-z0-9:.]+' +
        r'|^[A-Z]+\s+-\s*[IVXLCDM]+' +
        r'|^\d+(?:\.\d+)*[A-Za-z]?(?:\.|\s+(?=\())'
    )
    
    # 1. First, fix OCR errors where rules are pasted mid-line right after a sentence
    inline_pattern_la = (
        r'(?<=[.!?])\s+' + 
        r'(?=' +
        keywords + r'\s+[-A-Za-z0-9:.]+' +
        r'|[A-Z]+\s+-\s*[IVXLCDM]+' +
        r'|\d+(?:\.\d+)*[A-Za-z]?(?:\.|\s+(?=\())' + 
        r')'
    )
    text = re.sub(inline_pattern_la, r'\n\n', text)

    # 2. Force single-newlines before top-level section markers into double-newlines
    text = re.sub(r'\n(?=' + top_level_pattern_str + r')', r'\n\n', text, flags=re.MULTILINE)
    
    # Split text into paragraphs
    raw_paragraphs = re.split(r'\n{2,}', text)
    paragraphs = []
    
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        if is_low_value(para):
            # Exclude this chunk as low-value/index/form list
            continue
        paragraphs.append(para)

    # 3. Group paragraphs into logical rules/sections
    rules = []
    current_rule = []
    
    # Only treat numerical/word-prefixed as HIGH-LEVEL boundaries
    top_level_pattern = re.compile(top_level_pattern_str)

    for para in paragraphs:
        if top_level_pattern.match(para):
            if current_rule:
                rules.append(current_rule)
            current_rule = [para]
        else:
            current_rule.append(para)
            
    if current_rule:
        rules.append(current_rule)
        
    chunks = []

    # 4. Build chunks strictly respecting rule integrity
    for rule_paras in rules:
        if not rule_paras:
            continue

        rule_words = sum(len(p.split()) for p in rule_paras)
        
        # If this single rule is small enough, it forms exactly one chunk.
        # No mixing with unrelated rules!
        if rule_words <= TARGET_MAX_WORDS:
            chunks.append("\n\n".join(rule_paras))
        else:
            # Rule is too long, we must split it internally by subclauses.
            heading = rule_paras[0]
            # Take the first line or up to 200 chars as the rule number and heading
            heading_text = heading.split('\n')[0][:200].strip()
            
            sub_chunks_lists = []
            current_sub_chunk_paras = []
            sub_wc = 0
            
            for p in rule_paras:
                p_wc = len(p.split())
                
                if sub_wc > 0 and (sub_wc + p_wc) > TARGET_MAX_WORDS:
                    sub_chunks_lists.append(current_sub_chunk_paras)
                    
                    # Start new sub-chunk, ALWAYS prepending the rule number and heading
                    if not p.startswith(heading_text):
                        current_sub_chunk_paras = [heading_text, p]
                        sub_wc = len(heading_text.split()) + p_wc
                    else:
                        current_sub_chunk_paras = [p]
                        sub_wc = p_wc
                else:
                    current_sub_chunk_paras.append(p)
                    sub_wc += p_wc
                    
            if current_sub_chunk_paras:
                # Merge small leftover fragments with the previous sub-chunk
                if sub_chunks_lists and sub_wc < 150:
                    # If the only thing in current sub-chunk besides the heading is the fragment:
                    if len(current_sub_chunk_paras) >= 2 and current_sub_chunk_paras[0] == heading_text:
                        sub_chunks_lists[-1].extend(current_sub_chunk_paras[1:])
                    else:
                        sub_chunks_lists[-1].extend(current_sub_chunk_paras)
                else:
                    sub_chunks_lists.append(current_sub_chunk_paras)

            for sc_list in sub_chunks_lists:
                chunks.append("\n\n".join(sc_list))

    return chunks

def main():
    print("Starting chunking process for Legal RAG...")
    
    if not CLEANED_DIR.exists():
        print(f"Error: Directory {CLEANED_DIR} does not exist.")
        return

    txt_files = list(CLEANED_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No text files found in {CLEANED_DIR}")
        return

    all_chunks_data = []

    # 2. Read each cleaned text file
    for filepath in txt_files:
        filename = filepath.name
        print(f"Chunking: {filename}")
        
        doc_id, title = get_metadata_from_filename(filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # If the file is basically empty (e.g. OCR failure), skip it
        if len(text.strip()) < 50:
            print(f"  [Skipping] {filename} is too short.")
            continue
            
        # 3. Create Chunks
        chunks = chunk_text_legally(text)
        
        # 4. Format and index the chunks
        for i, chunk_txt in enumerate(chunks, start=1):
            
            chunk_word_count = len(chunk_txt.split())
            
            # Skip arbitrarily tiny stranded chunks
            if chunk_word_count < 10: 
                continue
                
            chunk_id = f"{doc_id}_CH{i:03d}"
            
            all_chunks_data.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "source_filename": filename,
                "title": title,
                "chunk_index": i,
                "chunk_text": chunk_txt,
                "word_count": chunk_word_count
            })

    # 5. Save all chunks securely to CSV
    print(f"\nWriting {len(all_chunks_data)} total chunks to {OUTPUT_CSV_PATH}...")
    
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "chunk_id", "doc_id", "source_filename", "title", 
            "chunk_index", "chunk_text", "word_count"
        ]
        
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_chunks_data)
        
    print("Chunking complete!")

if __name__ == "__main__":
    main()