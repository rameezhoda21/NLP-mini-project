import re

text = """3.12. Accession – The Officer in charge of the library shall ---
(a) Stamp the Court Seal on the title page of each book;
(b) fix a number label on the back;
(c) check the register...
(d) paste correction slips...
(e) The Officer taking charge... If any books are missing or damaged, he shall prepare a list...
(f) Every Judge on taking charge...

3.13. Nazarat – The Nazirs or where there are no Nazir the Accountant shall keep the account of the Sessions Courts.
"""

def chunk_text_legally(text):
    TARGET_MIN_WORDS = 250
    TARGET_MAX_WORDS = 450
    
    paragraphs = re.split(r'\n+', text)
    boundary_pattern = re.compile(
        r'^(Section|Article|Chapter|Part|Rule|Clause)\s+[A-Za-z0-9]+|^\d+(\.\d+)*[A-Za-z]?\.',
        re.IGNORECASE
    )
    
    rules = []
    current_rule = []
    
    for para in paragraphs:
        para = para.strip()
        if not para: continue
        
        if boundary_pattern.match(para):
            if current_rule:
                rules.append(current_rule)
            current_rule = [para]
        else:
            current_rule.append(para)
            
    if current_rule:
        rules.append(current_rule)
        
    chunks = []
    current_chunk_paras = []
    current_word_count = 0
    
    for rule_paras in rules:
        rule_words = sum(len(p.split()) for p in rule_paras)
        
        # If this single rule is huge, we must split it internally
        if rule_words > TARGET_MAX_WORDS:
            # flush current
            if current_chunk_paras:
                chunks.append("\n\n".join(current_chunk_paras))
                current_chunk_paras = []
                current_word_count = 0
                
            heading = rule_paras[0]
            heading_text = heading.split('\n')[0][:100]
            
            sub_chunk_paras = []
            sub_wc = 0
            
            for p in rule_paras:
                p_wc = len(p.split())
                if sub_wc + p_wc > TARGET_MAX_WORDS and sub_wc > 0:
                    chunks.append("\n\n".join(sub_chunk_paras))
                    # start new sub-chunk with heading
                    added_heading = f"{heading_text} (Cont.)"
                    if not p.startswith(added_heading):
                        sub_chunk_paras = [added_heading, p]
                        sub_wc = len(added_heading.split()) + p_wc
                    else:
                        sub_chunk_paras = [p]
                        sub_wc = p_wc
                else:
                    sub_chunk_paras.append(p)
                    sub_wc += p_wc
                    
            if sub_chunk_paras:
                # Instead of dumping it into chunks immediately, we can make it the current_chunk 
                # so the next rule might append to it if it's small, OR we just finalize it.
                # Usually, it's safer to just finalize it since the rule was huge.
                chunks.append("\n\n".join(sub_chunk_paras))
                
        else:
            # Rule is small enough. Decide whether to add to current chunk or start a new one.
            if current_word_count + rule_words > TARGET_MAX_WORDS and current_word_count > 0:
                chunks.append("\n\n".join(current_chunk_paras))
                current_chunk_paras = list(rule_paras)
                current_word_count = rule_words
            else:
                current_chunk_paras.extend(rule_paras)
                current_word_count += rule_words
                
    if current_chunk_paras:
        chunks.append("\n\n".join(current_chunk_paras))
        
    return chunks

for i, c in enumerate(chunk_text_legally(text)):
    print(f"--- CHUNK {i+1} ---")
    print(c)
