import re

TARGET_MIN_WORDS = 250
TARGET_MAX_WORDS = 450

def chunk_text_legally(text: str) -> list:
    keywords = r'(?:Section|SECTION|Article|ARTICLE|Chapter|CHAPTER|Part|PART|Rule|RULE|Clause|CLAUSE)'
    top_level_pattern_str = (
        r'^' + keywords + r'\s+[-A-Za-z0-9:.]+' +
        r'|^[A-Z]+\s+-\s*[IVXLCDM]+' +
        r'|^\d+(?:\.\d+)*[A-Za-z]?(?:\.|\s+(?=\())'
    )
    
    # 1. First, split inline OCR merges 
    inline_pattern_la = (
        r'(?<=[.!?])\s+' + 
        r'(?=' +
        keywords + r'\s+[-A-Za-z0-9:.]+' +
        r'|[A-Z]+\s+-\s*[IVXLCDM]+' +
        r'|\d+(?:\.\d+)*[A-Za-z]?(?:\.|\s+(?=\())' + 
        r')'
    )
    text = re.sub(inline_pattern_la, r'\n\n', text)

    # 2. Then ensure newlines before all top level boundaries that are already at the beginning of a line
    text = re.sub(r'\n(?=' + top_level_pattern_str + r')', r'\n\n', text, flags=re.MULTILINE)

    # 3. Split into paragraphs
    raw_paragraphs = re.split(r'\n{2,}', text)
    paragraphs = []
    
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        paragraphs.append(para)

    # 4. Group paragraphs into rules
    rules = []
    current_rule = []
    
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

    # 5. Build chunks
    for rule_paras in rules:
        if not rule_paras:
            continue

        rule_words = sum(len(p.split()) for p in rule_paras)
        
        # Short rule: single chunk, never mixed with other rules
        if rule_words <= TARGET_MAX_WORDS:
            chunks.append("\n\n".join(rule_paras))
        else:
            # Long rule: split into sub-chunks, ensuring each sub-chunk starts with the heading
            heading = rule_paras[0]
            # Take the first line or first 200 characters as the "heading text"
            heading_text = heading.split('\n')[0][:200].strip()
            
            sub_chunks_lists = []
            current_sub_chunk = []
            current_sub_wc = 0
            
            for p in rule_paras:
                p_wc = len(p.split())
                
                if current_sub_wc > 0 and (current_sub_wc + p_wc) > TARGET_MAX_WORDS:
                    sub_chunks_lists.append(current_sub_chunk)
                    
                    if not p.startswith(heading_text):
                        current_sub_chunk = [heading_text, p]
                        current_sub_wc = len(heading_text.split()) + p_wc
                    else:
                        current_sub_chunk = [p]
                        current_sub_wc = p_wc
                else:
                    current_sub_chunk.append(p)
                    current_sub_wc += p_wc
                    
            if current_sub_chunk:
                # Merge if it's a small leftover fragment
                if sub_chunks_lists and current_sub_wc < 100:
                    # check if appending heading to previous subchunk was done, here we just extend
                    # but wait! if current_sub_chunk is just [heading, fragment], 
                    # we should only merge the fragment, not the heading repeated!
                    if len(current_sub_chunk) == 2 and current_sub_chunk[0] == heading_text:
                        sub_chunks_lists[-1].extend(current_sub_chunk[1:])
                    else:
                        sub_chunks_lists[-1].extend(current_sub_chunk)
                else:
                    sub_chunks_lists.append(current_sub_chunk)
            
            for sc_list in sub_chunks_lists:
                chunks.append("\n\n".join(sc_list))

    return chunks

test_text = """
8.17. Evidence with reference to maps and places – The evidence of a witness, with reference to a map or plan shall be recorded in such a way that the places mentioned by the witness are easily identifiable on the map or plan. 8.18. Variations in description of individual – It is frequently happens that the same individual is known by more names than one. Sometimes only the surname, sometimes only the name of the caste, or occupation of the individual is mentioned or he is spoken of by a nickname, such as Baba Ladla, Langra. Such variations in description require explanation to render them intelligible to an appellate Court. A court of first instance shall therefore, take care not only to ascertain, but to make clear by evidence duly recorded, the identity of any individual who is so referred to under varying appellations and if such an individual is an accused person, his name and serial number according to the charge sheet should be cited. 8.19. Original public record – The original public records shall not ordinarily be admitted in evidence, where certified copies are obtainable and will answer the required purpose. When the originals are required, the requisition shall state clearly the time and place but should not ask for production. The requisition shall be signed and sealed in the same way as a summons. Due regard shall in all cases be had to privilege from disclosure of official communication and affairs of State. 8.20. Marking of exhibits – (i) Every document, weapon or other article admitted in evidence before a Court be clearly marked with the number it bears in the general index of the case and the number and other particulars of the case and of the Police Station.
(ii) The Court shall mark the documents admitted in evidence on behalf of the prosecution with the letter P followed by a serial number indicating the order in which they are admitted thus; Exhibit P-1, P- 2, P-3, etc. And the documents admitted on behalf of the defence with the letter D followed by a numeral, thus Exhibit D-1, D-2, D-3 etc.
(iii) In the same manner every material exhibits admitted in evidence shall be marked with numerals in serial order thus, Exb-1, Exb-2, Exb-3 etc.
(iv) All exhibit marks on documents and material exhibits shall be initialled by the Presiding Officer.
(v) No document or material exhibit which has been admitted in evidence and exhibited shall be returned or destroyed until the period for appeal has expires or until the appeal has been disposed of, if an appeal be preferred against the conviction and sentence.
(vi) Documents or material exhibits which have not been admitted in evidence should not be made part of the record, but should be returned to the party by whom they were produced. 8.21. Proof of Statements, under section 161 of the Code – (1) When a statement recorded under Section 161 of the Code is used in the manner indicated in Section 162 of the Code, the passage which has been specifically put to the witness in order to contradict him shall first be marked for identification and exhibited after it is proved.
"""

chunks = chunk_text_legally(test_text)
for i, c in enumerate(chunks):
    print(f"--- CHUNK {i} ---")
    print(c)
