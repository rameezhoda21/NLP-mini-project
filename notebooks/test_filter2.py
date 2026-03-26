import re
from pathlib import Path

text = Path("data/cleaned_text/01_sindh_criminal_court_rules_2012.txt").read_text(encoding='utf-8')

# Let's adjust the splitting logic slightly to grab bigger tables if they are separated by double newlines.
paragraphs = re.split(r'\n{2,}', text)

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
    # We often see "No. Subjects Chapter & Rules Page Nos."
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

kept = []
dropped = []
for p in paragraphs:
    if not p.strip(): continue
    if is_low_value(p):
        dropped.append(p)
    else:
        kept.append(p)

print(f"Dropped {len(dropped)} paras. Kept {len(kept)} paras.")
for d in dropped[:6]:
    print(repr(d[:80]))
