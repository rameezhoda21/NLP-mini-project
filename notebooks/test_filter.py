import re
from pathlib import Path

text = Path("data/cleaned_text/01_sindh_criminal_court_rules_2012.txt").read_text(encoding='utf-8')

# Same split as before
top_level_pattern_str = r'^(?:Section|SECTION|Article|ARTICLE|Chapter|CHAPTER|Part|PART|Rule|RULE|Clause|CLAUSE)\s+[-A-Za-z0-9:.]+' \
                        r'|^[A-Z]+\s+-\s*[IVXLCDM]+' \
                        r'|^\d+(?:\.\d+)*[A-Za-z]?(?:\.|\s+(?=\())'
text = re.sub(r'\n(?=' + top_level_pattern_str + r')', r'\n\n', text, flags=re.MULTILINE)
paragraphs = re.split(r'\n{2,}', text)

def is_low_value(para):
    para_lower = para.lower()
    
    # 1. Obvious ToC/Index headers
    if re.search(r'^\s*(contents|table of contents|index|appendices)\s*$', para_lower, re.MULTILINE):
        return True
        
    # 2. Page number tables (lines ending in numbers, often with chapters/rules)
    lines = para.strip().split('\n')
    if not lines: return True
    
    toc_line_count = 0
    form_list_count = 0
    register_count = 0
    
    for line in lines:
        line_l = line.lower()
        # line ends with a number (like page number)
        if re.search(r'\d+\s*$', line_l) and len(line.split()) > 3:
            toc_line_count += 1
        if 'form of' in line_l or 'form no.' in line_l:
            form_list_count += 1
        if 'register of' in line_l or 'register no.' in line_l:
            register_count += 1
            
    # If more than 40% of lines look like ToC entries (ending in numbers, etc)
    if len(lines) > 2 and (toc_line_count / len(lines)) > 0.4:
        return True
        
    # If it's heavily dense with terms like "form of", "register of"
    if len(lines) > 2 and ((form_list_count + register_count) / len(lines)) > 0.4:
        return True
        
    # Ignore cover notification if it's very short and not substantive
    if "n o t i f i c a t i o n" in para_lower and len(lines) < 5:
         pass # Maybe keep it, it's just a title
         
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
print("--- FIRST 3 DROPPED ---")
for d in dropped[:3]:
    print(repr(d[:200]))
print("--- FIRST 3 KEPT ---")
for k in kept[:3]:
    print(repr(k[:200]))

