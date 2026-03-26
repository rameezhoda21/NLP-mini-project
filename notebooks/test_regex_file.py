import re
from pathlib import Path

text = Path("data/cleaned_text/02_sindh_criminal_prosecution_service_act_2009.txt").read_text(encoding='utf-8')

# keywords strictly capitalized
keywords = r'(?:Section|SECTION|Article|ARTICLE|Chapter|CHAPTER|Part|PART|Rule|RULE|Clause|CLAUSE)'

# matched patterns
pat_str = r'^' + keywords + r'\s+[-A-Za-z0-9:.]+' + \
          r'|^[A-Z]+\s+-\s*[IVXLCDM]+' + \
          r'|^\d+(?:\.\d+)*[A-Za-z]?(?:\.|\s+(?=\())'

pat = re.compile(pat_str, re.MULTILINE)

for m in pat.finditer(text):
    print(repr(m.group(0)))

