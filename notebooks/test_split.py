import re

text = """or request for discharge of an accused or suspect.
11. (1) the Prosecutor General shall -
(a) submit an annual report of the Service to Government within
three months of the conclusion of the calendar year to which the
report pertains;
"""

top_level_pattern_str = r'(?:Section|Article|Chapter|Part|Rule|Clause)\s+[A-Za-z0-9]+|\d+(?:\.\d+)*[A-Za-z]?\.'
# We only want to match at the beginning of a line.
# \n followed by top_level
text2 = re.sub(r'\n(?=' + top_level_pattern_str + r')', r'\n\n', text, flags=re.IGNORECASE)

print("--- TEXT 2 ---")
print(repr(text2))

