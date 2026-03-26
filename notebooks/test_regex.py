import re

pat = re.compile(r'^(Section|Article|Chapter|Part|Rule|Clause)\s+[A-Za-z0-9]+|^\d+(?:\.\d+)*[A-Za-z]?\.', re.IGNORECASE)
print(bool(pat.match("section 9;")))
print(bool(pat.match("Section 9.")))
print(bool(pat.match("11.")))

pat2 = re.compile(r'^(Section|Article|Chapter|Part|Rule|Clause)\s+[A-Za-z0-9]+\.|^\d+(?:\.\d+)*[A-Za-z]?\.(?=\s|$)')
print(bool(pat2.match("section 9;")))
print(bool(pat2.match("Section 9.")))
print(bool(pat2.match("11. ")))

