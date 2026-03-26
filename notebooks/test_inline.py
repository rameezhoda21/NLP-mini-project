import re

text = """
had. 8.22. Examination of Accused – (1) Object set...
or plan. 8.18. Variations in descrip tion of individual – It is
see section 8.22 of the Act.
"""

# Let's adjust top_level_pattern_str to not require ^ but look for sentence boundaries or word boundaries.
# A boundary can be ^ OR (?<=[.!?])\s+ OR (?<=\n)
# Actually, let's just find \b (?:Section|...|Article) \d+ OR \d+\.\d+\.\s+[A-Z]
keywords = r'(?:Section|SECTION|Article|ARTICLE|Chapter|CHAPTER|Part|PART|Rule|RULE|Clause|CLAUSE)'

# We only want to insert double newlines BEFORE a rule if it is immediately preceded by a dot or space,
# to fix OCR joining lines.
# If a rule heading starts with a Keyword, it usually looks like "Section 12." or "Section 12-"
# If it starts with digits, it usually is "1.2. Title" or "1. Title"

# Let's see what happens if we use \b
inline_pattern = (
    r'(?<=[.!?])\s+' + # must follow a sentence-ending punctuation and some space
    r'(?:' +
    keywords + r'\s+[-A-Za-z0-9:.]+' +
    r'|[A-Z]+\s+-\s*[IVXLCDM]+' +
    r'|\d+(?:\.\d+)*[A-Za-z]?(?:\.|\s+(?=\())' + 
    r')'
)

# wait, we need the lookahead to keep the matched rule text!
inline_pattern_la = (
    r'(?<=[.!?])\s+' + 
    r'(?=' +
    keywords + r'\s+[-A-Za-z0-9:.]+' +
    r'|[A-Z]+\s+-\s*[IVXLCDM]+' +
    r'|\d+(?:\.\d+)*[A-Za-z]?(?:\.|\s+(?=\())' + 
    r')'
)

print(re.sub(inline_pattern_la, r'\n\n', text))
