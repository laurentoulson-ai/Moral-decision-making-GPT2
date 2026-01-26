"""
This script supports with analysis of moral statements based on neuron activation differences, to answer RQ3.
It identifies the top 25 statements with the highest positive activation differences and the bottom 25 statements closest to zero difference.
It also performs a keyword analysis to identify common words in the top and bottom statements.
Outputs:
- CSV file with top and bottom statements.
- Text file summarizing keyword counts in top vs bottom statements.
"""

import pandas as pd
import numpy as np
from collections import Counter
import re

# =============================
# CONFIG - change these per dataset
INPUT_FILE = "n625_util.csv"   # input CSV file
OUTPUT_STATEMENTS_FILE = "n625_top_bottom_statements.csv"  # output CSV with statements
OUTPUT_KEYWORDS_FILE = "n625_keyword_counts.txt"           # output keyword summary
TOP_COUNT = 25    # top 25 statements with highest positive activation differences
BOTTOM_COUNT = 25 # bottom 25 statements closest to zero
# =============================

# --- Load data ---
df = pd.read_csv(INPUT_FILE)

# ensure required columns exist
if not {"moral_text", "difference"}.issubset(df.columns):
    raise ValueError("Input file must contain 'moral_text' and 'difference' columns")

# --- Select top 25 (highest positive activation differences) ---
top_df = df[df["difference"] > 0].nlargest(TOP_COUNT, "difference")

# --- Select bottom 25 (closest to zero absolute difference) ---
bottom_df = df.iloc[(df["difference"].abs()).argsort()[:BOTTOM_COUNT]]

# --- Save statements (top + bottom) ---
export_df = pd.concat([top_df, bottom_df])
export_df.to_csv(OUTPUT_STATEMENTS_FILE, index=False)

# --- Keyword analysis ---
# tokenize text (very simple: lowercase words, strip non-alphabetic)
def tokenize(text):
    return re.findall(r"[a-zA-Z]+", str(text).lower())

# get all tokens across dataset
all_tokens = []
for t in df["moral_text"]:
    all_tokens.extend(tokenize(t))

# List of words to exclude
EXCLUDE_WORDS = {"he", "she", "the", "that", "by", "to", "his", "her", "of", "in", "it", "for", "would", "they", "and"}

# most common words across dataset, excluding the specified words
common_words = [w for w, _ in Counter(all_tokens).most_common(50 + len(EXCLUDE_WORDS)) if w not in EXCLUDE_WORDS][:50]

# count word frequencies in top and bottom sets
def count_keywords(rows, keywords):
    counts = Counter()
    for t in rows["moral_text"]:
        tokens = tokenize(t)
        for w in keywords:
            counts[w] += tokens.count(w)
    return counts

counts_top = count_keywords(top_df, common_words)
counts_bottom = count_keywords(bottom_df, common_words)

# --- Save keyword counts summary ---
with open(OUTPUT_KEYWORDS_FILE, "w") as f:
    f.write("Keyword counts for top vs bottom statements\n")
    f.write("=====================================\n\n")
    f.write(f"Top {TOP_COUNT} statements (highest positive difference) vs Bottom {BOTTOM_COUNT} (closest to zero)\n\n")
    f.write("Word,Top_Count,Bottom_Count\n")
    for w in common_words:
        f.write(f"{w},{counts_top[w]},{counts_bottom[w]}\n")

print("Analysis complete.")
print(f"Saved top/bottom statements -> {OUTPUT_STATEMENTS_FILE}")
print(f"Saved keyword counts -> {OUTPUT_KEYWORDS_FILE}")
