import os, sys
from datasets import load_dataset
f = list(load_dataset("locuslab/TOFU", "full")["train"])
fg = list(load_dataset("locuslab/TOFU", "forget10_perturbed")["train"])
print("full rows:", len(f), "| forget10 rows:", len(fg))
print("forget keys:", list(fg[0].keys()))
# author grouping heuristic: look at first token / entity repetition across consecutive QA
print("\n-- first 6 forget QA (question head) --")
for r in fg[:6]:
    print("  Q:", r["question"][:70])
print("\n-- rows 18-24 (author boundary?) --")
for i in range(18, 25):
    print(f"  [{i}]", fg[i]["question"][:70])
# count distinct author-name-ish tokens if a 'question' mentions names
