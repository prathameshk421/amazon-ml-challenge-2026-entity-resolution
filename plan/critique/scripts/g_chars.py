"""Critique job G: every distinct character in the Indic range (U+0900-U+0D7F) and outside ASCII in the
train and test pool NAMES, with the reference romanizer's class (translit_ref.classify) and counts.
Flags characters that fall back to 'space' (word boundary), 'ignore', or an unknown-consonant fallback.
Output: plan/critique/out/g_chars.json
"""
import json
import os
import sys
import unicodedata
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "plan", "reference"))
from translit_ref import CONS, VOW, classify  # noqa: E402

D = os.path.join(ROOT, "student_resource", "dataset")
cnt = {"train": Counter(), "test": Counter()}
for split, files in (("train", ["train/train_source2.tsv", "train/train_source3.tsv"]),
                     ("test", ["test/test_source2.tsv", "test/test_source3.tsv"])):
    for fn in files:
        with open(os.path.join(D, fn), encoding="utf-8", newline="") as f:
            next(f)
            for line in f:
                name = line.split("\t", 2)[1]
                if name.isascii():
                    continue
                cnt[split].update(c for c in name if ord(c) > 127)
R = {}
for split in cnt:
    rows = []
    for c, v in cnt[split].most_common():
        o = ord(c)
        nm = unicodedata.name(c, "?")
        if 0x0900 <= o <= 0x0D7F:
            k, val = classify(c)
            rest = nm.split(" ", 1)[1] if " " in nm else ""
            fallback = k == "cons" and rest.startswith("LETTER ") and rest[7:] not in CONS and rest[7:] not in VOW
            if k in ("space", "ignore") or fallback:
                rows.append([c, nm, v, k, val])
        elif unicodedata.category(c).startswith("L") and not "".join(x for x in unicodedata.normalize("NFKD", c) if unicodedata.category(x) != "Mn").isascii():
            rows.append([c, nm, v, "non-indic letter, not NFKD-ascii", ""])
    R[split] = {"n_distinct_nonascii": len(cnt[split]), "flagged": rows[:60]}
json.dump(R, open(os.path.join(ROOT, "plan", "critique", "out", "g_chars.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False)[:4000])
