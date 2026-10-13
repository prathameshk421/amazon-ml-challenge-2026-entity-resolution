"""Count focused candidate normalization tokens per country (fit-S1 + full pool).
Writes artifacts/metrics/normdict_evidence.json. Run: .venv/bin/python legacy/src/normdict_evidence.py
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
MET = os.path.join(ROOT, "artifacts/metrics")
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

CANDS = """street road avenue drive lane court boulevard place circle suite apartment
rd st ave avenu dr ln ct blvd ste apt hwy hway highway pkwy pike
corporation incorporated limited private company services associates brothers
corp inc ltd pvt llc llp pllc lp pc pa co l t d p v t svc assoc bros
and sri shri shree shrey messrs sons enterprises near opp opposite behind off
nagar no plot floor sector colony block phase cross main layout extentions extension""".split()


def main():
    fit = set(json.load(open(os.path.join(ROOT, "artifacts/split.json")))["fit_ids"])
    counts = defaultdict(Counter)  # (field, country) -> Counter
    files = [("train_source1.tsv", True), ("train_source2.tsv", False), ("train_source3.tsv", False)]
    want = set(CANDS)
    for fn, filt in files:
        with open(os.path.join(TR, fn), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if filt and row["entity_id"] not in fit:
                    continue
                co = row["country"]
                for t in TOKEN_RE.findall((row["business_name"] or "").lower()):
                    if t in want:
                        counts[("name", co)][t] += 1
                for t in TOKEN_RE.findall((row["business_address"] or "").lower()):
                    if t in want:
                        counts[("addr", co)][t] += 1
        print("done", fn, flush=True)
    out = {"%s|%s" % k: dict(v) for k, v in sorted(counts.items(), key=lambda x: str(x[0]))}
    with open(os.path.join(MET, "normdict_evidence.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k in sorted(out):
        print(k, sorted(out[k].items(), key=lambda x: -x[1])[:30])


if __name__ == "__main__":
    main()
