"""Phonetic marginal: true pairs found ONLY via phonetic (no EX/NT/TR/AD vote).
Same 10K queries (seed 123). Writes artifacts/metrics/blocking_phon.json.
Run: .venv/bin/python legacy/src/block_phon.py
"""

import csv
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_query2 import Q
from normalize import normalize_name, normalize_address

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
IDX = os.path.join(ROOT, "artifacts/index")
MET = os.path.join(ROOT, "artifacts/metrics")

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


def main():
    rng = random.Random(123)
    split = json.load(open(os.path.join(ROOT, "artifacts/split.json")))
    qids = rng.sample(split["fit_ids"], 10000)
    want = set(qids)
    recs, truth = {}, {}
    with open(os.path.join(TR, "train_source1.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in want:
                recs[row["entity_id"]] = (row["business_name"], row["business_address"])
    with open(os.path.join(TR, "train_ground_truth.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["source1_entity_id"] in want:
                m = row["matched_entity_ids"]
                truth[row["source1_entity_id"]] = m.split(",") if m else []
    src = np.load(os.path.join(IDX, "pool.src.npy"))
    num = np.load(os.path.join(IDX, "pool.num.npy"))
    key = (src.astype(np.int64) << 32) | num
    order = np.argsort(key)
    skey = key[order]
    q = Q()
    tot = phonly = found = 0
    ex_only = {"EX": 0, "NT": 0, "TR": 0, "AD": 0, "PH": 0}
    for e in qids:
        tro = []
        for x in truth[e]:
            s = 0 if x.startswith("S2-") else 1
            k = (np.int64(s) << np.int64(32)) | np.int64(int(x.split("-", 1)[1]))
            j = int(np.searchsorted(skey, k))
            if j < len(skey) and skey[j] == k:
                tro.append(int(order[j]))
        top, flags = q.ranked(normalize_name(recs[e][0]), normalize_address(recs[e][1]), K=500)
        for x in tro:
            tot += 1
            if x in flags:
                found += 1
                f = flags[x]
                only = []
                if f & 1:
                    only.append("EX")
                if f & 2:
                    only.append("NT")
                if f & 4:
                    only.append("TR")
                if f & 8:
                    only.append("AD")
                if f & 16:
                    only.append("PH")
                if only == ["PH"]:
                    phonly += 1
                if len(only) == 1:
                    ex_only[only[0]] += 1
    out = {"n_true": tot, "found_top500": found,
           "phonetic_only": phonly, "phonetic_only_frac": phonly / tot if tot else 0,
           "exclusive_by_source": ex_only,
           "exclusive_frac": {k: v / tot if tot else 0 for k, v in ex_only.items()}}
    json.dump(out, open(os.path.join(MET, "blocking_phon.json"), "w"), indent=1)
    log(json.dumps(out))


if __name__ == "__main__":
    main()
