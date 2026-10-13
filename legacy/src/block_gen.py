"""Phase 2c: parallel candidate generation (fork; parent pre-loads read-only index).
Usage: ./.venv/bin/python legacy/src/block_gen.py <entity_json> <out.parquet> <K>
entity_json: {"ids": [...]} ; out: parquet {s1_id, cand: list<int32 rowid>}.
Encoding: rowid -> (src,num) via artifacts/index/pool.{src,num}.npy (documented).
Candidates = LAST blocking stage the matcher scores (C6).
"""

import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_query2 import Q
from normalize import normalize_name, normalize_address

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")

QQ = None
def init():
    global QQ
    QQ = Q()

def work(chunk):
    import csv
    want = set(e for e, _, _ in chunk)
    recs = {}
    with open(os.path.join(TR, "train_source1.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in want:
                recs[row["entity_id"]] = (row["business_name"], row["business_address"])
    out = []
    for e, _, K in chunk:
        nm, ad = recs[e]
        top, _ = QQ.ranked(normalize_name(nm), normalize_address(ad), K=K)
        out.append((e, np.array(top, dtype=np.int32)))
    return out


def main():
    ent = json.load(open(sys.argv[1]))["ids"]
    outp, K = sys.argv[2], int(sys.argv[3])
    global QQ
    QQ = Q()  # parent preloads BEFORE fork so workers share read-only pages
    log = lambda m: print(m, flush=True)
    log("index preloaded in parent; entities=%d K=%d" % (len(ent), K))
    chunks = [(ent[i], None, K) for i in range(len(ent))]
    # batch into 24 chunks of triples
    B = (len(chunks) + 23) // 24
    batches = [[(e, None, K) for e in ent[i:i + B]] for i in range(0, len(ent), B)]
    ctx = mp.get_context("fork")
    allrows = []
    with ctx.Pool(6, maxtasksperchild=2) as pool:
        for bi, res in enumerate(pool.imap_unordered(work, batches)):
            allrows.extend(res)
            log("batch %d/%d done (%d rows)" % (bi + 1, len(batches), len(allrows)))
    df = pd.DataFrame({"s1_id": [e for e, _ in allrows],
                       "cand": [c for _, c in allrows]})
    df.to_parquet(outp)
    log("WROTE %s rows=%d avg_cand=%.1f" % (outp, len(df), np.mean([len(c) for _, c in allrows])))


if __name__ == "__main__":
    main()
