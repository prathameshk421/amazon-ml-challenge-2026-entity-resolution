"""Fork-parallel featurization (parent preloads; workers inherit via fork).
Usage: ./.venv/bin/python legacy/src/featurize.py <pairs_parquet> <out_parquet>
"""

import csv
import json
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build, FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
IDX = os.path.join(ROOT, "artifacts/index")
ART = os.path.join(ROOT, "artifacts")

VN = VA = TIDF = S1T = None
NBLOB = NOFF = ABLOB = AOFF = PSRC = PNUM = PCO = INV = SKEY = ORDER = None


def preload(s1ids):
    global VN, VA, TIDF, S1T, NBLOB, NOFF, ABLOB, AOFF, PSRC, PNUM, PCO, INV, SKEY, ORDER
    with open(os.path.join(ART, "tfidf_name.pkl"), "rb") as f:
        VN = pickle.load(f)
    with open(os.path.join(ART, "tfidf_addr.pkl"), "rb") as f:
        VA = pickle.load(f)
    TIDF = json.load(open(os.path.join(ART, "tok_idf.json")))
    want = set(s1ids)
    S1T = {}
    with open(os.path.join(TR, "train_source1.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in want:
                S1T[row["entity_id"]] = (row["business_name"] or "", row["business_address"] or "",
                                         row["country"] or "")
    NBLOB = np.memmap(os.path.join(IDX, "pool.names.bin"), mode="r", dtype=np.uint8)
    NOFF = np.load(os.path.join(IDX, "pool.names.off.npy"), mmap_mode="r")
    ABLOB = np.memmap(os.path.join(IDX, "pool.addrs.bin"), mode="r", dtype=np.uint8)
    AOFF = np.load(os.path.join(IDX, "pool.addrs.off.npy"), mmap_mode="r")
    PSRC = np.load(os.path.join(IDX, "pool.src.npy"), mmap_mode="r")
    PNUM = np.load(os.path.join(IDX, "pool.num.npy"), mmap_mode="r")
    PCO = np.load(os.path.join(IDX, "pool.co.npy"), mmap_mode="r")
    cmap = json.load(open(os.path.join(IDX, "pool.meta.json")))["cmap"]
    INV = {v: k for k, v in cmap.items()}
    key = (PSRC.astype(np.int64) << 32) | PNUM
    ORDER = np.argsort(key)
    SKEY = key[ORDER]
    print("preloaded s1=%d" % len(S1T), flush=True)


def pool_text(x):
    s = 0 if x.startswith("S2-") else 1
    k = (np.int64(s) << np.int64(32)) | np.int64(int(x.split("-", 1)[1]))
    j = int(np.searchsorted(SKEY, k))
    ri = int(ORDER[j])
    return (bytes(NBLOB[NOFF[ri]:NOFF[ri + 1]]).decode("utf-8", "replace"),
            bytes(ABLOB[AOFF[ri]:AOFF[ri + 1]]).decode("utf-8", "replace"),
            INV[int(PCO[ri])])


def work(chunk):
    rows = []
    pt_cache = {}
    for e, c in zip(chunk["s1_id"].tolist(), chunk["cand_id"].tolist()):
        q = S1T[e]
        r = pt_cache.get(c)
        if r is None:
            r = pool_text(c)
            pt_cache[c] = r
        rows.append((q[0], q[1], q[2], r[0], r[1], r[2]))
    df = pd.DataFrame(rows, columns=["s1_name", "s1_addr", "s1_co", "c_name", "c_addr", "c_co"])
    X = build(df, vec_n=VN, vec_a=VA, tok_idf=TIDF)
    X["s1_id"] = chunk["s1_id"].values
    X["cand_id"] = chunk["cand_id"].values
    if "label" in chunk:
        X["label"] = chunk["label"].values
    return X


def main():
    pairs, outp = sys.argv[1], sys.argv[2]
    df = pd.read_parquet(pairs)
    print("pairs=%d" % len(df), flush=True)
    preload(df["s1_id"].tolist())
    CH = 500000
    chunks = [df[i:i + CH] for i in range(0, len(df), CH)]
    ctx = mp.get_context("fork")
    parts = []
    with ctx.Pool(6, maxtasksperchild=1) as pool:
        for bi, res in enumerate(pool.imap_unordered(work, chunks)):
            parts.append(res)
            print("chunk %d/%d" % (bi + 1, len(chunks)), flush=True)
    X = pd.concat(parts, ignore_index=True)
    X.to_parquet(outp)
    print("WROTE %s rows=%d" % (outp, len(X)), flush=True)


if __name__ == "__main__":
    main()
