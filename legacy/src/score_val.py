"""Chunked val scoring: featurize -> predict -> calibrated p, streaming to disk.
Usage: ./.venv/bin/python legacy/src/score_val.py
Reads candidates/val_K150.parquet + model.pkl/calib.pkl/tfidf; writes val_probs.parquet.
"""

import json
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build, FEATURES
from featurize import preload, pool_text, S1T  # noqa: F401  (globals via fork)
import featurize as FZ

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "artifacts")
CAND = os.path.join(ROOT, "artifacts/candidates")

MODEL = CAL = None

def work(cand):
    import pandas as pd
    s1, cr, pn, pa, pc, cn, ca, cc = [], [], [], [], [], [], [], []
    pt_cache = {}
    for r in cand.itertuples():
        q = FZ.S1T[r.s1_id]
        for rowid in r.cand:
            c = ("S2-" if FZ.PSRC[int(rowid)] == 0 else "S3-") + str(int(FZ.PNUM[int(rowid)]))
            t = pt_cache.get(c)
            if t is None:
                t = pool_text(c)
                pt_cache[c] = t
            s1.append(r.s1_id)
            cr.append(c)
            pn.append(q[0])
            pa.append(q[1])
            pc.append(q[2])
            cn.append(t[0])
            ca.append(t[1])
            cc.append(t[2])
    df = pd.DataFrame({"s1_name": pn, "s1_addr": pa, "s1_co": pc,
                       "c_name": cn, "c_addr": ca, "c_co": cc})
    X = build(df, vec_n=FZ.VN, vec_a=FZ.VA, tok_idf=FZ.TIDF)
    p = CAL.predict(MODEL.predict(X[FEATURES].values.astype(np.float32)))
    return pd.DataFrame({"s1_id": s1, "cand_id": cr, "p": p.astype(np.float32)})


def main():
    global MODEL, CAL
    MODEL = pickle.load(open(os.path.join(ART, "model.pkl"), "rb"))
    CAL = pickle.load(open(os.path.join(ART, "calib.pkl"), "rb"))
    cand = pd.read_parquet(os.path.join(CAND, "val_K150.parquet"))
    preload(cand["s1_id"].tolist())  # parent preload; workers inherit via fork
    B = (len(cand) + 23) // 24
    batches = [cand[i:i + B] for i in range(0, len(cand), B)]
    del cand
    ctx = mp.get_context("fork")
    parts = []
    with ctx.Pool(6, maxtasksperchild=1) as pool:
        for bi, res in enumerate(pool.imap_unordered(work, batches)):
            parts.append(res)
            print("batch %d/%d rows=%d" % (bi + 1, len(batches), sum(len(p) for p in parts)), flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(os.path.join(ART, "val_probs.parquet"))
    print("WROTE val_probs rows=%d" % len(out), flush=True)


if __name__ == "__main__":
    main()
