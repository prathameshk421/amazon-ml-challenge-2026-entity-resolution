"""Ablation-lite: drop-one-feature retrains on 15K-entity subsample, delta on 30K val subset.
Fixed OOF threshold t. Writes artifacts/metrics/ablation.json.
"""

import json
import os
import pickle
import random
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES
from metric import macro_f05
from train_oof import PARAMS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "artifacts")
MET = os.path.join(ROOT, "artifacts/metrics")

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


def main():
    rng = random.Random(5)
    X = pd.read_parquet(os.path.join(ART, "X_train.parquet"))
    ents = sorted(X.s1_id.unique().tolist())
    sub = set(rng.sample(ents, 15000))
    Xs = X[X.s1_id.isin(sub)].reset_index(drop=True)
    log("sub train %s" % (Xs.shape,))
    th = json.load(open(os.path.join(ART, "threshold.json")))
    t = th["t"]
    hard = (Xs["n_tset"].values >= 0.7) & (Xs.label.values.astype(int) == 0)
    w = np.where(hard, 3.0, 1.0) if th["weighting"] == "hardx3" else None
    V = pd.read_parquet(os.path.join(ART, "val_probs.parquet"), columns=["s1_id"])
    vents = sorted(V.s1_id.unique().tolist())
    vsub = set(rng.sample(vents, 30000))
    Pv = pd.read_parquet(os.path.join(ART, "val_probs.parquet"))
    Pv = Pv[Pv.s1_id.isin(vsub)].reset_index(drop=True)
    import csv
    truth = {}
    with open(os.path.join(ROOT, "student_resource/dataset/train/train_ground_truth.tsv"),
              newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["source1_entity_id"] in vsub:
                m = row["matched_entity_ids"]
                truth[row["source1_entity_id"]] = set(m.split(",")) if m else set()
    # val subset features: need X for these pairs -> featurize on the fly (reuse val chunk flow)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from featurize import preload
    import featurize as FZ
    from features import build
    preload(Pv["s1_id"].tolist())
    CH = 500000
    parts = []
    for i in range(0, len(Pv), CH):
        ch = Pv[i:i + CH]
        rows = []
        for e, c in zip(ch["s1_id"], ch["cand_id"]):
            qq = FZ.S1T[e]
            from featurize import pool_text
            rr = pool_text(c)
            rows.append((qq[0], qq[1], qq[2], rr[0], rr[1], rr[2]))
        d = pd.DataFrame(rows, columns=["s1_name", "s1_addr", "s1_co", "c_name", "c_addr", "c_co"])
        parts.append(build(d, vec_n=FZ.VN, vec_a=FZ.VA, tok_idf=FZ.TIDF))
    Xv = pd.concat(parts, ignore_index=True)
    log("val-sub features %s" % (Xv.shape,))

    def score(feats):
        ds = lgb.Dataset(Xs[feats].values.astype(np.float32), label=Xs.label.values.astype(int),
                         weight=w)
        m = lgb.train(PARAMS, ds, num_boost_round=300)
        p = m.predict(Xv[feats].values.astype(np.float32))
        keep = Pv[p >= t]
        pred = keep.groupby("s1_id")["cand_id"].apply(set).to_dict()
        return macro_f05(pred, truth, all_entities=sorted(vsub))["F0.5"]

    base = score(FEATURES)
    log("base(sub-train, val-sub) F0.5=%.4f" % base)
    res = {"base": base, "t": t, "note": "15K-entity retrains; 30K val subset; fixed OOF t (val-full infeasible tonight)"}
    for f in FEATURES:
        feats = [x for x in FEATURES if x != f]
        d = score(feats)
        res[f] = {"f05": d, "delta": d - base}
        log("drop %-14s F0.5=%.4f delta=%+.4f" % (f, d, d - base))
    json.dump(res, open(os.path.join(MET, "ablation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
