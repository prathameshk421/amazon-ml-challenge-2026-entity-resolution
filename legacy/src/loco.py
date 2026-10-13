"""LOCO: train on fit-sample country A, eval val country B (both directions, full val-B).
Writes artifacts/metrics/loco.json. Uses winner weighting + OOF t from threshold.json.
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES
from metric import macro_f05
from train_oof import PARAMS, oof_eval

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "artifacts")
MET = os.path.join(ROOT, "artifacts/metrics")
TR = os.path.join(ROOT, "student_resource/dataset/train")

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


def main():
    import csv
    X = pd.read_parquet(os.path.join(ART, "X_train.parquet"))
    th = json.load(open(os.path.join(ART, "threshold.json")))
    t = th["t"]
    co = {}
    with open(os.path.join(TR, "train_source1.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            co[row["entity_id"]] = row["country"]
    X["co"] = [co.get(e, "") for e in X.s1_id]
    Pv = pd.read_parquet(os.path.join(ART, "val_probs.parquet"), columns=["s1_id", "cand_id"])
    Pv["co"] = [co.get(e, "") for e in Pv.s1_id]
    truth = {}
    with open(os.path.join(TR, "train_ground_truth.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            m = row["matched_entity_ids"]
            truth[row["source1_entity_id"]] = set(m.split(",")) if m else set()
    # featurize val-B subsets on the fly
    from featurize import preload
    import featurize as FZ
    from features import build
    from featurize import pool_text
    res = {}
    for A, B in [("US", "India"), ("India", "US")]:
        Xa = X[X.co == A].reset_index(drop=True)
        ya = Xa.label.values.astype(int)
        hard = (Xa["n_tset"].values >= 0.7) & (ya == 0)
        w = np.where(hard, 3.0, 1.0) if th["weighting"] == "hardx3" else None
        ds = lgb.Dataset(Xa[FEATURES].values.astype(np.float32), label=ya, weight=w)
        m = lgb.train(PARAMS, ds, num_boost_round=400)
        sub = Pv[Pv.co == B].reset_index(drop=True)
        ents = sorted(sub.s1_id.unique().tolist())
        preload(ents)
        CH = 500000
        ps = []
        for i in range(0, len(sub), CH):
            ch = sub[i:i + CH]
            rows = []
            for e, c in zip(ch["s1_id"], ch["cand_id"]):
                qq = FZ.S1T[e]
                rr = pool_text(c)
                rows.append((qq[0], qq[1], qq[2], rr[0], rr[1], rr[2]))
            d = pd.DataFrame(rows, columns=["s1_name", "s1_addr", "s1_co", "c_name", "c_addr", "c_co"])
            Xb = build(d, vec_n=FZ.VN, vec_a=FZ.VA, tok_idf=FZ.TIDF)
            ps.append(m.predict(Xb[FEATURES].values.astype(np.float32)))
        p = np.concatenate(ps)
        keep = sub[p >= t]
        pred = keep.groupby("s1_id")["cand_id"].apply(set).to_dict()
        mm = macro_f05(pred, truth, all_entities=ents)
        res["fit%s_eval%s" % (A, B)] = {"n_train": len(Xa), "n_eval": len(ents), **mm}
        log("fit%s->val%s: %s" % (A, B, json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                    for k, v in mm.items()})))
    json.dump(res, open(os.path.join(MET, "loco.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
