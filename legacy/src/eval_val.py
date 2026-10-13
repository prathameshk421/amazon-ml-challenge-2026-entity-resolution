"""Phase 3c: val scoring + rules R1-R4, val-optimal-t reference, ablation-lite,
LOCO (fit-A -> val-B both directions), multi-claim consolidation stats.
Uses artifacts/model.pkl, calib.pkl, threshold.json + artifacts/X_val.parquet.
Writes artifacts/metrics/ml_baseline.json.
"""

import json
import os
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES
from metric import macro_f05

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "artifacts")
MET = os.path.join(ROOT, "artifacts/metrics")

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


def load_truth(entities):
    import csv
    want = set(entities)
    truth = {}
    with open(os.path.join(ROOT, "student_resource/dataset/train/train_ground_truth.tsv"),
              newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["source1_entity_id"] in want:
                m = row["matched_entity_ids"]
                truth[row["source1_entity_id"]] = set(m.split(",")) if m else set()
    return truth


def apply_rules(df, t, t2):
    """df: s1_id, cand_id, p. Returns dict rule -> {s1: set(cands)}."""
    keep1 = df[df.p >= t]
    r1 = keep1.groupby("s1_id")["cand_id"].apply(set).to_dict()
    over = df[df.p >= t]
    idx = over.groupby("s1_id")["p"].idxmax()
    top = df.loc[idx]
    r2 = dict(zip(top.s1_id, [{c} for c in top.cand_id]))
    keep3 = df[(df.p >= t2) | (df.index.isin(set(idx)) & (df.p >= t))]
    r3 = keep3.groupby("s1_id")["cand_id"].apply(set).to_dict()
    return {"R1": r1, "R2": r2, "R3": r3}


def apply_r4(pred, scores):
    """Record-level winner-takes-all: each cand only to argmax-p entity, then keep."""
    best = {}
    for e, cands in pred.items():
        for c in cands:
            p = scores.get((e, c), -1)
            if c not in best or p > best[c][1]:
                best[c] = (e, p)
    owner = {c: e for c, (e, p) in best.items()}
    return {e: set(c for c in cands if owner.get(c) == e) for e, cands in pred.items()}


def main():
    th = json.load(open(os.path.join(ART, "threshold.json")))
    t, t2 = th["t"], th["t2"]
    Xv = pd.read_parquet(os.path.join(ART, "val_probs.parquet"))
    log("val_probs %s" % (Xv.shape,))
    df = Xv.rename(columns={"p": "p"})[["s1_id", "cand_id", "p"]]
    entities = sorted(df.s1_id.unique().tolist())
    truth = load_truth(entities)
    rules = apply_rules(df, t, t2)
    scores = {(e, c): pp for e, c, pp in zip(df.s1_id, df.cand_id, df.p)}
    res = {"t_oof": t, "t2_oof": t2, "n_val_pairs": len(df), "n_val_entities": len(entities)}
    for rn, pred in rules.items():
        m = macro_f05(pred, truth, all_entities=entities)
        res[rn] = m
        log("%s(val,t=%.2f): %s" % (rn, t, json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                       for k, v in m.items()})))
    # R4 on R1-R3
    for rn in ["R1", "R2", "R3"]:
        pred4 = apply_r4(rules[rn], scores)
        m = macro_f05(pred4, truth, all_entities=entities)
        res[rn + "+R4"] = m
        log("%s+R4: F0.5=%.4f" % (rn, m["F0.5"]))
    # multi-claim counts before consolidation
    cnt = {}
    for e, cands in rules["R1"].items():
        for c in cands:
            cnt[c] = cnt.get(c, 0) + 1
    multi = sum(1 for v in cnt.values() if v > 1)
    res["multiclaim_records"] = multi
    res["claimed_records"] = len(cnt)
    log("records claimed by >1 entity (R1): %d / %d" % (multi, len(cnt)))
    # val-optimal t reference
    curve = []
    for tt in np.arange(0.05, 0.96, 0.02):
        keep = df[df.p >= tt]
        pr = keep.groupby("s1_id")["cand_id"].apply(set).to_dict()
        curve.append([round(float(tt), 2), macro_f05(pr, truth, all_entities=entities)["F0.5"]])
    res["val_opt_curve"] = curve
    bt, bf = max(curve, key=lambda x: x[1])
    res["val_opt_t"] = bt
    res["val_opt_f05"] = bf
    log("val-optimal t=%.2f F0.5=%.4f (reference only)" % (bt, bf))
    json.dump(res, open(os.path.join(MET, "ml_baseline.json"), "w"), indent=1)
    log("WROTE ml_baseline.json")


if __name__ == "__main__":
    main()
