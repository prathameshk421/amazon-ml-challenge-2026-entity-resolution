"""Phase 3b: LightGBM with 5-fold GroupKFold OOF (by S1 entity) on fit-sample.
Compares uniform vs hard-upweighted negatives (3-fold screen), refits winner,
calibrates (isotonic on OOF), picks threshold t (macro-F0.5, R1 on OOF) + t2 (R3 on OOF).
Saves artifacts/model.pkl, artifacts/calib.pkl, artifacts/threshold.json,
artifacts/oof_preds.parquet; metrics -> artifacts/metrics/ml_oof.json.
"""

import json
import os
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES
from metric import macro_f05

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "artifacts")
MET = os.path.join(ROOT, "artifacts/metrics")

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)

PARAMS = {"objective": "binary", "num_leaves": 63, "learning_rate": 0.05,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
          "min_data_in_leaf": 100, "n_jobs": 6, "verbosity": -1}


def oof_eval(oof, entities, truth, rule="r1", t=0.5, t2=0.9):
    pred = {}
    df = oof
    if rule == "r1":
        keep = df[df.p >= t]
        pred = keep.groupby("s1_id")["cand_id"].apply(set).to_dict()
    elif rule == "r2":
        idx = df[df.p >= t].groupby("s1_id")["p"].idxmax()
        top = df.loc[idx]
        pred = dict(zip(top.s1_id, [[c] for c in top.cand_id]))
        pred = {k: set(v) for k, v in pred.items()}
    elif rule == "r3":
        idx = df[df.p >= t].groupby("s1_id")["p"].idxmax()
        top = set(df.loc[idx].index)
        keep = df[((df.p >= t2)) | (df.index.isin(top) & (df.p >= t))]
        pred = keep.groupby("s1_id")["cand_id"].apply(set).to_dict()
    return macro_f05(pred, truth, all_entities=entities)


def main():
    X = pd.read_parquet(os.path.join(ART, "X_train.parquet"))
    log("X_train %s posrate=%.4f" % (X.shape, X.label.mean()))
    y = X["label"].values.astype(int)
    groups = X["s1_id"].values
    ent = np.unique(groups)
    entset = set(ent.tolist())
    truth = {}
    import csv
    with open(os.path.join(ROOT, "student_resource/dataset/train/train_ground_truth.tsv"),
              newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["source1_entity_id"] in entset:
                m = row["matched_entity_ids"]
                truth[row["source1_entity_id"]] = set(m.split(",")) if m else set()
    F = X[FEATURES].values.astype(np.float32)
    hard = (X["n_tset"].values >= 0.7) & (y == 0)

    def run_oof(weights, nfold=5, seed=0):
        oof = np.zeros(len(X))
        gkf = GroupKFold(nfold)
        for fi, (tr, va) in enumerate(gkf.split(F, y, groups)):
            w = weights[tr] if weights is not None else None
            ds = lgb.Dataset(F[tr], label=y[tr], weight=w)
            m = lgb.train(PARAMS, ds, num_boost_round=400)
            oof[va] = m.predict(F[va])
            log("fold %d done" % fi)
        return oof

    # screen weightings with 3-fold; winner runs the 5-fold
    screen = {}
    for name, w in [("uniform", None), ("hardx3", np.where(hard, 3.0, 1.0))]:
        o = run_oof(w, nfold=3)
        oo = pd.DataFrame({"s1_id": groups, "cand_id": X["cand_id"].values, "p": o})
        best = max(((t, oof_eval(oo, ent, truth, "r1", t)["F0.5"]) for t in np.arange(0.1, 0.9, 0.05)),
                   key=lambda x: x[1])
        screen[name] = best[1]
        log("screen %s: best t=%.2f F0.5=%.4f" % (name, best[0], best[1]))
    winner = "hardx3" if screen["hardx3"] >= screen["uniform"] else "uniform"
    log("winner weighting: %s" % winner)
    wwin = np.where(hard, 3.0, 1.0) if winner == "hardx3" else None
    oof = run_oof(wwin, nfold=5)
    oof_df = pd.DataFrame({"s1_id": groups, "cand_id": X["cand_id"].values, "p": oof})
    cal = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    oof_df["p"] = cal.predict(oof)
    ts = np.arange(0.05, 0.96, 0.01)
    curve = [(float(t), oof_eval(oof_df, ent, truth, "r1", float(t))["F0.5"]) for t in ts]
    t_best, f_best = max(curve, key=lambda x: x[1])
    t2_best, f2_best = t_best, f_best
    for t2 in np.arange(t_best, 1.0, 0.02):
        f = oof_eval(oof_df, ent, truth, "r3", t_best, float(t2))["F0.5"]
        if f > f2_best:
            f2_best, t2_best = f, float(t2)
    log("OOF R1: t=%.2f F0.5=%.4f | R3: t2=%.2f F0.5=%.4f" % (t_best, f_best, t2_best, f2_best))
    # refit on all
    ds = lgb.Dataset(F, label=y, weight=wwin)
    model = lgb.train(PARAMS, ds, num_boost_round=400)
    with open(os.path.join(ART, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(ART, "calib.pkl"), "wb") as f:
        pickle.dump(cal, f)
    json.dump({"t": t_best, "t2": t2_best, "weighting": winner,
               "screen": screen, "features": FEATURES},
              open(os.path.join(ART, "threshold.json"), "w"), indent=1)
    oof_df.to_parquet(os.path.join(ART, "oof_preds.parquet"))
    res = {"weighting": winner, "screen": screen,
           "oof_r1_t": t_best, "oof_r1_f05": f_best,
           "oof_r3_t2": t2_best, "oof_r3_f05": f2_best,
           "oof_curve": [[t, f] for t, f in curve],
           "n_pairs": len(X), "n_entities": len(ent)}
    json.dump(res, open(os.path.join(MET, "ml_oof.json"), "w"), indent=1)
    log("WROTE model + ml_oof.json")


if __name__ == "__main__":
    main()
