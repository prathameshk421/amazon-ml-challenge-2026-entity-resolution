"""Items 16 (adaptive threshold surface) and 19 (calibration variants) on a run's MATCH_TRAIN OOF scores.

No model is trained. Every variant is CROSS-FITTED over the run's 5 GroupKFold folds: calibration maps and
thresholds are fitted on 4 folds' OOF rows and applied to the 5th, so no row is scored by a rule fitted on it.
One-owner (within MATCH_TRAIN) is applied in every variant, as in the live decision.

Variants (paired per-entity delta vs A):
  A  global isotonic + global threshold                          (= the live recipe, cross-fitted)
  B  isotonic per list-size bucket (1-3, 4-5, 6-8, 9+) + global threshold
  C  Platt scaling (logistic on logit p_raw) + global threshold
  D  global isotonic + threshold per list-size bucket            (threshold surface, 1-D)
  E  global isotonic + threshold per (list-size bucket x top-2 p_rerank gap < 0.1)   (2-D surface)

Run: PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/decision_variants.py RUN
"""
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from ber import exp, io_utils, match

GRID = np.round(np.arange(0.30, 0.951, 0.01), 2)


def size_bucket(n):
    return np.digitize(n, [4, 6, 9])  # 0: 1-3, 1: 4-5, 2: 6-8, 3: 9+


def iso_fit(p, y):
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)


def best_threshold(sc_rows, qpos_rows, y, p, n_true_q, pool, mask_rows=None):
    """Threshold maximizing mean entity F0.5 over the S1 touched by the fit rows (one-owner applied)."""
    best, bt = -1.0, 0.5
    pc = match.one_owner(pool, p)
    for t in GRID:
        keep = pc >= t
        if mask_rows is not None:
            keep = keep & mask_rows
        f = entity_f(qpos_rows, keep, y, n_true_q)
        if f > best:
            best, bt = f, t
    return bt


def entity_f(qpos, keep, y, n_true_q):
    nq = n_true_q.size
    n_pred = np.bincount(qpos, weights=keep, minlength=nq)
    n_tp = np.bincount(qpos, weights=keep & y, minlength=nq)
    touched = np.bincount(qpos, minlength=nq) > 0
    return match.entity_f05(n_pred, n_true_q, n_tp)[touched].mean()


def main():
    run = sys.argv[1]
    cfg, mcfg = io_utils.load_config(), match.load_match_config()
    t = pq.read_table(match.runs_dir(run) / "oof.parquet").to_pandas()
    s1, pool, y, p_raw = t["s1_idx"].to_numpy(), t["pool_idx"].to_numpy(), t["y"].to_numpy(), t["p_raw"].to_numpy()
    queries = match.set_queries("MATCH_TRAIN")
    n_true = match.true_counts(queries)
    sc = match.Scorer(queries, s1, y, n_true)
    qpos = sc.qpos
    X, fs1, fpool, _ = match.load_features("MATCH_TRAIN")
    if not (np.array_equal(fs1, s1) and np.array_equal(fpool, pool)):
        raise RuntimeError("OOF rows and feature rows differ in order")
    lsize = np.bincount(qpos, minlength=queries.size)[qpos]
    sb = size_bucket(lsize)
    prr = X[:, match._COL["p_rerank"]].astype(np.float64)
    top = pd.Series(prr).groupby(s1).transform(lambda v: np.sort(v.to_numpy())[-1] - (np.sort(v.to_numpy())[-2] if v.size > 1 else 0.0)).to_numpy()
    tie = (top < 0.1).astype(int)
    cell = sb * 2 + tie

    gkf = GroupKFold(n_splits=mcfg["n_folds"], shuffle=True, random_state=mcfg["fold_seed"])
    keeps = {k: np.zeros(y.size, bool) for k in "ABCDE"}
    for tr, te in gkf.split(p_raw, y, s1):
        # A: global isotonic + global threshold
        iso = iso_fit(p_raw[tr], y[tr])
        pa_tr, pa_te = iso.predict(p_raw[tr]), iso.predict(p_raw[te])
        tA = best_threshold(None, qpos[tr], y[tr], pa_tr, n_true, pool[tr])
        keeps["A"][te] = match.one_owner(pool[te], pa_te) >= tA
        # B: isotonic per list-size bucket
        pb_tr, pb_te = np.zeros(tr.size), np.zeros(te.size)
        for b in range(4):
            mtr, mte = sb[tr] == b, sb[te] == b
            cal = iso_fit(p_raw[tr][mtr], y[tr][mtr]) if mtr.sum() > 1000 else iso
            if mtr.any():
                pb_tr[mtr] = cal.predict(p_raw[tr][mtr])
            if mte.any():
                pb_te[mte] = cal.predict(p_raw[te][mte])
        tB = best_threshold(None, qpos[tr], y[tr], pb_tr, n_true, pool[tr])
        keeps["B"][te] = match.one_owner(pool[te], pb_te) >= tB
        # C: Platt
        lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]
        lr = LogisticRegression(C=1e6).fit(lo(p_raw[tr]), y[tr])
        pc_tr, pc_te = lr.predict_proba(lo(p_raw[tr]))[:, 1], lr.predict_proba(lo(p_raw[te]))[:, 1]
        tC = best_threshold(None, qpos[tr], y[tr], pc_tr, n_true, pool[tr])
        keeps["C"][te] = match.one_owner(pool[te], pc_te) >= tC
        # D / E: threshold per cell, fitted on the rows of that cell only (other rows held at the global rule)
        for key, cells in (("D", sb), ("E", cell)):
            oo_tr, oo_te = match.one_owner(pool[tr], pa_tr), match.one_owner(pool[te], pa_te)
            kd = oo_te >= tA
            base_tr = oo_tr >= tA
            for c in np.unique(cells):
                mtr = cells[tr] == c
                if mtr.sum() < 2000:
                    continue
                # choose the cell threshold with other cells fixed at tA (coordinate step)
                best, bt = -1.0, tA
                for th in GRID:
                    kk = np.where(mtr, oo_tr >= th, base_tr)
                    f = entity_f(qpos[tr], kk, y[tr], n_true)
                    if f > best:
                        best, bt = f, th
                mte = cells[te] == c
                kd[mte] = (oo_te >= bt)[mte]
            keeps[key][te] = kd

    per = {k: sc.per_entity(v)[0] for k, v in keeps.items()}
    band = exp.band_mask(queries)
    out = {"script": "plan/checks/decision_variants.py", "run": run,
           "f05": {k: float(v.mean()) for k, v in per.items()},
           "vs_A": {k: match.paired_delta(per[k], per["A"], n_boot=2000) for k in "BCDE"},
           "vs_A_band_s1": {k: match.paired_delta(per[k][band], per["A"][band]) for k in "BCDE"}}
    io_utils.write_metrics(f"decision_variants_{run}", out, queries.size, int(n_true.sum()), cfg)
    for k in "BCDE":
        print(k, round(out["f05"][k], 5), out["vs_A"][k], out["vs_A_band_s1"][k]["delta"])
    print("A", out["f05"]["A"])


if __name__ == "__main__":
    main()
