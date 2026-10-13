"""O1: combine c0 p_ce with LightGBM p_cal on in-list pairs (override accept / veto / logit blend), then one-owner and
the S13 DRACE additions (c0 >= 0.995, pool records owned by the base excluded), plus a per-source count cap check.
MATCH_TRAIN (M027 OOF, the S13-equivalent base of admit_sweep.py); labels for scoring only; rule chosen by S1 half
and scored on the other half. Writes artifacts/metrics/c0_override.json.

  PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/c0_override.py
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from gate2_eval import f05

THR = 0.73


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    q = np.load(match.runs_dir("M027") / "queries.npy")
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas()
    gt = gt[gt.s1_idx.isin(q)]
    Tn = gt.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    o = pq.read_table(match.runs_dir("M027") / "oof.parquet").to_pandas()
    ce = pq.read_table("work/ce/ce_rec_c0/MATCH_TRAIN.parquet").to_pandas()
    o = o.merge(ce, on=["s1_idx", "pool_idx"], how="left")
    print(f"in-list rows {len(o):,}, p_ce missing {o.p_ce.isna().sum():,}", flush=True)
    o["p_ce"] = o.p_ce.fillna(0.0)
    a = pd.read_parquet("work/retrieval/gate2_meta.parquet")
    a["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
    a = a[(a.s1_rank <= 5) | (a.seed_rank <= 2)]
    d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet")
    d["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2DS_TRAIN.parquet").column("p_ce").to_numpy()
    cols = ["s1_idx", "pool_idx", "p", "label"]
    dr = pd.concat([a[cols], d[cols]]).drop_duplicates(["s1_idx", "pool_idx"])
    dr = dr[dr.p >= 0.995].sort_values("p", ascending=False)
    s1, pl, y = o.s1_idx.to_numpy(), o.pool_idx.to_numpy(), o.y.to_numpy()
    pc, pe = o.p_cal.to_numpy(), o.p_ce.to_numpy()

    def score(s, thr=THR):
        k = match.apply_decision({"rule": "threshold", "threshold": thr, "one_owner": True}, s1, pl, s)
        acc = o[k]
        mm = dr[~dr.pool_idx.isin(set(acc.pool_idx.tolist()))].drop_duplicates("pool_idx")
        P = (acc.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy()
             + mm.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy()).astype(float)
        TP = (acc[acc.y].groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy()
              + mm[mm.label == 1].groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy()).astype(float)
        return f05(Tn, P, TP), k

    base, kb = score(pc)
    print(f"base F0.5 {base.mean():.5f}", flush=True)
    rej, accd = ~kb, kb
    for lo, hi in ((0.0, 0.2), (0.2, 0.5), (0.5, THR)):
        m = rej & (pc >= lo) & (pc < hi)
        for t in (0.99, 0.999):
            mm = m & (pe >= t)
            print(f"rejected p_cal [{lo},{hi}) p_ce>={t}: {mm.sum():,} pairs, true {y[mm].mean() if mm.any() else 0:.3f}")
    for t in (1e-3, 1e-2, 0.1):
        mm = accd & (pe <= t)
        print(f"accepted p_ce<={t}: {mm.sum():,} pairs, true {y[mm].mean() if mm.any() else 0:.3f}")

    rules = {}
    for A in (0.99, 0.995, 0.999):
        for B in (0.0, 0.2, 0.4, 0.5, 0.6):
            s = pc.copy()
            s[(pe >= A) & (pc >= B)] = np.maximum(s[(pe >= A) & (pc >= B)], THR + 1e-4)
            rules[f"over A{A} B{B}"] = s
    for V in (1e-3, 1e-2, 0.05):
        for C in (0.8, 0.9, 0.95, 1.01):
            s = pc.copy()
            s[(pe <= V) & (pc < C)] = 0.0
            rules[f"veto V{V} C{C}"] = s
    lc, le = logit(pc), logit(pe)
    for w in (0.1, 0.2, 0.3):
        z = 1 / (1 + np.exp(-((1 - w) * lc + w * le)))
        for thr in (0.6, 0.65, 0.7, 0.73, 0.76, 0.8):
            rules[f"blend w{w} t{thr}"] = (z, thr)
    fs = {}
    for name, s in rules.items():
        f, _ = score(*s) if isinstance(s, tuple) else score(s)
        fs[name] = f
        print(f"{name}: {f.mean():.5f} ({f.mean() - base.mean():+.5f})", flush=True)
    half = (q % 2).astype(bool)
    out = {"base_f05": float(base.mean()), "rules": {k: float(v.mean() - base.mean()) for k, v in fs.items()}}
    cf = 0.0
    for h in (True, False):
        best = max(fs, key=lambda k: fs[k][half == h].mean() - base[half == h].mean())
        cf += (fs[best][half != h] - base[half != h]).sum()
        out[f"chosen_half_{int(h)}"] = best
        print(f"half {h}: chose {best}")
    out["crossfit_delta"] = float(cf / q.size)
    top = max(fs, key=lambda k: fs[k].mean())
    out["best"] = {"rule": top, **match.paired_delta(fs[top], base, n_boot=500)}
    print(f"cross-fitted delta {cf / q.size:+.5f}; best {top} {out['best']['delta']:+.5f} {out['best']['ci95']}")
    io_utils.write_metrics("c0_override", out, q.size, int(Tn.sum()))


if __name__ == "__main__":
    main()
