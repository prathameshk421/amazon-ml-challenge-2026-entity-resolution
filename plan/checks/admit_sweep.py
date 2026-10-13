"""A1: c0 admission threshold sweep for DRACE additions (S13 config = untrained e5 S1@5/seed@2 + distilled e5 S1@5),
and a breakdown of true pairs missed by every channel (capacity: S1 with > 5 true pool records). MATCH_TRAIN, labels
for scoring only; cross-fitted threshold choice by S1 half. Writes artifacts/metrics/admit_sweep.json.

  PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/admit_sweep.py
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from gate2_eval import f05, key

TS = [0.9, 0.95, 0.97, 0.98, 0.99, 0.993, 0.995, 0.997, 0.998, 0.999]


def main():
    q = np.load(match.runs_dir("M027") / "queries.npy")
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas()
    gt = gt[gt.s1_idx.isin(q)]
    Tn = gt.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    o = pq.read_table(match.runs_dir("M027") / "oof.parquet").to_pandas()
    k = match.apply_decision({"rule": "threshold", "threshold": 0.73, "one_owner": True},
                             o.s1_idx.to_numpy(), o.pool_idx.to_numpy(), o.p_cal.to_numpy())
    acc = o[k]
    P0 = acc.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    TP0 = acc[acc.y].groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    owned = set(acc.pool_idx.tolist())
    a = pd.read_parquet("work/retrieval/gate2_meta.parquet")
    a["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
    a = a[(a.s1_rank <= 5) | (a.seed_rank <= 2)]
    d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet")
    d["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2DS_TRAIN.parquet").column("p_ce").to_numpy()
    cols = ["s1_idx", "pool_idx", "p", "label"]
    m = pd.concat([a[cols], d[cols]]).drop_duplicates(["s1_idx", "pool_idx"])
    m = m[~m.pool_idx.isin(owned)].sort_values("p", ascending=False)
    pool = pq.read_table("work/records/train_pool.parquet", columns=["addr_raw"]).column(0).to_pandas()
    empty = pool.fillna("").astype(str).str.strip().eq("").to_numpy()
    half = (q % 2).astype(bool)
    out = {"sweep": [], "cand_true_below": {}}
    fs = {}
    for t in TS:
        mm = m[m.p >= t].drop_duplicates("pool_idx")
        ad = mm.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        tp = mm[mm.label == 1].groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        f = f05(Tn, P0 + ad, TP0 + tp)
        fs[t] = f
        e = empty[mm.pool_idx.to_numpy()]
        out["sweep"].append({"T": t, "f05": float(f.mean()), "added": len(mm), "added_true": int(mm.label.sum()),
                             "empty_added": int(e.sum()), "empty_true": int(mm.label.to_numpy()[e].sum())})
        print(f"T {t:.3f}: F0.5 {f.mean():.5f} added {len(mm):,} true {int(mm.label.sum()):,} "
              f"(name-only {e.sum():,}, true {int(mm.label.to_numpy()[e].sum()):,})", flush=True)
    ref = fs[0.995]
    for t in TS:
        dlt = match.paired_delta(fs[t], ref, n_boot=500)
        out["sweep"][TS.index(t)]["vs_0995"] = dlt
        print(f"  T {t}: vs 0.995 {dlt['delta']:+.5f} [{dlt['ci95'][0]:+.5f}, {dlt['ci95'][1]:+.5f}]")
    # cross-fit: choose T on one half, score on the other
    cf = 0.0
    for h in (True, False):
        best = max(TS, key=lambda t: fs[t][half == h].mean())
        cf += (fs[best][half != h] - ref[half != h]).sum()
        print(f"  half {h}: chose {best}")
    out["crossfit_vs_0995"] = float(cf / q.size)
    print(f"cross-fitted vs 0.995: {cf / q.size:+.5f}")
    # missed-by-all breakdown
    cand = pq.read_table("work/candidates/MATCH_TRAIN/part-00000.parquet", columns=["s1_idx", "pool_idx"]).to_pandas()
    known = np.unique(np.concatenate([key(cand.s1_idx, cand.pool_idx), key(a.s1_idx, a.pool_idx),
                                      key(d.s1_idx, d.pool_idx)]))
    gk = key(gt.s1_idx, gt.pool_idx)
    miss = gt[~np.isin(gk, known)].copy()
    miss["T"] = pd.Series(Tn, index=q).reindex(miss.s1_idx).to_numpy()
    for lo, hi in ((1, 3), (4, 5), (6, 10), (11, 10 ** 6)):
        s = miss[(miss["T"] >= lo) & (miss["T"] <= hi)]
        print(f"missed with S1 true-count {lo}-{hi}: {len(s):,} (name-only {empty[s.pool_idx].mean():.0%})")
    out["missed_by_T"] = miss["T"].describe().to_dict()
    out["missed_T_gt5_share"] = float((miss["T"] > 5).mean())
    print(f"missed pairs: {len(miss):,}; share from S1 with > 5 true: {(miss['T'] > 5).mean():.1%}; "
          f"all true pairs from S1 with > 5 true: {(Tn[Tn > 5].sum() / Tn.sum()):.1%}")
    io_utils.write_metrics("admit_sweep", out, q.size, int(Tn.sum()))


if __name__ == "__main__":
    main()
