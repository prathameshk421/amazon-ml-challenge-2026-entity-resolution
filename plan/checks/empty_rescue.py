"""ER1: for S1 left with NO accepted pool record after the S16-equivalent pipeline (M027 at 0.73 + DRACE c0 >= 0.995),
admit that S1's best DRACE pair (top-1 only) if c0 >= T2 (< 0.995). Cross-fitted by S1 half. MATCH_TRAIN, labels for
scoring only. Writes artifacts/metrics/empty_rescue.json.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from gate2_eval import f05

T2S = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99]


def main():
    q = np.load(match.runs_dir("M027") / "queries.npy")
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas(); gt = gt[gt.s1_idx.isin(q)]
    Tn = gt.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    o = pq.read_table(match.runs_dir("M027") / "oof.parquet").to_pandas()
    k = match.apply_decision({"rule": "threshold", "threshold": 0.73, "one_owner": True},
                             o.s1_idx.to_numpy(), o.pool_idx.to_numpy(), o.p_cal.to_numpy())
    acc = o[k]
    cols = ["s1_idx", "pool_idx", "p", "label"]
    g = pd.read_parquet("work/retrieval/gate2_meta.parquet"); g["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
    d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet"); d["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2DS_TRAIN.parquet").column("p_ce").to_numpy()
    dr = pd.concat([g[cols], d[cols]]).drop_duplicates(["s1_idx", "pool_idx"])
    dr = dr[~dr.pool_idx.isin(set(acc.pool_idx.tolist()))].sort_values("p", ascending=False)
    add = dr[dr.p >= 0.995].drop_duplicates("pool_idx")
    P0 = (acc.groupby("s1_idx").size().reindex(q, fill_value=0) + add.groupby("s1_idx").size().reindex(q, fill_value=0)).to_numpy().astype(float)
    TP0 = (acc[acc.y].groupby("s1_idx").size().reindex(q, fill_value=0)
           + add[add.label == 1].groupby("s1_idx").size().reindex(q, fill_value=0)).to_numpy().astype(float)
    base = f05(Tn, P0, TP0)
    empty = pd.Series(P0 == 0, index=q)
    taken = set(add.pool_idx.tolist())
    cand = dr[(dr.p < 0.995) & empty.reindex(dr.s1_idx).to_numpy() & ~dr.pool_idx.isin(taken)]
    cand = cand.drop_duplicates("s1_idx")  # best pair per empty S1 (sorted by p)
    print(f"S16-equiv {base.mean():.5f}; empty S1 {int((P0 == 0).sum()):,} (true-empty {int(((P0 == 0) & (Tn == 0)).sum()):,}); "
          f"their best sub-0.995 DRACE pair: {len(cand):,}")
    half = pd.Series((q % 2).astype(bool), index=q)
    fs, out = {}, {"base": float(base.mean()), "rows": []}
    for t2 in T2S:
        c = cand[cand.p >= t2].drop_duplicates("pool_idx")
        ad = c.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        tp = c[c.label == 1].groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        fs[t2] = f05(Tn, P0 + ad, TP0 + tp)
        dl = match.paired_delta(fs[t2], base, n_boot=500)
        out["rows"].append({"t2": t2, "added": len(c), "true": int(c.label.sum()), "delta": dl})
        print(f"T2 {t2}: add {len(c):,} ({int(c.label.sum()):,} true) Δ {dl['delta']:+.5f} [{dl['ci95'][0]:+.5f}, {dl['ci95'][1]:+.5f}]", flush=True)
    hv = half.to_numpy(); cf = 0.0
    for h in (True, False):
        best = max(T2S + [None], key=lambda t: (base if t is None else fs[t])[hv == h].mean())
        f = base if best is None else fs[best]
        cf += (f[hv != h] - base[hv != h]).sum(); print(f"half {h}: chose {best}")
    out["crossfit"] = float(cf / q.size); print(f"cross-fitted Δ {cf / q.size:+.5f}")
    io_utils.write_metrics("empty_rescue", out, q.size, int(Tn.sum()))


if __name__ == "__main__":
    main()
