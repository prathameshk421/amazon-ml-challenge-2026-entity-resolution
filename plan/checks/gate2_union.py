"""R1 gate 2: does the c0-distilled e5 (S1 query) add F0.5 on top of current DRACE? MATCH_TRAIN, labels for scoring only.

  PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/gate2_union.py

Base M027 OOF decision. Additions: pairs with c0 >= T (0.995, the test rule) and pool records M027 does not own;
one-owner across all added pairs (highest p_ce wins).
  A  current DRACE: untrained e5, S1 rank <= 5 or seed rank <= 2           (gate2_meta + GATE2_TRAIN scores)
  B  A + distilled e5 S1 rank <= 2                                          (gate2ds_train_meta + GATE2DS_TRAIN scores)
  C  A + distilled e5 S1 rank <= 5
Writes artifacts/metrics/gate2_union.json.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from gate2_eval import f05

T = 0.995


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
    base = f05(Tn, P0, TP0)
    owned = set(acc.pool_idx.tolist())

    a = pd.read_parquet("work/retrieval/gate2_meta.parquet")
    a["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
    a = a[(a.s1_rank <= 5) | (a.seed_rank <= 2)]
    d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet")
    sc = pq.read_table("work/ce/ce_rec_c0/GATE2DS_TRAIN.parquet").to_pandas()
    assert np.array_equal(sc.pool_idx.to_numpy().astype(np.int64), d.pool_idx.to_numpy().astype(np.int64))
    d["p"] = sc.p_ce.to_numpy()

    def run(parts):
        m = pd.concat(parts, ignore_index=True).drop_duplicates(["s1_idx", "pool_idx"])
        n_cand = len(m) / q.size
        m = m[(m.p >= T) & ~m.pool_idx.isin(owned)].sort_values("p", ascending=False).drop_duplicates("pool_idx")
        ad = m.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        tp = m[m.label == 1].groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        return f05(Tn, P0 + ad, TP0 + tp), n_cand, len(m), int(m.label.sum())
    cols = ["s1_idx", "pool_idx", "p", "label"]
    fa, ca, na, ta = run([a[cols]])
    out = {"base_f05": float(base.mean()), "A": {"vs_M027": match.paired_delta(fa, base, n_boot=1000),
                                                "new_cand_per_s1": ca, "added": na, "added_true": ta}}
    print(f"base {base.mean():.5f}; A: +{ca:.2f} cand/S1, added {na:,} ({ta:,} true), vs M027 {fa.mean() - base.mean():+.5f}")
    for name, kmax in (("B", 2), ("C", 5)):
        f, c, n, t = run([a[cols], d[d.s1_rank <= kmax][cols]])
        vs = match.paired_delta(f, fa, n_boot=2000)
        s1c = pq.read_table("work/records/train_s1.parquet", columns=["s1_idx", "country_raw"]).to_pandas() \
            .set_index("s1_idx").country_raw.fillna("NA").str.strip().reindex(q).to_numpy()
        seg = {c_: match.paired_delta(f[s1c == c_], fa[s1c == c_], n_boot=1000) for c_ in ("US", "India")}
        seg["singleton"] = match.paired_delta(f[Tn == 0], fa[Tn == 0], n_boot=1000)
        out[name] = {"distilled_s1_rank_max": kmax, "new_cand_per_s1": c, "added": n, "added_true": t,
                     "vs_A": vs, "segments_vs_A": seg}
        print(f"{name}: +{c:.2f} cand/S1, added {n:,} ({t:,} true); vs A {vs['delta']:+.5f} "
              f"[{vs['ci95'][0]:+.5f}, {vs['ci95'][1]:+.5f}]; " +
              "; ".join(f"{k_} {v['delta']:+.5f}" for k_, v in seg.items()))
    io_utils.write_metrics("gate2_union", out, q.size, int(Tn.sum()))


if __name__ == "__main__":
    main()
