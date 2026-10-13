"""D10 gate: does widening the distilled-e5 S1 query from top-5 (S13) to top-10 add F0.5? MATCH_TRAIN, labels for
scoring only. Base M027 OOF decision (threshold 0.73, one-owner); additions at c0 >= T, pool records the base owns
excluded, one-owner across additions (highest p wins) — the test merge rule.
  C    S13: untrained e5 S1@5 | seed@2 + distilled S1@5
  C10  C + distilled S1 ranks 6-10 (GATE2D10_TRAIN scores, plus gate2_meta pairs already scored)
Also C10 restricted to S1 with >= n S13 acceptances (adaptive depth). Writes artifacts/metrics/d10_eval.json.

  PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/d10_eval.py
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from gate2_eval import f05, key

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
    owned = set(acc.pool_idx.tolist())
    cols = ["s1_idx", "pool_idx", "p", "label"]
    g = pd.read_parquet("work/retrieval/gate2_meta.parquet")
    g["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
    a = g[(g.s1_rank <= 5) | (g.seed_rank <= 2)]
    d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet")
    d["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2DS_TRAIN.parquet").column("p_ce").to_numpy()
    e = pd.read_parquet("work/retrieval/gate2d10_train_meta.parquet")
    sc = pq.read_table("work/ce/ce_rec_c0/GATE2D10_TRAIN.parquet").to_pandas()
    assert np.array_equal(sc.pool_idx.to_numpy().astype(np.int64), e.pool_idx.to_numpy().astype(np.int64))
    e["p"] = sc.p_ce.to_numpy()
    top = pq.read_table("work/retrieval/gate1_distil_topk.parquet").to_pandas()
    top = top[(top.qid < 10 ** 9) & (top["rank"] >= 6) & (top["rank"] <= 10)]
    g10 = g[np.isin(key(g.s1_idx, g.pool_idx), key(top.qid, top.pool_idx))]
    e = pd.concat([e[cols], g10[cols]]).drop_duplicates(["s1_idx", "pool_idx"])
    print(f"distilled ranks 6-10 scored pairs {len(e):,}, true {int(e.label.sum()):,}; p>=T {int((e.p >= T).sum()):,} "
          f"(true {int(e.label[e.p >= T].sum()):,})")

    def run(parts):
        m = pd.concat(parts, ignore_index=True).drop_duplicates(["s1_idx", "pool_idx"])
        n_cand = len(m) / q.size
        m = m[(m.p >= T) & ~m.pool_idx.isin(owned)].sort_values("p", ascending=False).drop_duplicates("pool_idx")
        ad = m.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        tp = m[m.label == 1].groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        return f05(Tn, P0 + ad, TP0 + tp), n_cand, len(m), int(m.label.sum()), ad
    fc, cc, nc, tc, adc = run([a[cols], d[cols]])
    base = f05(Tn, P0, TP0)
    s1c = pq.read_table("work/records/train_s1.parquet", columns=["s1_idx", "country_raw"]).to_pandas() \
        .set_index("s1_idx").country_raw.fillna("NA").str.strip().reindex(q).to_numpy()
    out = {"base_f05": float(base.mean()), "C": {"f05": float(fc.mean()), "cand_per_s1": cc, "added": nc, "added_true": tc}}
    print(f"M027 {base.mean():.5f}; C (S13) {fc.mean():.5f} +{cc:.2f} cand/S1 added {nc:,} ({tc:,} true)")
    nacc = P0 + adc
    for minacc in (0, 2, 3, 4, 5):
        ee = e[pd.Series(nacc, index=q).reindex(e.s1_idx).to_numpy() >= minacc]
        f, c, n, t, _ = run([a[cols], d[cols], ee])
        vs = match.paired_delta(f, fc, n_boot=2000)
        seg = {c_: match.paired_delta(f[s1c == c_], fc[s1c == c_], n_boot=500)["delta"] for c_ in ("US", "India")}
        seg["singleton"] = match.paired_delta(f[Tn == 0], fc[Tn == 0], n_boot=500)["delta"]
        out[f"C10_min{minacc}"] = {"f05": float(f.mean()), "cand_per_s1": c, "added": n, "added_true": t, "vs_C": vs,
                                   "segments_vs_C": seg, "pairs_scored": int(len(ee))}
        print(f"C10 minacc {minacc}: +{c:.2f} cand/S1 added {n:,} ({t:,} true); vs C {vs['delta']:+.5f} "
              f"[{vs['ci95'][0]:+.5f}, {vs['ci95'][1]:+.5f}]; " + "; ".join(f"{k_} {v:+.5f}" for k_, v in seg.items()),
              flush=True)
    io_utils.write_metrics("d10_eval", out, q.size, int(Tn.sum()))


if __name__ == "__main__":
    main()
