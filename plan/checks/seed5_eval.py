"""SD5: S13 + untrained-e5 seed-expansion ranks 3-5 on MATCH_TRAIN; pairs already
c0-scored, no new scoring. Same rule as d10_eval (M027 base, c0 >= 0.995, one-owner). Writes artifacts/metrics/seed5_eval.json.

  PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/seed5_eval.py
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
    owned = set(acc.pool_idx.tolist())
    cols = ["s1_idx", "pool_idx", "p", "label"]
    g = pd.read_parquet("work/retrieval/gate2_meta.parquet")
    g["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
    d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet")
    d["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2DS_TRAIN.parquet").column("p_ce").to_numpy()

    def run(parts):
        m = pd.concat(parts, ignore_index=True).drop_duplicates(["s1_idx", "pool_idx"])
        n = len(m) / q.size
        m = m[(m.p >= T) & ~m.pool_idx.isin(owned)].sort_values("p", ascending=False).drop_duplicates("pool_idx")
        ad = m.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        tp = m[m.label == 1].groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        return f05(Tn, P0 + ad, TP0 + tp), n, len(m), int(m.label.sum())
    s13 = [g[(g.s1_rank <= 5) | (g.seed_rank <= 2)][cols], d[cols]]
    fc, cc, nc, tc = run(s13)
    out = {"C": {"f05": float(fc.mean()), "cand": cc, "added": nc, "added_true": tc}}
    print(f"S13-equiv {fc.mean():.5f} cand {cc:.2f} added {nc:,} ({tc:,})")
    variants = {"seed3": [g[g.seed_rank <= 3][cols]], "seed5": [g[g.seed_rank <= 5][cols]], }
    e = pd.read_parquet("work/retrieval/gate2d10_train_meta.parquet")
    e["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2D10_TRAIN.parquet").column("p_ce").to_numpy()
    variants["seed5+d10"] = [g[g.seed_rank <= 5][cols], e[cols]]
    variants["seed5+d10_r67"] = [g[g.seed_rank <= 5][cols], e[e.s1_rank <= 7][cols]]
    for name, extra in variants.items():
        f, c, n, t = run(s13 + extra)
        vs = match.paired_delta(f, fc, n_boot=1000)
        out[name] = {"f05": float(f.mean()), "cand": c, "added": n, "added_true": t, "vs_C": vs}
        print(f"{name}: cand {c:.2f} added {n:,} ({t:,} true) vs S13 {vs['delta']:+.5f} [{vs['ci95'][0]:+.5f}, {vs['ci95'][1]:+.5f}]", flush=True)
    io_utils.write_metrics("seed5_eval", out, q.size, int(Tn.sum()))


if __name__ == "__main__":
    main()
