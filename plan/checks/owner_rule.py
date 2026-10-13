"""OW1: who gets a pool record claimed by several S1 among DRACE additions (all c0 >= 0.995)? Live rule: highest c0.
Variants: fewest base matches first (tie: c0); most base matches first. S16/S17-equivalent on MATCH_TRAIN (M027 at 0.74,
channels seed<=5, distilled S1@5, distilled ranks 6-7). Labels for scoring only. Writes artifacts/metrics/owner_rule.json.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from gate2_eval import f05


def main():
    q = np.load(match.runs_dir("M027") / "queries.npy")
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas(); gt = gt[gt.s1_idx.isin(q)]
    Tn = gt.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    o = pq.read_table(match.runs_dir("M027") / "oof.parquet").to_pandas()
    k = match.apply_decision({"rule": "threshold", "threshold": 0.74, "one_owner": True},
                             o.s1_idx.to_numpy(), o.pool_idx.to_numpy(), o.p_cal.to_numpy())
    acc = o[k]
    nb = acc.groupby("s1_idx").size()
    P0 = nb.reindex(q, fill_value=0).to_numpy().astype(float)
    TP0 = acc[acc.y].groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    cols = ["s1_idx", "pool_idx", "p", "label"]
    g = pd.read_parquet("work/retrieval/gate2_meta.parquet"); g["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
    d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet"); d["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2DS_TRAIN.parquet").column("p_ce").to_numpy()
    e = pd.read_parquet("work/retrieval/gate2d10_train_meta.parquet"); e["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2D10_TRAIN.parquet").column("p_ce").to_numpy()
    dr = pd.concat([g[cols], d[cols], e[e.s1_rank <= 7][cols]]).drop_duplicates(["s1_idx", "pool_idx"])
    dr = dr[(dr.p >= 0.995) & ~dr.pool_idx.isin(set(acc.pool_idx.tolist()))].copy()
    dr["nb"] = nb.reindex(dr.s1_idx).fillna(0).to_numpy()
    multi = dr.pool_idx.duplicated(keep=False)
    print(f"DRACE pairs {len(dr):,}; pool records claimed by > 1 S1: {dr[multi].pool_idx.nunique():,} ({int(multi.sum()):,} pairs, "
          f"{int(dr[multi].label.sum()):,} true)")

    def run(order):
        m = dr.sort_values(order[0], ascending=order[1]).drop_duplicates("pool_idx")
        ad = m.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        tp = m[m.label == 1].groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        return f05(Tn, P0 + ad, TP0 + tp), int(m.label.sum())
    live, tl = run((["p"], [False]))
    out = {"live": float(live.mean())}
    print(f"live (highest c0): {live.mean():.5f} ({tl:,} true added)")
    for name, order in (("fewest_first", (["nb", "p"], [True, False])), ("most_first", (["nb", "p"], [False, False]))):
        f, t = run(order)
        dl = match.paired_delta(f, live, n_boot=1000)
        out[name] = {"f05": float(f.mean()), "true_added": t, "vs_live": dl}
        print(f"{name}: {f.mean():.5f} ({t:,} true) vs live {dl['delta']:+.5f} [{dl['ci95'][0]:+.5f}, {dl['ci95'][1]:+.5f}]")
    io_utils.write_metrics("owner_rule", out, q.size, int(Tn.sum()))


if __name__ == "__main__":
    main()
