"""Where does the S10-equivalent system (M027 OOF + DRACE gate-2 additions at c0 >= 0.995) still lose F0.5 on MATCH_TRAIN?

  PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/loss_s10.py

Buckets (pair level) and the oracle F0.5 gain of fixing only that bucket:
  FP_lgb      wrong pair accepted by LightGBM                 (oracle: remove it)
  FP_drace    wrong pair added by DRACE                        (remove it)
  FN_list     true pair in the candidate list, rejected        (accept it)
  FN_drace    true pair DRACE retrieved (S1@5/seed@5) but c0 < 0.995 or one-owner blocked (accept it)
  FN_retr50   true pair not in list, not in DRACE@5, but in gate-1 top-50 (S1 or seed)   (accept it)
  FN_never    true pair found by nothing                        (accept it)
Also: per bucket, share with empty pool address, name token-set, country. Writes artifacts/metrics/loss_s10.json.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from gate2_eval import f05, key

SEED_BASE = 10 ** 9


def main():
    q = np.load(match.runs_dir("M027") / "queries.npy")
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas()
    gt = gt[gt.s1_idx.isin(q)]
    T = gt.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    oof = pq.read_table(match.runs_dir("M027") / "oof.parquet").to_pandas()
    pred = match.apply_decision({"rule": "threshold", "threshold": 0.73, "one_owner": True},
                                oof.s1_idx.to_numpy(), oof.pool_idx.to_numpy(), oof.p_cal.to_numpy())
    acc = oof.loc[pred, ["s1_idx", "pool_idx", "y"]].assign(src="lgb")
    owned = set(acc.pool_idx.tolist())
    m = pd.read_parquet("work/retrieval/gate2_meta.parquet")
    m["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
    add = m[~m.pool_idx.isin(owned) & (m.p >= 0.995)].sort_values("p", ascending=False).drop_duplicates("pool_idx")
    acc = pd.concat([acc, pd.DataFrame({"s1_idx": add.s1_idx, "pool_idx": add.pool_idx, "y": add.label.astype(bool),
                                        "src": "drace"})], ignore_index=True)
    acc["k"] = key(acc.s1_idx, acc.pool_idx)
    P = acc.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    TP = acc[acc.y].groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    base = f05(T, P, TP)

    gk = key(gt.s1_idx, gt.pool_idx)
    in_list = np.isin(gk, key(oof.s1_idx, oof.pool_idx))
    accepted = np.isin(gk, acc.k.to_numpy())
    in_drace5 = np.isin(gk, key(m.s1_idx, m.pool_idx))
    top = pq.read_table("work/retrieval/gate1_topk.parquet").to_pandas()
    seeds = pd.read_parquet("work/retrieval/gate1_seeds.parquet")
    t50 = np.concatenate([key(top.qid[top.qid < SEED_BASE], top.pool_idx[top.qid < SEED_BASE]),
                          key(*top[top.qid >= SEED_BASE].merge(seeds[["qid", "s1_idx"]], on="qid")[["s1_idx", "pool_idx"]].T.values)])
    in_top50 = np.isin(gk, t50)
    fn = ~accepted
    bucket = np.select([fn & in_list, fn & ~in_list & in_drace5, fn & ~in_list & ~in_drace5 & in_top50, fn],
                       ["FN_list", "FN_drace", "FN_retr50", "FN_never"], "TP")
    g = gt.assign(bucket=bucket)
    fp = acc[~acc.y]

    pool = pq.read_table("work/records/train_pool.parquet", columns=["pool_idx", "addr_raw"]).to_pandas()
    empty = pool.addr_raw.fillna("").str.strip().eq("").to_numpy()
    s1c = pq.read_table("work/records/train_s1.parquet", columns=["s1_idx", "country_raw"]).to_pandas() \
        .set_index("s1_idx").country_raw.fillna("NA").str.strip()
    out = {"f05": float(base.mean()), "loss": float((1 - base).sum() / q.size), "ceiling_all_fixed": 1.0, "buckets": {}}
    print(f"S10-equivalent on 180K MATCH_TRAIN S1: F0.5 {base.mean():.5f} (loss {1 - base.mean():.5f}); "
          f"accepted {len(acc):,} ({(acc.src == 'drace').sum():,} DRACE)")

    def report(name, n, gain, sub_pool, sub_s1):
        d = {"pairs": int(n), "oracle_gain": float(gain),
             "empty_pool_addr": float(empty[sub_pool].mean()) if len(sub_pool) else 0.0,
             "india_share": float((s1c.reindex(sub_s1).to_numpy() == "India").mean()) if len(sub_s1) else 0.0}
        out["buckets"][name] = d
        print(f"  {name:10s} pairs {n:>7,}  oracle gain {gain:+.5f}  empty pool addr {d['empty_pool_addr']:.0%}  "
              f"India {d['india_share']:.0%}")
    for src in ("lgb", "drace"):
        b = fp[fp.src == src]
        rm = b.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        report(f"FP_{src}", len(b), f05(T, P - rm, TP).mean() - base.mean(), b.pool_idx.to_numpy(), b.s1_idx.to_numpy())
    for bk in ("FN_list", "FN_drace", "FN_retr50", "FN_never"):
        b = g[g.bucket == bk]
        ad = b.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        report(bk, len(b), f05(T, P + ad, TP + ad).mean() - base.mean(), b.pool_idx.to_numpy(), b.s1_idx.to_numpy())
    io_utils.write_metrics("loss_s10", out, q.size, int(T.sum()))


if __name__ == "__main__":
    main()
