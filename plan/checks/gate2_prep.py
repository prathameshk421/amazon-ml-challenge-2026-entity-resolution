"""Gate 2 inputs: the new (not already in the candidate list) pairs of gate 1's S1@K + seed@K retrieval, in the
cross-encoder's text format, for ce/hf_job/score_job.py (set GATE2_TRAIN).

  PYTHONPATH=submission/code/business_entity_resolution/src:ce .venv/bin/python plan/checks/gate2_prep.py [--k 5]

Writes work/ce/pairs_GATE2_TRAIN.parquet (s1_idx, pool_idx, rank=0, text_a, text_b, label) and
work/retrieval/gate2_meta.parquet (s1_idx, pool_idx, s1_rank, seed_rank, cos, label). Labels are train labels, used
only to fit and score the gate-2 rule on MATCH_TRAIN.
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prepare_pairs import record_text

SEED_BASE = 10 ** 9


def key(a, b):
    return np.asarray(a, dtype=np.int64) << 32 | np.asarray(b, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tag", default="gate1")
    ap.add_argument("--set", default="GATE2_TRAIN")
    ap.add_argument("--skip-scored", default="", help="meta parquet of pairs already scored (excluded)")
    args = ap.parse_args()
    top = pq.read_table(f"work/retrieval/{args.tag}_topk.parquet").to_pandas()
    top = top[top["rank"] <= args.k]
    seeds = pd.read_parquet("work/retrieval/gate1_seeds.parquet")
    a = top[top.qid < SEED_BASE].rename(columns={"qid": "s1_idx", "rank": "s1_rank"})
    a["s1_idx"] = a.s1_idx.astype(np.int64)
    b = top[top.qid >= SEED_BASE].merge(seeds[["qid", "s1_idx"]], on="qid").rename(columns={"rank": "seed_rank"})
    b = b.groupby(["s1_idx", "pool_idx"], as_index=False).agg(seed_rank=("seed_rank", "min"), cos_seed=("cos", "max"))
    a = a.groupby(["s1_idx", "pool_idx"], as_index=False).agg(s1_rank=("s1_rank", "min"), cos_s1=("cos", "max"))
    m = a.merge(b, on=["s1_idx", "pool_idx"], how="outer")
    cand = pq.read_table("work/candidates/MATCH_TRAIN/part-00000.parquet", columns=["s1_idx", "pool_idx"]).to_pandas()
    m = m[~np.isin(key(m.s1_idx, m.pool_idx), key(cand.s1_idx, cand.pool_idx))].reset_index(drop=True)
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas()
    if args.skip_scored:
        old = pd.read_parquet(args.skip_scored)
        m = m[~np.isin(key(m.s1_idx, m.pool_idx), key(old.s1_idx, old.pool_idx))].reset_index(drop=True)
    m["label"] = np.isin(key(m.s1_idx, m.pool_idx), key(gt.s1_idx, gt.pool_idx)).astype(np.int8)
    m = m.sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
    s1_txt, pool_txt = record_text("train", "s1"), record_text("train", "pool")
    s1, pool = m.s1_idx.to_numpy().astype(np.int32), m.pool_idx.to_numpy().astype(np.uint32)
    pq.write_table(pa.table({"s1_idx": pa.array(s1), "pool_idx": pa.array(pool),
                             "rank": pa.array(np.zeros(len(m), dtype=np.int16)),
                             "text_a": pa.array(s1_txt[s1].tolist(), pa.string()),
                             "text_b": pa.array(pool_txt[pool].tolist(), pa.string()),
                             "label": pa.array(m.label.to_numpy())}),
                   f"work/ce/pairs_{args.set}.parquet", compression="zstd")
    m.to_parquet(f"work/retrieval/{args.set.lower()}_meta.parquet")
    print(f"gate2 k={args.k}: {len(m):,} new pairs, {int(m.label.sum()):,} true ({m.label.mean():.3%}); "
          f"from S1 query {m.s1_rank.notna().sum():,}, from seeds {m.seed_rank.notna().sum():,}")


if __name__ == "__main__":
    main()
