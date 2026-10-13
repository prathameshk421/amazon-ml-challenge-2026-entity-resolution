"""DS2: train pairs for distilled-e5 seed expansion ranks 1-2 (gate2d_train_meta, seeds = M027 accepted records), minus
pairs in the lists or already c0-scored (GATE2_TRAIN, GATE2DS_TRAIN, GATE2D10_TRAIN). Writes work/ce/pairs_GATE2DSEED_TRAIN.parquet
+ work/retrieval/gate2dseed_train_meta.parquet. Labels only for scoring the gate.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gate2_prep import key
from prepare_pairs import record_text


def main():
    m = pd.read_parquet("work/retrieval/gate2d_train_meta.parquet")
    m = m[m.seed_rank <= 2][["s1_idx", "pool_idx", "seed_rank", "cos_seed", "label"]]
    cand = pq.read_table("work/candidates/MATCH_TRAIN/part-00000.parquet", columns=["s1_idx", "pool_idx"]).to_pandas()
    old = pd.concat([pd.read_parquet(f"work/retrieval/{f}")[["s1_idx", "pool_idx"]] for f in
                     ("gate2_meta.parquet", "gate2ds_train_meta.parquet", "gate2d10_train_meta.parquet")])
    kk = key(m.s1_idx, m.pool_idx)
    m = m[~np.isin(kk, key(cand.s1_idx, cand.pool_idx)) & ~np.isin(kk, key(old.s1_idx, old.pool_idx))]
    m = m.sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
    s1t, pt = record_text("train", "s1"), record_text("train", "pool")
    s1, pool = m.s1_idx.to_numpy().astype(np.int32), m.pool_idx.to_numpy().astype(np.uint32)
    pq.write_table(pa.table({"s1_idx": pa.array(s1), "pool_idx": pa.array(pool), "rank": pa.array(np.zeros(len(m), dtype=np.int16)),
                             "text_a": pa.array(s1t[s1].tolist(), pa.string()), "text_b": pa.array(pt[pool].tolist(), pa.string()),
                             "label": pa.array(m.label.to_numpy().astype(np.int8))}),
                   "work/ce/pairs_GATE2DSEED_TRAIN.parquet", compression="zstd")
    m.to_parquet("work/retrieval/gate2dseed_train_meta.parquet")
    print(f"GATE2DSEED_TRAIN: {len(m):,} pairs, {int(m.label.sum()):,} true")


if __name__ == "__main__":
    main()
