"""I1 train gate: profile-query top-5 pairs (iter_train_topk) not reached by any S17 channel (lists, untrained S1@5/seed@5,
distilled S1@5, distilled S1 ranks 6-7), pool not owned by the S13-equivalent state. Writes work/ce/pairs_ITER_TRAIN.parquet
+ work/retrieval/iter_gate_train_meta.parquet (labels only for scoring the gate).
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gate2_prep import key
from prepare_pairs import record_text


def main():
    top = pq.read_table("work/retrieval/iter_train_topk.parquet").to_pandas()
    qs = pq.read_table("work/retrieval/iter_train_queries.parquet", columns=["qid", "kind", "s1_idx"]).to_pandas()
    top = top.merge(qs, on="qid"); top = top[(top.kind == "profile") & (top["rank"] <= 5)]
    m = top[["s1_idx", "pool_idx"]].drop_duplicates().astype(np.int64)
    cand = pq.read_table("work/candidates/MATCH_TRAIN/part-00000.parquet", columns=["s1_idx", "pool_idx"]).to_pandas()
    e = pd.read_parquet("work/retrieval/gate2d10_train_meta.parquet")
    old = pd.concat([cand, pd.read_parquet("work/retrieval/gate2_meta.parquet")[["s1_idx", "pool_idx"]],
                     pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet")[["s1_idx", "pool_idx"]],
                     e[e.s1_rank <= 7][["s1_idx", "pool_idx"]]])
    st = pd.read_parquet("work/retrieval/iter_train_state.parquet")
    kk = key(m.s1_idx, m.pool_idx)
    m = m[~np.isin(kk, key(old.s1_idx, old.pool_idx)) & ~m.pool_idx.isin(st.pool_idx)]
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas()
    m = m.sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
    m["label"] = np.isin(key(m.s1_idx, m.pool_idx), key(gt.s1_idx, gt.pool_idx)).astype(np.int8)
    s1t, pt = record_text("train", "s1"), record_text("train", "pool")
    s1, pool = m.s1_idx.to_numpy().astype(np.int32), m.pool_idx.to_numpy().astype(np.uint32)
    pq.write_table(pa.table({"s1_idx": pa.array(s1), "pool_idx": pa.array(pool), "rank": pa.array(np.zeros(len(m), dtype=np.int16)),
                             "text_a": pa.array(s1t[s1].tolist(), pa.string()), "text_b": pa.array(pt[pool].tolist(), pa.string()),
                             "label": pa.array(m.label.to_numpy())}), "work/ce/pairs_ITER_TRAIN.parquet", compression="zstd")
    m.to_parquet("work/retrieval/iter_gate_train_meta.parquet")
    print(f"ITER_TRAIN: {len(m):,} pairs, {int(m.label.sum()):,} true")


if __name__ == "__main__":
    main()
