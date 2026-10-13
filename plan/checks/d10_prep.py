"""D10: train pairs for distilled-e5 S1 ranks 6-10 (MATCH_TRAIN) not in the candidate lists and not already scored
(GATE2_TRAIN, GATE2DS_TRAIN). Writes work/ce/pairs_GATE2D10_TRAIN.parquet + work/retrieval/gate2d10_train_meta.parquet.
Labels only for scoring the gate.

  PYTHONPATH=submission/code/business_entity_resolution/src:ce:plan/checks .venv/bin/python plan/checks/d10_prep.py
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gate2_prep import key
from prepare_pairs import record_text

SET = "GATE2D10_TRAIN"


def main():
    top = pq.read_table("work/retrieval/gate1_distil_topk.parquet").to_pandas()
    q = pq.read_table("work/retrieval/gate1_queries.parquet", columns=["qid", "kind"]).to_pandas()
    top = top[top.qid.isin(q.qid[q.kind == "s1"]) & (top["rank"] >= 6) & (top["rank"] <= 10)]
    m = top.rename(columns={"qid": "s1_idx", "rank": "s1_rank", "cos": "cos_s1"})[["s1_idx", "pool_idx", "s1_rank", "cos_s1"]]
    m["s1_idx"] = m.s1_idx.astype(np.int64)
    cand = pq.read_table("work/candidates/MATCH_TRAIN/part-00000.parquet", columns=["s1_idx", "pool_idx"]).to_pandas()
    old = pd.concat([pd.read_parquet(f"work/retrieval/{f}")[["s1_idx", "pool_idx"]]
                     for f in ("gate2_meta.parquet", "gate2ds_train_meta.parquet")])
    kk = key(m.s1_idx, m.pool_idx)
    m = m[~np.isin(kk, key(cand.s1_idx, cand.pool_idx)) & ~np.isin(kk, key(old.s1_idx, old.pool_idx))]
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas()
    m = m.sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
    m["label"] = np.isin(key(m.s1_idx, m.pool_idx), key(gt.s1_idx, gt.pool_idx)).astype(np.int8)
    s1_txt, pool_txt = record_text("train", "s1"), record_text("train", "pool")
    s1, pool = m.s1_idx.to_numpy().astype(np.int32), m.pool_idx.to_numpy().astype(np.uint32)
    pq.write_table(pa.table({"s1_idx": pa.array(s1), "pool_idx": pa.array(pool),
                             "rank": pa.array(np.zeros(len(m), dtype=np.int16)),
                             "text_a": pa.array(s1_txt[s1].tolist(), pa.string()),
                             "text_b": pa.array(pool_txt[pool].tolist(), pa.string()),
                             "label": pa.array(m.label.to_numpy())}),
                   f"work/ce/pairs_{SET}.parquet", compression="zstd")
    m.to_parquet(f"work/retrieval/{SET.lower()}_meta.parquet")
    print(f"{SET}: {len(m):,} new pairs, {int(m.label.sum()):,} true ({m.label.mean():.3%})")


if __name__ == "__main__":
    main()
