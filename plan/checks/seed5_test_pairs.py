"""SD5 test pairs: untrained-e5 seed-expansion ranks 3-5 from the existing drace_test_topk (no new retrieval), minus pairs
in the candidate lists, pairs already scored (DRACE/DRACD/DRACD5), and pool records S13 already assigns (base or DRACE).
Writes work/ce/pairs_SEED5_TEST.parquet + work/retrieval/seed5_test_meta.parquet (cos, seed_rank for prioritising).

  PYTHONPATH=submission/code/business_entity_resolution/src:ce:plan/checks .venv/bin/python plan/checks/seed5_test_pairs.py \
      [--s13 submission/output_v13/matching_results.tsv]
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from drace_test import R, SEED_BASE, key, read_results, test_candidates
from prepare_pairs import record_text


def main(s13_path):
    top = pq.read_table(R / "drace_test_topk.parquet").to_pandas()
    top = top[(top.qid >= SEED_BASE) & (top["rank"] >= 3)]
    seeds = pd.read_parquet(R / "drace_test_seeds.parquet")
    m = top.merge(seeds[["qid", "s1_idx"]], on="qid")
    m = m.groupby(["s1_idx", "pool_idx"], as_index=False).agg(cos=("cos", "max"), seed_rank=("rank", "min"))
    cand = test_candidates()
    old = pd.concat([pd.read_parquet(R / f)[["s1_idx", "pool_idx"]] for f in
                     ("drace_test_meta.parquet", "dracd_test_meta.parquet", "dracd5_test_meta.parquet")])
    s13, _ = read_results(s13_path)
    kk = key(m.s1_idx, m.pool_idx)
    n0 = len(m)
    m = m[~np.isin(kk, key(cand.s1_idx, cand.pool_idx)) & ~np.isin(kk, key(old.s1_idx, old.pool_idx))
          & ~m.pool_idx.isin(s13.pool_idx)]
    m = m.sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
    s1_txt, pool_txt = record_text("test", "s1"), record_text("test", "pool")
    s1, pool = m.s1_idx.to_numpy().astype(np.int32), m.pool_idx.to_numpy().astype(np.uint32)
    pq.write_table(pa.table({"s1_idx": pa.array(s1), "pool_idx": pa.array(pool),
                             "rank": pa.array(np.zeros(len(m), dtype=np.int16)),
                             "text_a": pa.array(s1_txt[s1].tolist(), pa.string()),
                             "text_b": pa.array(pool_txt[pool].tolist(), pa.string())}),
                   "work/ce/pairs_SEED5_TEST.parquet", compression="zstd")
    m.to_parquet(R / "seed5_test_meta.parquet")
    print(f"seed ranks 3-5: {n0:,} (S1, pool) pairs -> {len(m):,} new to score; by seed_rank "
          f"{m.seed_rank.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s13", default="submission/output_v13/matching_results.tsv",
                    help="S13 matching_results.tsv (its assigned pool records are excluded)")
    main(ap.parse_args().s13)
