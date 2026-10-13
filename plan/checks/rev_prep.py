"""Reverse-name retrieval inputs for ce/hf_job/retrieve_job.py (name-only pool record -> nearest S1 names), and the
features built from its output. Inputs only (no labels).

  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/rev_prep.py prep train|test
  (HF) retrieve_job.py RETR_TAG=rev_<split> RETR_K=20        -> work/retrieval/rev_<split>_topk.parquet
  ... rev_prep.py feats <SET> [...]                          -> work/retrieval/rev_<SET>.parquet (feature-row order)

"pool" file = every S1 of the split, text "query: <S1 name>" (pool_idx column = s1_idx).
"queries"   = every pool record with an EMPTY address that appears in any candidate list of the split's sets
              (train: MATCH_TRAIN, MATCH_VALID, DEV_REST_REST; test: TEST), text "query: <pool name>", qid = pool_idx.
Features (NaN unless the candidate's pool address is empty):
  rv_in20   1 if this S1 is among the pool name's 20 nearest S1 names (same country), else 0
  rv_rank   rank of this S1 in that list (21 when absent)
  rv_cos    cos to this S1 name (NaN when absent)
  rv_top    cos of the nearest S1 name
  rv_gap    rv_top - rv_cos (NaN when absent)
  rv_margin rv_cos - best other S1 cos (NaN when absent)
  rv_nclose S1 names within 0.02 of rv_top (how many owners look equally plausible)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber import io_utils, match

R = Path(os.environ.get("REV_DIR", "work/retrieval"))  # REV_DIR: alternative topk in / features out (ann_rev_check.py)
SETS = {"train": ["MATCH_TRAIN", "MATCH_VALID", "DEV_REST_REST"], "test": ["TEST"]}


def cand(set_name):
    parts = sorted(Path(f"work/candidates/{set_name}").glob("part-*.parquet"))
    return pd.concat([pq.read_table(p, columns=["s1_idx", "pool_idx"]).to_pandas() for p in parts], ignore_index=True)


def cty(s):
    return pd.Series(s).fillna("NA").astype(str).str.strip().tolist()


def prep(split):
    s1 = pq.read_table(f"work/records/{split}_s1.parquet", columns=["s1_idx", "name_raw", "country_raw"]).to_pandas()
    pool = pq.read_table(f"work/records/{split}_pool.parquet", columns=["pool_idx", "name_raw", "addr_raw", "country_raw"]).to_pandas()
    empty = pool.addr_raw.fillna("").astype(str).str.strip().eq("").to_numpy()
    used = np.unique(np.concatenate([cand(s).pool_idx.to_numpy() for s in SETS[split]]))
    q = pool.iloc[used[empty[used]]]
    nm = lambda s: ("query: " + s.fillna("").astype(str).str.strip()).tolist()
    pq.write_table(pa.table({"pool_idx": pa.array(s1.s1_idx.to_numpy().astype(np.uint32)), "country": cty(s1.country_raw),
                             "text": nm(s1.name_raw)}), R / f"rev_{split}_pool.parquet", compression="zstd")
    pq.write_table(pa.table({"qid": pa.array(q.pool_idx.to_numpy().astype(np.int64)), "kind": ["pool"] * len(q),
                             "country": cty(q.country_raw), "text": nm(q.name_raw),
                             "self_pool": pa.array(np.full(len(q), -1, dtype=np.int64))}),
                   R / f"rev_{split}_queries.parquet", compression="zstd")
    print(f"rev {split}: S1 names {len(s1):,}; name-only pool queries {len(q):,} (of {used.size:,} candidate pool records)")


def feats(set_name):
    split = "test" if set_name == "TEST" else "train"
    X, s1, pool, _ = match.load_features(set_name, match.bucketing_fingerprint(io_utils.load_config()))
    top = pq.read_table(R / f"rev_{split}_topk.parquet").to_pandas()
    top = top.rename(columns={"qid": "p", "pool_idx": "s"})
    head = top[top["rank"] == 1].set_index("p").cos.astype(np.float32)
    ncl = top.merge(head.rename("top"), left_on="p", right_index=True)
    ncl = ncl[ncl.cos >= ncl.top - 0.02].groupby("p").size()
    rows = pd.DataFrame({"s": s1.astype(np.int64), "p": pool.astype(np.int64)})
    q = set(top.p.unique().tolist())
    isq = rows.p.isin(q).to_numpy()
    own = rows.merge(top[["p", "s", "rank", "cos"]], on=["p", "s"], how="left")
    best_other = top.merge(rows[isq], on="p")
    best_other = best_other[best_other.s_x != best_other.s_y].groupby(["p", "s_y"]).cos.max()
    bo = pd.Series(best_other.to_numpy(), index=pd.MultiIndex.from_arrays(
        [best_other.index.get_level_values(0), best_other.index.get_level_values(1)]))
    bo = bo.reindex(pd.MultiIndex.from_arrays([rows.p, rows.s])).to_numpy()
    F = {}
    nan = np.full(len(rows), np.nan, dtype=np.float32)
    in20 = own["rank"].notna().to_numpy()
    F["rv_in20"] = np.where(isq, in20.astype(np.float32), nan)
    F["rv_rank"] = np.where(isq, np.where(in20, own["rank"].to_numpy(), 21), nan).astype(np.float32)
    F["rv_cos"] = np.where(isq & in20, own["cos"].to_numpy(), nan).astype(np.float32)
    F["rv_top"] = np.where(isq, head.reindex(rows.p).to_numpy(), nan).astype(np.float32)
    F["rv_gap"] = (F["rv_top"] - F["rv_cos"]).astype(np.float32)
    F["rv_margin"] = np.where(isq & in20, own["cos"].to_numpy() - bo, nan).astype(np.float32)
    F["rv_nclose"] = np.where(isq, ncl.reindex(rows.p).to_numpy(), nan).astype(np.float32)
    pq.write_table(pa.table({"s1_idx": pa.array(s1), "pool_idx": pa.array(pool), **{k: pa.array(v) for k, v in F.items()}}),
                   R / f"rev_{set_name}.parquet", compression="zstd")
    print(f"rev feats {set_name}: {isq.sum():,} of {len(rows):,} rows name-only; own S1 in top 20: "
          f"{in20[isq].mean():.3f}, rank 1: {(own['rank'].to_numpy()[isq] == 1).mean():.3f}")


if __name__ == "__main__":
    R.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "prep":
        prep(sys.argv[2])
    else:
        for s in sys.argv[2:]:
            feats(s)
