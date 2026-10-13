"""I1: iterative retrieval on MATCH_TRAIN — inputs for ce/hf_job/iter_job.py.

  PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/iter_prep.py

S13-equivalent accepted set on the 180K MATCH_TRAIN S1: M027 OOF decision + DRACE additions at c0 >= 0.995 (untrained
e5 S1@5 / seed@2, and distilled e5 S1@5), one-owner. Labels are NOT used to build it (only c0 / LightGBM decisions).
Writes work/retrieval/iter_train_queries.parquet:
  kind 'profile'  one row per S1 with >= 1 accepted record: qid = s1_idx, members = accepted pool_idx list
  kind 'hop2'     one row per pool record added by DRACE (not by LightGBM): qid = 10^9 + row, self = pool_idx
and work/retrieval/iter_train_state.parquet (s1_idx, pool_idx, src: lgb|drace) for the evaluation.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber import match

T = 0.995
SB = 10 ** 9


def main():
    o = pq.read_table(match.runs_dir("M027") / "oof.parquet").to_pandas()
    k = match.apply_decision({"rule": "threshold", "threshold": 0.73, "one_owner": True},
                             o.s1_idx.to_numpy(), o.pool_idx.to_numpy(), o.p_cal.to_numpy())
    base = o.loc[k, ["s1_idx", "pool_idx"]].assign(src="lgb")
    a = pd.read_parquet("work/retrieval/gate2_meta.parquet")
    a["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
    a = a[(a.s1_rank <= 5) | (a.seed_rank <= 2)][["s1_idx", "pool_idx", "p"]]
    d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet")
    d["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2DS_TRAIN.parquet").column("p_ce").to_numpy()
    add = pd.concat([a, d[["s1_idx", "pool_idx", "p"]]]).drop_duplicates(["s1_idx", "pool_idx"])
    add = add[(add.p >= T) & ~add.pool_idx.isin(set(base.pool_idx))].sort_values("p", ascending=False) \
        .drop_duplicates("pool_idx")
    state = pd.concat([base, add[["s1_idx", "pool_idx"]].assign(src="drace")], ignore_index=True)
    state[["s1_idx", "pool_idx"]] = state[["s1_idx", "pool_idx"]].astype(np.int64)
    state.to_parquet("work/retrieval/iter_train_state.parquet")
    s1c = pq.read_table("work/records/train_s1.parquet", columns=["country_raw"]).column(0).to_pandas() \
        .fillna("NA").astype(str).str.strip().to_numpy()
    pc = pq.read_table("work/records/train_pool.parquet", columns=["country_raw"]).column(0).to_pandas() \
        .fillna("NA").astype(str).str.strip().to_numpy()
    prof = state.groupby("s1_idx").pool_idx.apply(list)
    hop = state[state.src == "drace"].reset_index(drop=True)
    q = pa.table({
        "qid": pa.array(np.concatenate([prof.index.to_numpy(np.int64), SB + np.arange(len(hop), dtype=np.int64)])),
        "kind": ["profile"] * len(prof) + ["hop2"] * len(hop),
        "country": s1c[prof.index.to_numpy()].tolist() + pc[hop.pool_idx.to_numpy()].tolist(),
        "s1_idx": pa.array(np.concatenate([prof.index.to_numpy(np.int64), hop.s1_idx.to_numpy(np.int64)])),
        "self_pool": pa.array(np.concatenate([np.full(len(prof), -1), hop.pool_idx.to_numpy(np.int64)])),
        "members": pa.array([list(map(int, m)) for m in prof.to_list()] + [[] for _ in range(len(hop))],
                            pa.list_(pa.int64())),
    })
    pq.write_table(q, "work/retrieval/iter_train_queries.parquet", compression="zstd")
    print(f"accepted {len(state):,} ({(state.src == 'drace').sum():,} by DRACE); profile queries {len(prof):,}; "
          f"hop-2 seeds {len(hop):,}")


if __name__ == "__main__":
    main()
