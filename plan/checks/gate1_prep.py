"""Gate 1 inputs (retrieval recall channel): texts for ce/hf_job/retrieve_job.py, train side only.

  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/gate1_prep.py [--limit-s1 N]

Writes work/retrieval/gate1_pool.parquet (every train pool record) and work/retrieval/gate1_queries.parquet:
  kind 's1'   each MATCH_TRAIN S1 (qid = s1_idx)
  kind 'seed' each pool record M027's OOF decision accepts for a MATCH_TRAIN S1 (qid = 10^9 + row; copy expansion);
              the seed -> S1 map goes to work/retrieval/gate1_seeds.parquet
Labels are not used here; seeds come from model predictions, as they would on test.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber import match

OUT = Path("work/retrieval")
SEED_BASE = 10 ** 9


def text(names, addrs):
    n = pd.Series(names).fillna("").astype(str).str.strip()
    a = pd.Series(addrs).fillna("").astype(str).str.strip()
    return ("query: " + n + " | " + a).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-s1", type=int, default=0, help="smoke: first N MATCH_TRAIN S1 and a pool sample")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "gate1" if not args.limit_s1 else "gate1smoke"

    pool = pq.read_table("work/records/train_pool.parquet", columns=["pool_idx", "name_raw", "addr_raw", "country_raw"])
    s1t = pq.read_table("work/records/train_s1.parquet", columns=["s1_idx", "name_raw", "addr_raw", "country_raw"])
    queries = np.load(match.runs_dir("M027") / "queries.npy")
    oof = pq.read_table(match.runs_dir("M027") / "oof.parquet").to_pandas()
    dec = {"rule": "threshold", "threshold": 0.73, "one_owner": True}  # exp_M027.json decision
    pred = match.apply_decision(dec, oof.s1_idx.to_numpy(), oof.pool_idx.to_numpy(), oof.p_cal.to_numpy())
    seeds = oof.loc[pred, ["s1_idx", "pool_idx"]].reset_index(drop=True)

    pdf = pool.to_pandas()
    sdf = s1t.to_pandas().set_index("s1_idx")
    if args.limit_s1:
        queries = queries[:args.limit_s1]
        seeds = seeds[seeds.s1_idx.isin(queries)].reset_index(drop=True)
        gt = pq.read_table("work/records/train_gt.parquet").to_pandas()
        keep = set(gt.loc[gt.s1_idx.isin(queries), "pool_idx"]) | set(seeds.pool_idx) | set(
            oof.loc[oof.s1_idx.isin(queries), "pool_idx"])
        rng = np.random.default_rng(0)
        extra = rng.choice(len(pdf), size=50_000, replace=False)
        pdf = pdf[pdf.pool_idx.isin(keep) | pdf.index.isin(extra)]
    cty = lambda s: pd.Series(s).fillna("NA").astype(str).str.strip().tolist()
    pq.write_table(pa.table({"pool_idx": pa.array(pdf.pool_idx.to_numpy().astype(np.uint32)),
                             "country": cty(pdf.country_raw), "text": text(pdf.name_raw, pdf.addr_raw)}),
                   OUT / f"{tag}_pool.parquet", compression="zstd")

    q = sdf.loc[queries]
    pp = pdf.set_index("pool_idx").loc[seeds.pool_idx] if not args.limit_s1 else \
        pool.to_pandas().set_index("pool_idx").loc[seeds.pool_idx]
    qid = np.concatenate([queries.astype(np.int64), SEED_BASE + np.arange(len(seeds), dtype=np.int64)])
    pq.write_table(pa.table({
        "qid": pa.array(qid),
        "kind": ["s1"] * len(queries) + ["seed"] * len(seeds),
        "country": cty(q.country_raw) + cty(pp.country_raw),
        "text": text(q.name_raw, q.addr_raw) + text(pp.name_raw, pp.addr_raw),
        "self_pool": pa.array(np.concatenate([np.full(len(queries), -1), seeds.pool_idx.to_numpy()]).astype(np.int64)),
    }), OUT / f"{tag}_queries.parquet", compression="zstd")
    seeds.assign(qid=SEED_BASE + np.arange(len(seeds))).to_parquet(OUT / f"{tag}_seeds.parquet")
    print(f"{tag}: pool {len(pdf):,}, s1 queries {len(queries):,}, seed queries {len(seeds):,} -> {OUT}")


if __name__ == "__main__":
    main()
