"""Pull cross-encoder scores written by ce/hf_job/score_job.py from the HF dataset repo and assemble them.

  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python ce/hf_job/collect.py [--ce ce_rec]

For each set: concatenates scores/<SET>/c*.parquet in chunk order, checks row count and (s1_idx, pool_idx) order
against work/ce/pairs_<SET>.parquet, writes work/ce/<ce>/<SET>.parquet (the file ce/ce_submit.py reads).
For TEST also writes work/ce/<ce>/TEST_pairs_readable.parquet: s1_idx, pool_idx, s1_id, pool_id, p_ce — one row per
(S1 record, S2/S3 candidate record) with its match probability, for reuse as a feature or for inspection.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submission/code/business_entity_resolution/src"))
from ber import hashing, io_utils  # noqa: E402

REPO = "prathameshk421/ber-ce-scoring"
SETS = ("MATCH_VALID", "MATCH_TRAIN_S20000", "TEST")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce", default="ce_rec")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--prefix", default="scores", help="score prefix in the HF repo (c0: scores_c0)")
    args = ap.parse_args()
    ce = io_utils.paths()["work_root"] / "ce"
    dl = Path(snapshot_download(REPO, repo_type="dataset", local_dir=str(ce / "hf_pull"), allow_patterns=[f"{args.prefix}/*"]))
    out = ce / args.ce
    out.mkdir(parents=True, exist_ok=True)
    for s in args.sets.split(","):
        parts = sorted((dl / args.prefix / s).glob("c*.parquet"))
        ref = pq.read_table(ce / f"pairs_{s}.parquet", columns=["s1_idx", "pool_idx"])
        t = pa.concat_tables([pq.read_table(p) for p in parts]) if parts else None
        if t is None or t.num_rows != ref.num_rows:
            print(f"{s}: {0 if t is None else t.num_rows:,} scored rows vs {ref.num_rows:,} pairs ({len(parts)} chunks) — incomplete, skipped")
            continue
        for c in ("s1_idx", "pool_idx"):
            if not np.array_equal(t.column(c).to_numpy(), ref.column(c).to_numpy()):
                raise RuntimeError(f"{s}: {c} order differs from pairs_{s}.parquet")
        p = t.column("p_ce").to_numpy()
        pq.write_table(t, out / f"{s}.parquet", compression="zstd")
        print(f"{out / f'{s}.parquet'}: {t.num_rows:,} rows, non-finite {int((~np.isfinite(p)).sum())}, mean p {np.nanmean(p):.4f}")
        if s == "TEST":
            s1_ids = pd.read_parquet(io_utils.work_path("records", "test_s1.parquet"), columns=["entity_id"])["entity_id"].to_numpy()
            store = io_utils.paths()["work_root"] / "norm" / "test_pool_store"
            src, num = np.load(store / "src.npy"), np.load(store / "num.npy")
            s1i, pi = t.column("s1_idx").to_numpy(), t.column("pool_idx").to_numpy()
            pool_ids = [hashing.format_id(a, b) for a, b in zip(src[pi].tolist(), num[pi].tolist())]
            pq.write_table(pa.table({"s1_idx": s1i, "pool_idx": pi, "s1_id": s1_ids[s1i], "pool_id": pool_ids, "p_ce": p}),
                           out / "TEST_pairs_readable.parquet", compression="zstd")
            print(f"{out / 'TEST_pairs_readable.parquet'}: (S1 id, pool id, p_ce) for {t.num_rows:,} pairs")


if __name__ == "__main__":
    main()
