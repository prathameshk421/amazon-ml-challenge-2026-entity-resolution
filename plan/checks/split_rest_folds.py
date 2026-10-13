"""Split work/ce/pairs_DEV_REST_REST.parquet by cross-encoder fold (ce/train_continue.py fold_of: top bit of
s1_idx * 0x9E3779B1 mod 2^32). Fold 0 trains c0; fold 1 is scored by c0 and used for the matcher and the e5
distillation. Writes work/ce/pairs_DEV_REST_REST_F1.parquet (fold-1 rows, original row order).

  .venv/bin/python plan/checks/split_rest_folds.py
"""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SRC = Path("work/ce/pairs_DEV_REST_REST.parquet")


def fold_of(s1):
    return ((s1.astype(np.uint64) * np.uint64(0x9E3779B1)) % np.uint64(2 ** 32)) >> np.uint64(31)


def main():
    t = pq.read_table(SRC)
    keep = fold_of(t.column("s1_idx").to_numpy()) == 1
    out = t.filter(keep)
    pq.write_table(out, SRC.with_name("pairs_DEV_REST_REST_F1.parquet"), compression="zstd", row_group_size=500_000)
    print(f"DEV_REST_REST: {t.num_rows:,} pairs -> fold 1: {out.num_rows:,} pairs")


if __name__ == "__main__":
    main()
