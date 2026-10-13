"""Precompute exp feature groups for TEST (inputs only; logged in reports/test_usage_log.md). Groups used by a
submission run must exist for TEST; exp-predict computes missing ones itself, this only moves the work earlier.
Run: PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/test_groups.py comp num"""
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber import exp, io_utils, match


def main():
    d = match.features_dir("TEST")
    t = pa.concat_tables([pq.read_table(p, columns=["s1_idx", "pool_idx"]) for p in sorted(d.glob("part-*.parquet"))])
    s1, pool = t.column("s1_idx").to_numpy().astype(np.int32), t.column("pool_idx").to_numpy()
    for g in sys.argv[1:]:
        io_utils.log_test_usage(f"exp feature group {g} over all test candidates (inputs only)",
                                "submission 3 features (test side of the adopted groups)", str(d))
        exp.load_group("TEST", g, None, s1, pool, io_utils.get_logger("test-groups"))
        print(g, "done", flush=True)


if __name__ == "__main__":
    main()
