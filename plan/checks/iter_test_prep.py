"""I1 on test: profile queries from S17's accepted pairs (inputs only). One row per S1 with >= 1 accepted pool record:
qid = s1_idx, kind 'profile', members = accepted pool_idx. Writes work/retrieval/iter_test_queries.parquet.
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from drace_test import cty, read_results


def main():
    acc, _ = read_results("submission/output_v17/matching_results.tsv")
    acc = acc.dropna().astype(np.int64)
    prof = acc.groupby("s1_idx").pool_idx.apply(list)
    s1c = np.asarray(cty(pq.read_table("work/records/test_s1.parquet", columns=["country_raw"]).column(0).to_pylist()), dtype=object)
    pq.write_table(pa.table({"qid": pa.array(prof.index.to_numpy(np.int64)), "kind": ["profile"] * len(prof),
                             "country": s1c[prof.index.to_numpy()].tolist(),
                             "s1_idx": pa.array(prof.index.to_numpy(np.int64)),
                             "self_pool": pa.array(np.full(len(prof), -1, dtype=np.int64)),
                             "members": pa.array([list(map(int, m)) for m in prof.to_list()], pa.list_(pa.int64()))}),
                   "work/retrieval/iter_test_queries.parquet", compression="zstd")
    print(f"profile queries {len(prof):,} from {len(acc):,} accepted pairs")


if __name__ == "__main__":
    main()
