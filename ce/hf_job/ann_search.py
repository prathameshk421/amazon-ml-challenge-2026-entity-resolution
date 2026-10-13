"""Approximate nearest-neighbour search (FAISS) for ce/hf_job/retrieve_job.py, run as its own process: on macOS the
faiss-cpu and torch wheels each bundle libomp, and the process aborts when both load (OMP Error #15).

  python ann_search.py --dir <emb dir> --index hnsw|ivfpq --k 50 --out <topk.parquet> [--exact <topk.parquet> --stats <json>]

<emb dir> holds P.npy, Q.npy (float16, L2-normalised, written by retrieve_job.py) and meta.npz (p_idx, p_cty, q_id,
q_cty, q_self). Search is per country, as in the exact path; a query country absent from the pool searches the whole
pool; a country with fewer than --min-ann pool vectors is searched exactly (IndexFlatIP).

Indexes (inner product = cosine on unit vectors):
  hnsw   IndexHNSWFlat, M --hnsw-m, efConstruction --ef-construction, efSearch --ef. Stores the full vectors, so every
         returned cosine is exact; the list changes only where the greedy graph walk misses a true neighbour.
  ivfpq  IndexIVFPQ (nlist = 4*sqrt(n) rounded to 64, 48 sub-quantisers x 8 bits, nprobe --nprobe) wrapped in
         IndexRefineFlat: PQ distances pick k*--refine candidates, then exact inner products re-rank them, so returned
         cosines are exact; misses come from the coarse IVF probe and PQ ranking error.
--exact: compare with the exact top-k built from the SAME embeddings (retrieve_job.py RETR_COMPARE=1). For each rank
cutoff in --cutoffs: recall of the exact list, share of queries whose top-c set differs, and how many of those differ
only by ties (every missed exact neighbour has cos <= the ANN list's c-th cos + 1e-3, i.e. within fp16 rounding).
Output parquet has the same schema as the exact path: qid:int64, rank:int16, pool_idx:uint32, cos:float16.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class Rows:
    """Accumulates per-query top lists (self match dropped, cut to K) into the topk parquet schema."""

    def __init__(self, p_idx, q_id, q_self, k):
        self.p_idx, self.q_id, self.q_self, self.k = p_idx, q_id, q_self, k
        self.cols = {"qid": [], "rank": [], "pool_idx": [], "cos": []}

    def add(self, rows, v, j):
        """rows: query positions; v, j: (len(rows), kk) cosines and GLOBAL pool positions (-1 = no result)."""
        for r, q in enumerate(rows):
            ok = j[r] >= 0
            pj, vv = j[r][ok], v[r][ok]
            keep = self.p_idx[pj] != self.q_self[q]
            pv, vv = self.p_idx[pj][keep][:self.k], vv[keep][:self.k]
            self.cols["qid"].append(np.full(pv.size, self.q_id[q], dtype=np.int64))
            self.cols["rank"].append(np.arange(1, pv.size + 1, dtype=np.int16))
            self.cols["pool_idx"].append(pv.astype(np.uint32))
            self.cols["cos"].append(vv.astype(np.float16))

    def table(self):
        return pa.table({k: pa.array(np.concatenate(v)) for k, v in self.cols.items()})


def build(x, kind, a):
    import faiss
    d = x.shape[1]
    if kind == "flat" or x.shape[0] < a.min_ann:
        ix = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        ix = faiss.IndexHNSWFlat(d, a.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        ix.hnsw.efConstruction = a.ef_construction
        ix.hnsw.efSearch = a.ef
    elif kind == "ivfpq":
        nlist = max(64, int(round(4 * np.sqrt(x.shape[0]) / 64)) * 64)
        base = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, a.pq_m, 8, faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(0)
        base.train(x[rng.choice(x.shape[0], min(x.shape[0], 256 * nlist), replace=False)])
        base.nprobe = a.nprobe
        ix = faiss.IndexRefineFlat(base)
        ix.k_factor = a.refine
    else:
        raise ValueError(kind)
    for s in range(0, x.shape[0], 500_000):
        ix.add(x[s:s + 500_000])
    return ix


def search(P, Q, meta, kind, a):
    import faiss
    faiss.omp_set_num_threads(os.cpu_count())
    p_cty, q_cty = meta["p_cty"], meta["q_cty"]
    out = Rows(meta["p_idx"], meta["q_id"], meta["q_self"], a.k)
    timing = {}
    for c in sorted(set(q_cty.tolist())):
        qi, pi = np.flatnonzero(q_cty == c), np.flatnonzero(p_cty == c)
        if pi.size == 0:
            pi = np.arange(P.shape[0])
        kk = min(a.k + 1, pi.size)
        t0 = time.time()
        ix = build(P[pi].astype(np.float32), kind, a)
        tb = time.time() - t0
        for s in range(0, qi.size, 65_536):
            rows = qi[s:s + 65_536]
            v, j = ix.search(Q[rows].astype(np.float32), kk)
            out.add(rows, v, np.where(j >= 0, pi[np.maximum(j, 0)], -1))
        timing[str(c)] = {"queries": int(qi.size), "pool": int(pi.size), "index": type(ix).__name__,
                          "build_s": round(tb, 1), "search_s": round(time.time() - t0 - tb, 1)}
        print(f"[{kind}] country {c!r}: {qi.size:,} queries x {pi.size:,} pool, {type(ix).__name__} "
              f"build {tb:.0f}s search {time.time() - t0 - tb:.0f}s", flush=True)
        del ix
    return out.table(), timing


def compare(exact, ann, q_country, cutoffs, tol=1e-3):
    """exact / ann: topk frames. q_country: Series qid -> country. Per cutoff and country: recall, differing queries."""
    res = {}
    for c in cutoffs:
        E = exact[exact["rank"] <= c]
        A = ann[ann["rank"] <= c]
        m = E.merge(A[["qid", "pool_idx"]], on=["qid", "pool_idx"], how="left", indicator=True)
        miss = m[m["_merge"] == "left_only"]
        a_last = A.groupby("qid").cos.min().astype(np.float32)
        tie = miss.cos.astype(np.float32).to_numpy() <= a_last.reindex(miss.qid).fillna(-np.inf).to_numpy() + tol
        q_miss = miss.groupby("qid").size()
        q_real = miss[~tie].groupby("qid").size()
        n_q = E.qid.nunique()
        row = {"exact_pairs": int(len(E)), "missed_pairs": int(len(miss)), "missed_pairs_not_tie": int((~tie).sum()),
               "recall": 1 - len(miss) / max(len(E), 1), "queries": int(n_q),
               "queries_set_differs": int(q_miss.size), "queries_differ_not_tie": int(q_real.size),
               "share_queries_differ": q_miss.size / max(n_q, 1)}
        by = {}
        ec = q_country.reindex(E.qid.unique())
        for cty, grp in ec.groupby(ec):
            qs = grp.index
            by[str(cty)] = {"queries": int(qs.size), "queries_set_differs": int(q_miss.index.isin(qs).sum()),
                            "queries_differ_not_tie": int(q_real.index.isin(qs).sum()),
                            "recall": 1 - miss.qid.isin(qs).sum() / max(E.qid.isin(qs).sum(), 1)}
        row["by_country"] = by
        res[f"@{c}"] = row
        print(f"top-{c}: recall {row['recall']:.5f}; queries differing {q_miss.size:,}/{n_q:,} "
              f"({row['share_queries_differ']:.5f}), not tie-only {q_real.size:,}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--index", choices=["hnsw", "ivfpq", "flat"], required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exact")
    ap.add_argument("--stats")
    ap.add_argument("--cutoffs", default="1,2,5,20,50")
    ap.add_argument("--min-ann", type=int, default=50_000)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=80)
    ap.add_argument("--ef", type=int, default=256)
    ap.add_argument("--pq-m", type=int, default=48)
    ap.add_argument("--nprobe", type=int, default=64)
    ap.add_argument("--refine", type=int, default=8)
    a = ap.parse_args()
    d = Path(a.dir)
    P, Q = np.load(d / "P.npy", mmap_mode="r"), np.load(d / "Q.npy", mmap_mode="r")
    meta = dict(np.load(d / "meta.npz", allow_pickle=True))
    t0 = time.time()
    tbl, timing = search(P, Q, meta, a.index, a)
    pq.write_table(tbl, a.out, compression="zstd")
    wall = time.time() - t0
    print(f"wrote {a.out}: {tbl.num_rows:,} rows; {a.index} search {wall / 60:.1f} min", flush=True)
    if a.exact:
        qc = pd.Series(meta["q_cty"], index=meta["q_id"])
        cuts = [c for c in map(int, a.cutoffs.split(",")) if c <= a.k]
        stats = {"index": a.index, "params": {k: v for k, v in vars(a).items() if k not in ("dir", "out", "exact", "stats")},
                 "wall_s": round(wall, 1), "per_country": timing,
                 "compare": compare(pd.read_parquet(a.exact), tbl.to_pandas(), qc, cuts)}
        if a.stats:
            Path(a.stats).write_text(json.dumps(stats, indent=1))
            print(f"wrote {a.stats}", flush=True)


if __name__ == "__main__":
    main()
