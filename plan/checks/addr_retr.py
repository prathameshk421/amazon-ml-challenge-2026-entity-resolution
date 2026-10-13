"""AD1: address-only lexical retrieval (new channel). Binary TF-IDF over address tokens (lowercase alnum, df <= DFMAX in
the pool), L2-normalised; S1 address query -> top-K pool records of the same country by cosine, pool names ignored.
Targets true copies whose name was replaced (name Jaccard 0) but whose address survived (N13: 1,224 such pairs missed by
every S14 channel on MATCH_TRAIN; c0 >= 0.995 on 80.7% of such in-list true pairs, 8 of 32,347 false).

  train:  ... addr_retr.py train  -> work/retrieval/addr_train_topk.parquet (MATCH_TRAIN S1) + recall/oracle report
  test:   ... addr_retr.py test   -> work/retrieval/addr_test_topk.parquet
Writes artifacts/metrics/addr_retr_<split>.json.
"""
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer

from ber import io_utils, match

K = 10
DFMAX = 3000
CH = 4000


def cty(s):
    return pd.Series(s).fillna("NA").astype(str).str.strip().to_numpy()


def run(split, q_s1):
    t0 = time.time()
    pool = pq.read_table(f"work/records/{split}_pool.parquet", columns=["pool_idx", "addr_raw", "country_raw"]).to_pandas()
    s1 = pq.read_table(f"work/records/{split}_s1.parquet", columns=["s1_idx", "addr_raw", "country_raw"]).to_pandas()
    s1 = s1.set_index("s1_idx").loc[q_s1].reset_index()
    pa_ = pool.addr_raw.fillna("").astype(str).str.lower().to_numpy()
    vec = TfidfVectorizer(token_pattern=r"[a-z0-9]{2,}|\d", binary=True, dtype=np.float32, max_df=DFMAX, min_df=2,
                          sublinear_tf=False)
    P = vec.fit_transform(pa_).tocsr()
    Q = vec.transform(s1.addr_raw.fillna("").astype(str).str.lower().to_numpy()).tocsr()
    print(f"[{split}] vocab {len(vec.vocabulary_):,}; pool {P.shape[0]:,} x S1 {Q.shape[0]:,} ({time.time() - t0:.0f}s)", flush=True)
    pc, qc = cty(pool.country_raw), cty(s1.country_raw)
    pidx = pool.pool_idx.to_numpy()
    PT = P.T.tocsr()
    out_s, out_p, out_r, out_c = [], [], [], []
    for a in range(0, Q.shape[0], CH):
        S = (Q[a:a + CH] @ PT.T.T if False else Q[a:a + CH] @ P.T).tocsr()
        for i in range(S.shape[0]):
            lo, hi = S.indptr[i], S.indptr[i + 1]
            if hi == lo:
                continue
            cols, vals = S.indices[lo:hi], S.data[lo:hi]
            m = pc[cols] == qc[a + i]
            cols, vals = cols[m], vals[m]
            if cols.size == 0:
                continue
            top = np.argsort(-vals, kind="stable")[:K]
            out_s.append(np.full(top.size, s1.s1_idx.iat[a + i])); out_p.append(pidx[cols[top]])
            out_r.append(np.arange(1, top.size + 1)); out_c.append(vals[top])
        if (a // CH) % 10 == 0:
            print(f"[{split}] {a + CH:,}/{Q.shape[0]:,} ({time.time() - t0:.0f}s)", flush=True)
    t = pd.DataFrame({"s1_idx": np.concatenate(out_s).astype(np.int64), "pool_idx": np.concatenate(out_p).astype(np.int64),
                      "rank": np.concatenate(out_r).astype(np.int16), "cos": np.concatenate(out_c).astype(np.float32)})
    t.to_parquet(f"work/retrieval/addr_{split}_topk.parquet")
    print(f"[{split}] wrote {len(t):,} rows ({time.time() - t0:.0f}s)")
    return t


if __name__ == "__main__":
    split = sys.argv[1]
    if split == "train":
        q = np.load(match.runs_dir("M027") / "queries.npy")
    else:
        q = pq.read_table("work/records/test_s1.parquet", columns=["s1_idx"]).column(0).to_numpy()
    run(split, q)
