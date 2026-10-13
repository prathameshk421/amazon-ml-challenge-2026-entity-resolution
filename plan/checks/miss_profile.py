"""N13: what do the true pairs missed by every S14 channel look like? MATCH_TRAIN, labels for scoring only.
Channels: stage-2 lists, untrained e5 S1@5 + seed@5, distilled e5 S1@5. Per missed pair: token-set Jaccard of names and
of addresses (lowercased alnum tokens), digits shared, empty-address flags. Compare with a sample of found true pairs.
Writes artifacts/metrics/miss_profile.json.
"""
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from gate2_eval import key

TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return set(TOK.findall(str(s).lower())) if isinstance(s, str) else set()


def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else np.nan


def prof(df, s1, pool):
    rows = []
    for s, p in zip(df.s1_idx.to_numpy(), df.pool_idx.to_numpy()):
        na, nb = toks(s1.name_raw[s]), toks(pool.name_raw[p])
        aa, ab = toks(s1.addr_raw[s]), toks(pool.addr_raw[p])
        da, db = {t for t in aa if t.isdigit()}, {t for t in ab if t.isdigit()}
        rows.append((jac(na, nb), jac(aa, ab) if ab else np.nan, len(ab) == 0, len(da & db) > 0 if db else np.nan))
    return pd.DataFrame(rows, columns=["name_j", "addr_j", "pool_empty", "digit_shared"])


def main():
    q = np.load(match.runs_dir("M027") / "queries.npy")
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas(); gt = gt[gt.s1_idx.isin(q)]
    cand = pq.read_table("work/candidates/MATCH_TRAIN/part-00000.parquet", columns=["s1_idx", "pool_idx"]).to_pandas()
    g = pd.read_parquet("work/retrieval/gate2_meta.parquet"); d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet")
    known = np.unique(np.concatenate([key(cand.s1_idx, cand.pool_idx), key(g.s1_idx, g.pool_idx), key(d.s1_idx, d.pool_idx)]))
    isk = np.isin(key(gt.s1_idx, gt.pool_idx), known)
    miss, found = gt[~isk], gt[isk].sample(20000, random_state=0)
    s1 = pq.read_table("work/records/train_s1.parquet", columns=["name_raw", "addr_raw", "country_raw"]).to_pandas()
    pool = pq.read_table("work/records/train_pool.parquet", columns=["name_raw", "addr_raw", "src"]).to_pandas()
    pm, pf = prof(miss, s1, pool), prof(found, s1, pool)
    out = {"n_missed": int(len(miss))}
    for lab, p, df in (("missed", pm, miss), ("found", pf, found)):
        ne = p[~p.pool_empty]
        r = {"pool_empty": float(p.pool_empty.mean()), "name_j_median": float(p.name_j.median()),
             "addr_j_median_nonempty": float(ne.addr_j.median()), "digit_shared_nonempty": float(ne.digit_shared.mean()),
             "addr_j_ge_0.6_nonempty": float((ne.addr_j >= 0.6).mean()), "name_j_ge_0.6": float((p.name_j >= 0.6).mean()),
             "name_lo_addr_hi": float(((ne.name_j < 0.4) & (ne.addr_j >= 0.6)).mean()),
             "name_hi_addr_lo": float(((ne.name_j >= 0.6) & (ne.addr_j < 0.3)).mean()),
             "src_S2": float((pool.src.to_numpy()[df.pool_idx] == 2).mean()),
             "india": float((s1.country_raw.fillna("").str.strip().to_numpy()[df.s1_idx] == "India").mean())}
        out[lab] = r
        print(lab, {k: round(v, 3) for k, v in r.items()})
    ex = miss[~pm.pool_empty.to_numpy()].sample(12, random_state=2)
    for s, p in zip(ex.s1_idx, ex.pool_idx):
        print(f"  S1: {s1.name_raw[s]} | {str(s1.addr_raw[s])[:60]}\n  PL: {pool.name_raw[p]} | {str(pool.addr_raw[p])[:60]}")
    io_utils.write_metrics("miss_profile", out, q.size, int(len(gt)))


if __name__ == "__main__":
    main()
