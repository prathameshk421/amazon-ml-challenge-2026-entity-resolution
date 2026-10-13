"""M036 group-profile features: compare each candidate with the set of pool records already accepted for its S1
(first pass), using character / token similarity (not exact names, not e5). Inputs only.

  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/grp_feats.py MATCH_TRAIN [TEST ...]

First-pass accepted set A(s): MATCH_TRAIN -> M027 out-of-fold decision (threshold 0.73 + one-owner); TEST -> S9
(work/outputs/full_c0/matching_results.tsv). The candidate itself is excluded from A(s).
Columns (NaN when A(s) \\ {p} is empty):
  g_n_acc      |A(s) \\ {p}|
  g_name_max   max token-set ratio(name p, name k), k in A
  g_name_rmax  max plain edit ratio(name p, name k)
  g_tok_cov    share of p's name tokens (len >= 2) found in the union of A's name tokens and the S1 name
  g_addr_max   max token-set ratio(addr p, addr k) (NaN if p's address is empty)
  g_same_src   number of k in A from the same source as p
Writes work/retrieval/grp_<SET>.parquet in feature-row order.
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from ber import io_utils, match

TOK = re.compile(r"[^\w]+", re.UNICODE)
SOURCE = "m027"  # "firstpass" -> plan/checks/firstpass_oof.py decisions (full-scale training rows)


def norm(s):
    return " ".join(TOK.sub(" ", str(s).lower()).split())


def accepted(set_name, s1, pool):
    fp = Path(f"work/retrieval/firstpass_{set_name}.parquet")
    if SOURCE == "firstpass" and set_name != "TEST":
        f = pq.read_table(fp).to_pandas()
        return f.loc[f.accepted, ["s1_idx", "pool_idx"]].astype(np.int64)
    if set_name == "MATCH_TRAIN":
        o = pq.read_table(match.runs_dir("M027") / "oof.parquet").to_pandas()
        k = match.apply_decision({"rule": "threshold", "threshold": 0.73, "one_owner": True},
                                 o.s1_idx.to_numpy(), o.pool_idx.to_numpy(), o.p_cal.to_numpy())
        return o.loc[k, ["s1_idx", "pool_idx"]]
    sys.path.insert(0, "plan/checks")
    from drace_test import read_results
    b, _ = read_results("work/outputs/full_c0/matching_results.tsv")
    return b.astype(np.int64)


def main(set_name):
    t0 = time.time()
    split = "test" if set_name == "TEST" else "train"
    X, s1, pool, _ = match.load_features(set_name, match.bucketing_fingerprint(io_utils.load_config()))
    pt = pq.read_table(f"work/records/{split}_pool.parquet", columns=["src", "name_raw", "addr_raw"]).to_pandas()
    st = pq.read_table(f"work/records/{split}_s1.parquet", columns=["name_raw"]).to_pandas()
    acc = accepted(set_name, s1, pool)
    groups = acc.groupby("s1_idx").pool_idx.apply(list).to_dict()
    pname = pt.name_raw.fillna("").map(norm).to_numpy()
    paddr = pt.addr_raw.fillna("").map(norm).to_numpy()
    psrc = pt.src.to_numpy()
    sname = st.name_raw.fillna("").map(norm).to_numpy()
    n = s1.size
    F = {c: np.full(n, np.nan, dtype=np.float32) for c in
         ("g_n_acc", "g_name_max", "g_name_rmax", "g_tok_cov", "g_addr_max", "g_same_src")}
    need = np.ones(n, bool)
    if set_name == "DEV_REST_REST":  # only fold-1 rows are ever used (fold 0 = c0 training S1)
        need = (((s1.astype(np.uint64) * np.uint64(0x9E3779B1)) % np.uint64(2 ** 32)) >> np.uint64(31)) == 1
    for i in np.flatnonzero(need):
        g = groups.get(int(s1[i]))
        if not g:
            continue
        p = int(pool[i])
        g = [k for k in g if k != p]
        if not g:
            continue
        nm, ad = pname[p], paddr[p]
        F["g_n_acc"][i] = len(g)
        F["g_name_max"][i] = max(fuzz.token_set_ratio(nm, pname[k]) for k in g) / 100
        F["g_name_rmax"][i] = max(fuzz.ratio(nm, pname[k]) for k in g) / 100
        toks = [t for t in nm.split() if len(t) >= 2]
        if toks:
            union = set(sname[int(s1[i])].split())
            for k in g:
                union.update(pname[k].split())
            F["g_tok_cov"][i] = sum(t in union for t in toks) / len(toks)
        if ad:
            F["g_addr_max"][i] = max(fuzz.token_set_ratio(ad, paddr[k]) for k in g) / 100
        F["g_same_src"][i] = sum(psrc[k] == psrc[p] for k in g)
        if i % 1_000_000 == 0 and i:
            print(f"{set_name}: {i:,}/{n:,} ({time.time() - t0:.0f}s)", flush=True)
    out = Path("work/retrieval") / (f"grp_{set_name}.parquet" if SOURCE == "m027" else f"grp_full_{set_name}.parquet")
    pq.write_table(pa.table({"s1_idx": pa.array(s1), "pool_idx": pa.array(pool), **{k: pa.array(v) for k, v in F.items()}}),
                   out, compression="zstd")
    print(f"grp {set_name}: {np.isfinite(F['g_n_acc']).mean():.3f} of rows have a group; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--firstpass":
        SOURCE = "firstpass"
        sys.argv.pop(1)
    for s in sys.argv[1:]:
        main(s)
