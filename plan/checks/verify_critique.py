"""Independent re-check of the critique's key numbers (plan/critique/CRITIQUE.md).
Uses the reference-normalized parquet files the critic produced (plan/critique/out/norm_*.parquet,
same normalizer as plan/reference/translit_ref.py) but recomputes everything here.
Writes plan/checks/verify_critique.json.
Checks:
 C1 France pool vs France S1 per-record token shares for r, rue, av, st, s, a (test, structural)
 C2 exact normalized-name equality share over ALL true links; same for chain>=10 S1 (chain = count of
    the S1 name among train S1 names)
 C3 skeleton df on the train pool for LMT/PRPT; share of Indic true pairs with >=1 shared skeleton / token
    whose pool df <= 0.002*n_pool (usable), on all Indic true pairs
 C4 d_EN (pool df of the query's exact name) median/p90 for chain>=10 true pairs
"""

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "plan/reference"))
from translit_ref import skeleton  # noqa: E402

O = os.path.join(ROOT, "plan/critique/out")
D = os.path.join(ROOT, "student_resource/dataset")


def share_tokens(names, toks):
    c = Counter()
    for s in names:
        st = set(s.split())
        for t in toks:
            c[t] += t in st
    return {t: round(c[t] / len(names), 4) for t in toks}


def main():
    out = {}
    # C1
    fr_pool = pd.read_parquet(os.path.join(O, "norm_test_pool_france.parquet"), columns=["addr_norm", "name_norm"])
    s1t = []
    with open(os.path.join(D, "test/test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            e, n, a, co = line.rstrip("\n").split("\t")
            if co == "France":
                s1t.append((n, a))
    from translit_ref import normalize_field
    s1_addr = [normalize_field(a, "addr")[0] for _, a in s1t]
    s1_name = [normalize_field(n, "name")[0] for n, _ in s1t]
    toks = ["r", "rue", "av", "avenue", "street", "suite", "saint", "imp", "rte", "no"]
    out["C1_france_addr_token_share"] = {"s1": share_tokens(s1_addr, toks),
                                         "pool": share_tokens(fr_pool["addr_norm"].tolist(), toks)}
    out["C1_france_name_token_share"] = {"s1": share_tokens(s1_name, ["s", "a", "sas", "sarl", "club"]),
                                         "pool": share_tokens(fr_pool["name_norm"].tolist(), ["s", "a", "sas", "sarl", "club"])}
    out["C1_n"] = {"france_s1": len(s1t), "france_pool": len(fr_pool)}
    del fr_pool
    # C2/C3/C4
    s1 = pd.read_parquet(os.path.join(O, "norm_train_s1.parquet"), columns=["num", "name_norm"])
    s1_name = dict(zip(s1["num"].tolist(), s1["name_norm"].tolist()))
    chain = Counter(s1["name_norm"].tolist())
    del s1
    pool_name = {}
    pool_indic = set()
    en_df = Counter()
    tok_df = Counter()
    sk_df = Counter()
    for sfx, srcid in (("s2", 2), ("s3", 3)):
        p = pd.read_parquet(os.path.join(O, "norm_train_pool_%s.parquet" % sfx), columns=["num", "name_norm", "had_indic"])
        for num, nm, hi in zip(p["num"].tolist(), p["name_norm"].tolist(), p["had_indic"].tolist()):
            pool_name[(srcid, num)] = nm
            en_df[nm] += 1
            ts = set(t for t in nm.split() if len(t) >= 2)
            tok_df.update(ts)
            sk_df.update(set(k for k in (skeleton(t) for t in ts) if len(k) >= 2))
            if hi:
                pool_indic.add((srcid, num))
        del p
    n_pool = len(pool_name)
    cap = 0.002 * n_pool
    out["C3_skeleton_df"] = {k: sk_df[k] for k in ("LMT", "PRPT", "LK")}
    out["C3_cap"] = cap
    eq = Counter()
    ind = Counter()
    dens = []
    with open(os.path.join(D, "train/train_ground_truth.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            if not b:
                continue
            q = s1_name[int(a.split("-")[1])]
            ch = chain[q] >= 10
            qt = set(t for t in q.split() if len(t) >= 2)
            qs = set(k for k in (skeleton(t) for t in qt) if len(k) >= 2)
            for x in b.split(","):
                key = (2 if x[1] == "2" else 3, int(x.split("-")[1]))
                pn = pool_name[key]
                e = pn == q
                eq["all"] += 1
                eq["all_eq"] += e
                if ch:
                    eq["ch10"] += 1
                    eq["ch10_eq"] += e
                    dens.append(en_df[q])
                if key in pool_indic:
                    pt = set(t for t in pn.split() if len(t) >= 2)
                    ps = set(k for k in (skeleton(t) for t in pt) if len(k) >= 2)
                    ind["n"] += 1
                    ind["tok_any"] += bool(qt & pt)
                    ind["tok_usable"] += any(tok_df[t] <= cap for t in qt & pt)
                    ind["sk_any"] += bool(qs & ps)
                    ind["sk_usable"] += any(sk_df[k] <= cap for k in qs & ps)
    out["C2_exact_name_equal"] = {"all": round(eq["all_eq"] / eq["all"], 4),
                                  "chain10": round(eq["ch10_eq"] / eq["ch10"], 4),
                                  "chain10_link_share": round(eq["ch10"] / eq["all"], 4)}
    out["C3_indic_pairs"] = {"n": ind["n"], **{k: round(ind[k] / ind["n"], 4) for k in ("tok_any", "tok_usable", "sk_any", "sk_usable")}}
    d = np.array(dens)
    out["C4_dEN_chain10"] = {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
                             "share_ge_50": round(float((d >= 50).mean()), 4)}
    json.dump(out, open(os.path.join(ROOT, "plan/checks/verify_critique.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
