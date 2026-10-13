"""Prototype: unsupervised, language-agnostic address-abbreviation miner.
Validates on the TRAIN pool (does it rediscover rd->road, st->street, ave->avenue ...?), then runs
on the TEST pool (unsupervised statistic on test inputs; allowed if logged) to see what it finds
for France. Normalization = reference steps 1-10 WITHOUT the abbreviation map.
Writes plan/checks/abbrev_miner.json.

Algorithm (per pool):
 1. df(t) over records; N = #records in the sample.
 2. short S: alphabetic, 1 <= len <= 4, df >= MIN_FRAC*N.   long L: alphabetic, len >= 4, df >= MIN_FRAC*N.
 3. (s, l) eligible iff l[0] == s[0], len(l) > len(s), s is a subsequence of l.
 4. context vector of t = [left-neighbour counts | right-neighbour counts] over the vocabulary,
    digits -> "<num>", record start/end -> "<s>"/"</s>"; L2-normalised.
 5. s -> l* = argmax cos(s, l) over eligible l, kept if cos >= THETA and df(l*) >= df(s) * RATIO.
"""

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "plan/reference"))
import translit_ref  # noqa: E402

translit_ref.ADDR_ABBREV = {}  # mine WITHOUT the train-derived map
D = os.path.join(ROOT, "student_resource/dataset")
MIN_FRAC = 0.0005
SAMPLE = 1500000


def is_sub(s, l):
    it = iter(l)
    return all(c in it for c in s)


def load(files, country=None, seed=0):
    rows = []
    for rel in files:
        with open(os.path.join(D, rel), encoding="utf-8") as f:
            f.readline()
            for line in f:
                e, n, a, co = line.rstrip("\n").split("\t")
                if country is None or co == country:
                    rows.append(a)
    random.Random(seed).shuffle(rows)
    return [translit_ref.normalize_field(a, "addr")[0].split() for a in rows[:SAMPLE]]


def mine(docs, thetas=(0.6, 0.9, 0.95, 0.98), ratio=0.1):
    N = len(docs)
    df = Counter()
    for d in docs:
        df.update(set(d))
    thr = MIN_FRAC * N
    alpha = re.compile(r"^[a-z]+$")
    S = [t for t, c in df.items() if c >= thr and alpha.match(t) and 1 <= len(t) <= 4]
    L = [t for t, c in df.items() if c >= thr and alpha.match(t) and len(t) >= 2]
    cand = defaultdict(list)
    for s in S:
        for l in L:
            if l != s and l[0] == s[0] and len(l) > len(s) and is_sub(s, l):
                cand[s].append(l)
    need = set(cand) | {l for v in cand.values() for l in v}
    vocab = {t: i for i, t in enumerate(sorted({t for t, c in df.items() if c >= thr} | {"<num>", "<s>", "</s>"}))}
    V = len(vocab)
    ctx = {t: np.zeros(2 * V) for t in need}
    for d in docs:
        m = ["<num>" if t.isdigit() else t for t in d]
        for i, t in enumerate(d):
            if t in ctx:
                lt = m[i - 1] if i > 0 else "<s>"
                rt = m[i + 1] if i + 1 < len(m) else "</s>"
                if lt in vocab:
                    ctx[t][vocab[lt]] += 1
                if rt in vocab:
                    ctx[t][V + vocab[rt]] += 1
    for t in ctx:
        n = np.linalg.norm(ctx[t])
        if n:
            ctx[t] /= n
    best = {}
    for s, ls in cand.items():
        sc = sorted(((float(ctx[s] @ ctx[l]), l) for l in ls), reverse=True)
        if sc and df[sc[0][1]] >= ratio * df[s]:
            second = sc[1][0] if len(sc) > 1 else 0.0
            best[s] = (sc[0][1], round(sc[0][0], 3), df[s], df[sc[0][1]], round(sc[0][0] - second, 3))
    return {str(th): {s: v for s, v in best.items() if v[1] >= th} for th in thetas}, N


def main():
    known = {"rd": "road", "st": "street", "ave": "avenue", "dr": "drive", "ln": "lane", "ct": "court",
             "blvd": "boulevard", "ste": "suite", "apt": "apartment", "hwy": "highway", "pkwy": "parkway"}
    out = {"params": {"MIN_FRAC": MIN_FRAC, "SAMPLE": SAMPLE}}
    tr_docs = load(["train/train_source2.tsv", "train/train_source3.tsv"])
    res, n = mine(tr_docs)
    out["train_pool"] = {"n_docs": n, "by_theta": {}}
    for th, m in res.items():
        hit = {s: m[s][0] for s in known if s in m}
        out["train_pool"]["by_theta"][th] = {
            "n_mined": len(m), "known_recovered_correct": sum(hit.get(s) == l for s, l in known.items()),
            "known_recovered_wrong": {s: hit[s] for s in hit if hit[s] != known[s]},
            "mined": m}
    del tr_docs
    fr_docs = load(["test/test_source2.tsv", "test/test_source3.tsv"], country="France")
    res, n = mine(fr_docs)
    out["test_pool_france"] = {"n_docs": n, "by_theta": {th: {"n_mined": len(m), "mined": m} for th, m in res.items()}}
    json.dump(out, open(os.path.join(ROOT, "plan/checks/abbrev_miner.json"), "w"), indent=1)
    for th in ("0.6", "0.9", "0.95", "0.98"):
        t = out["train_pool"]["by_theta"][th]
        print("TRAIN theta", th, "mined", t["n_mined"], "known correct", t["known_recovered_correct"], "/11 wrong", t["known_recovered_wrong"])
    print("TRAIN mined@0.95:", {s: v[:2] for s, v in out["train_pool"]["by_theta"]["0.95"]["mined"].items()})
    print("FRANCE mined@0.95:", {s: v[:2] for s, v in out["test_pool_france"]["by_theta"]["0.95"]["mined"].items()})


if __name__ == "__main__":
    main()
