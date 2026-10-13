"""Prototype for critique M3: does a LEARNED linear stage-1 score beat the hand composite (F1) and
the quota union at equal candidate count? Uses the OLD indexes (artifacts/index: v1 normalization +
aux PAIR/ROM keys) because the v2 index does not exist yet, so absolute numbers are v1-era; the
comparison between scorers is what matters.
Train queries: 4,000 fit S1 (seed 777) disjoint from the 10K eval set (seed 123 sample, same as
diag_*). Features per union candidate: 8 channel scores (EX NT TR AD PH PAIR ROMT ROMG), 4 coverage
ratios, log1p of each score. Logistic regression (sklearn, C=1.0, class_weight=None), negatives =
all top-300 by F1 + 300 random from the rest of the union (weight n_rest/300).
Eval (10K queries): pair recall, entity recall and mean per-entity F0.5 ceiling (singletons = 1)
for top-K of F1 vs learned score (K = 50, 150, 300, 500), and quota union of the diag_quota config
"B300+N100+A100+P100+R50" vs the same with B replaced by the learned score.
Run with PYTHONHASHSEED=0. Writes plan/checks/learned_stage1.json.
"""

import csv
import json
import os
import random
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "legacy", "src"))
from normalize import normalize_name, normalize_address  # noqa: E402
from block_query2 import Q  # noqa: E402
from diag_blocking import union_scores  # noqa: E402
from diag_blocking2 import Aux, aux_scores  # noqa: E402

assert os.environ.get("PYTHONHASHSEED") == "0"
TR = os.path.join(ROOT, "student_resource/dataset/train")
IDX = os.path.join(ROOT, "artifacts/index")
t0 = time.time()


def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


def f05(r):
    return 1.25 * r / (0.25 + r) if r > 0 else 0.0


def build(q, aux, nn, na):
    res = union_scores(q, nn, na)
    extra = aux_scores(aux, nn)
    if res is not None:
        u0, S0, qs = res
    else:
        u0, S0, qs = np.zeros(0, np.uint32), np.zeros((5, 0)), np.zeros(5)
    P = [u0] + [a for a, _, _ in extra]
    Pall = np.concatenate(P).astype(np.uint32)
    if len(Pall) == 0:
        return None
    u, inv = np.unique(Pall, return_inverse=True)
    j0 = np.searchsorted(u, u0)
    S = np.zeros((8, len(u)))
    S[:5, j0] = S0
    W = np.concatenate([np.zeros(len(u0))] + [np.full(len(a), w) for a, w, _ in extra])
    C = np.concatenate([np.zeros(len(u0), np.uint8)] + [np.full(len(a), c, np.uint8) for a, _, c in extra])
    qself = np.zeros(8)
    qself[:5] = qs
    for k, c in enumerate((5, 6, 7)):
        m = C == c
        if m.any():
            S[5 + k] = np.bincount(inv[m], weights=W[m], minlength=len(u))
            qself[5 + k] = sum(w for a, w, cc in extra if cc == c)
    cov = np.vstack([S[i] / qself[i] if qself[i] > 0 else np.zeros(len(u)) for i in (1, 2, 3, 5)])
    X = np.vstack([S, cov, np.log1p(S)]).T
    f1 = 10 * S[0] + S[1] + S[2] + 0.2 * S[4] + S[3]
    return u, X, f1, S


def topk(score, k):
    if len(score) <= k:
        return np.arange(len(score))
    return np.argpartition(-score, k - 1)[:k]


def main():
    split = json.load(open(os.path.join(ROOT, "artifacts/split.json")))
    fit = split["fit_ids"]
    evalq = random.Random(123).sample(fit, 10000)
    rest = sorted(set(fit) - set(evalq))
    trainq = random.Random(777).sample(rest, 4000)
    want = set(evalq) | set(trainq)
    recs = {}
    with open(os.path.join(TR, "train_source1.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in want:
                recs[row["entity_id"]] = (normalize_name(row["business_name"]), normalize_address(row["business_address"]))
    truth = {}
    with open(os.path.join(TR, "train_ground_truth.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["source1_entity_id"] in want:
                m = row["matched_entity_ids"]
                truth[row["source1_entity_id"]] = m.split(",") if m else []
    src = np.load(os.path.join(IDX, "pool.src.npy"))
    num = np.load(os.path.join(IDX, "pool.num.npy"))
    key = (src.astype(np.int64) << 32) | num
    order = np.argsort(key)
    skey = key[order]
    tro = {}
    for e, ids in truth.items():
        rows = []
        for x in ids:
            s = 0 if x.startswith("S2-") else 1
            k = (np.int64(s) << np.int64(32)) | np.int64(int(x.split("-", 1)[1]))
            j = int(np.searchsorted(skey, k))
            if j < len(skey) and skey[j] == k:
                rows.append(int(order[j]))
        tro[e] = set(rows)
    del key, order, skey
    q = Q()
    aux = Aux()
    log("indexes loaded")
    Xs, ys, ws = [], [], []
    rng = np.random.default_rng(0)
    for i, e in enumerate(trainq):
        b = build(q, aux, *recs[e])
        if b is None:
            continue
        u, X, f1, _ = b
        y = np.isin(u, list(tro[e])).astype(np.int8) if tro[e] else np.zeros(len(u), np.int8)
        top = topk(f1, 300)
        restidx = np.setdiff1d(np.arange(len(u)), top)
        pos = np.flatnonzero(y)
        samp = rng.choice(restidx, min(300, len(restidx)), replace=False) if len(restidx) else restidx
        keep = np.unique(np.concatenate([top, samp, pos]))
        w = np.ones(len(keep))
        insamp = np.isin(keep, samp) & ~np.isin(keep, top) & ~np.isin(keep, pos)
        if len(samp):
            w[insamp] = len(restidx) / len(samp)
        Xs.append(X[keep]); ys.append(y[keep]); ws.append(w)
        if (i + 1) % 1000 == 0:
            log("train q %d" % (i + 1))
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys); wtr = np.concatenate(ws)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit((Xtr - mu) / sd, ytr, sample_weight=wtr)
    log("fitted LR on %d rows, %d positives" % (len(ytr), int(ytr.sum())))
    KS = [50, 150, 300, 500]
    agg = {m: {K: [0, 0, 0.0] for K in KS} for m in ("F1", "LR")}
    quota = {"hand": [0, 0, 0.0, 0], "learned": [0, 0, 0.0, 0]}
    tot = 0
    nq = 0
    for i, e in enumerate(evalq):
        ts = tro[e]
        nq += 1
        b = build(q, aux, *recs[e])
        if b is None:
            for m in agg:
                for K in KS:
                    agg[m][K][2] += 1.0 if not ts else 0.0
            for k in quota:
                quota[k][2] += 1.0 if not ts else 0.0
            tot += len(ts)
            continue
        u, X, f1, S = b
        lrs = (((X - mu) / sd) @ lr.coef_[0])
        tot += len(ts)
        for m, sc in (("F1", f1), ("LR", lrs)):
            for K in KS:
                sel = set(u[topk(sc, K)].tolist())
                h = len(sel & ts)
                agg[m][K][0] += h
                agg[m][K][1] += int(h > 0) if ts else 0
                agg[m][K][2] += f05(h / len(ts)) if ts else 1.0
        name = 10 * S[0] + S[1] + S[2] + 0.2 * S[4]
        for k, bsc in (("hand", f1), ("learned", lrs)):
            sel = set()
            for sc, qq in ((bsc, 300), (name, 100), (S[3], 100), (S[5], 100), (S[6] + S[7], 50)):
                nz = np.flatnonzero(sc > 0) if sc is not lrs else np.arange(len(sc))
                if len(nz) > qq:
                    nz = nz[np.argpartition(-sc[nz], qq - 1)[:qq]]
                sel |= set(u[nz].tolist())
            h = len(sel & ts)
            quota[k][0] += h
            quota[k][1] += int(h > 0) if ts else 0
            quota[k][2] += f05(h / len(ts)) if ts else 1.0
            quota[k][3] += len(sel)
        if (i + 1) % 2000 == 0:
            log("eval q %d" % (i + 1))
    ne = sum(1 for e in evalq if tro[e])
    out = {"n_eval_queries": nq, "n_nonsingleton": ne, "n_true_pairs": tot,
           "n_train_rows": int(len(ytr)), "n_train_pos": int(ytr.sum()),
           "lr_coef": dict(zip(["EX", "NT", "TR", "AD", "PH", "PAIR", "ROMT", "ROMG", "covNT", "covTR", "covAD", "covPAIR"] +
                               ["log_" + c for c in ["EX", "NT", "TR", "AD", "PH", "PAIR", "ROMT", "ROMG"]],
                               [round(float(c), 3) for c in lr.coef_[0]])),
           "topK": {m: {str(K): {"pair_recall": round(v[0] / tot, 4), "entity_recall": round(v[1] / ne, 4),
                                  "f05_ceiling": round(v[2] / nq, 4)} for K, v in d.items()} for m, d in agg.items()},
           "quota_B300_N100_A100_P100_R50": {k: {"pair_recall": round(v[0] / tot, 4), "entity_recall": round(v[1] / ne, 4),
                                                 "f05_ceiling": round(v[2] / nq, 4), "mean_size": round(v[3] / nq, 1)}
                                             for k, v in quota.items()}}
    json.dump(out, open(os.path.join(ROOT, "plan/checks/learned_stage1.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("topK", "quota_B300_N100_A100_P100_R50", "lr_coef")}, indent=1))


if __name__ == "__main__":
    main()
