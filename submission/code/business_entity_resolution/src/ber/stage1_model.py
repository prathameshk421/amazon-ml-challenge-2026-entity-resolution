"""Stage-1 logistic model (plan §5.7 "Stage-1 model", evidence F22): `stage1-train`.

S1R_TRAIN queries through the stage-1 union with the current families and caps; rows = all positives,
the top `neg_top` union items by hand_both and `neg_rand` uniform from the rest (weight n_rest/n_sampled,
rng seed 7000 + s1_idx); standardized features; LogisticRegression(C, max_iter, random_state=0).
"""

import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from ber import io_utils, retrieve, splits
from ber.evaluate import truth_lists

SEED_BASE = 7000


def sample_rows(y, hand, s1_idx, neg_top, neg_rand):
    """(row indices ascending, weights) of one query's union."""
    keep = np.zeros(y.size, dtype=bool)
    keep[retrieve.top_q(hand, neg_top)] = True
    keep |= y
    w = np.ones(y.size)
    rest = np.flatnonzero(~keep)
    if rest.size:
        k = min(neg_rand, rest.size)
        samp = np.random.default_rng(SEED_BASE + int(s1_idx)).choice(rest, k, replace=False)
        keep[samp] = True
        w[samp] = rest.size / k
    rows = np.flatnonzero(keep)
    return rows, w[rows]


def _train_chunk(r, idx, names, addrs, chains, per_query, shared):
    """Worker function: sampled (X float32, y, w) rows per query."""
    neg_top, neg_rand = shared
    out = []
    for s, n, a, t in zip(idx, names, addrs, per_query):
        un = r.union(n, a)
        y = np.isin(un.u, t)
        rows, w = sample_rows(y, r.hand_score(un), s, neg_top, neg_rand)
        out.append((r.features(un)[rows].astype(np.float32), y[rows], w))
    return out


def run_stage1_train(args):
    """CLI `stage1-train`: writes stage1/model_v1.json and metrics stage1_train.json."""
    t0 = time.time()
    lg = io_utils.get_logger("stage1-train")
    cfg = io_utils.load_config()
    p = cfg["stage1_model"]
    s1 = splits.load_query_sets()["S1R_TRAIN"]
    truth = truth_lists(s1)
    Xs, ys, ws = [], [], []
    with retrieve.QueryRunner("train", cfg) as qr:
        for chunk in qr.map(cfg, None, s1, _train_chunk, per_query=truth,
                            shared=(p["neg_top"], p["neg_rand"])):
            for X, y, w in chunk:
                Xs.append(X)
                ys.append(y)
                ws.append(w)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    del Xs, ys, ws
    lg.info("training rows %d (positives %d) from %d queries", y.size, int(y.sum()), s1.size)
    mean = X.mean(axis=0, dtype=np.float64)
    std = X.std(axis=0, dtype=np.float64)
    std[std == 0] = 1.0
    Z = (X - mean) / std
    del X
    lr = LogisticRegression(C=p["C"], max_iter=p["max_iter"], random_state=0)
    lr.fit(Z, y.astype(np.int8), sample_weight=w)
    names = retrieve.feature_names(retrieve.kept_families(cfg))
    doc = {"features": names, "families": retrieve.kept_families(cfg),
           "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
           "mean": mean.tolist(), "std": std.tolist(), "config_hash": io_utils.config_hash(cfg),
           "stage1_config": retrieve.stage1_config(cfg)}
    io_utils.atomic_write_json(retrieve.model_path(), doc)
    payload = {"n_rows": int(y.size), "n_positive_rows": int(y.sum()),
               "sum_weights": float(w.sum()), "n_iter": int(lr.n_iter_[0]),
               "converged": bool(lr.n_iter_[0] < p["max_iter"]),
               "coef": dict(zip(names, doc["coef"])), "intercept": doc["intercept"],
               "model_file": str(retrieve.model_path()), "wall_s": round(time.time() - t0, 1),
               "peak_rss_mb": round(io_utils.peak_rss_mb(), 1)}
    io_utils.write_metrics("stage1_train", payload, s1.size, sum(t.size for t in truth), cfg)
    lg.info("stage1-train done: %s", {k: payload[k] for k in ("n_rows", "n_iter", "wall_s")})
