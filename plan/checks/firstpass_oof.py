"""First-pass out-of-fold decisions on the full S9 training rows (MATCH_TRAIN + MATCH_VALID + DEV_REST_REST fold 1), so
stacked features (grp) can be built for every training row without in-sample leakage. Laptop, LightGBM only.

  PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/firstpass_oof.py

Model = S9's recipe (68 live features + cex_p:ce_rec_c0), 5-fold GroupKFold by S1 over all rows, early stop on 10% of
each fold's training S1, isotonic on the OOF, decision threshold 0.72 + one-owner (S9's rule). Writes
work/retrieval/firstpass_<SET>.parquet (s1_idx, pool_idx, p_raw, accepted) for MATCH_TRAIN, MATCH_VALID, DEV_REST_REST
(fold-1 rows only) and artifacts/metrics/firstpass_oof.json (folds, best_iteration, OOF F0.5 is not computed here).
"""
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber import exp, io_utils, match

SETS = ["MATCH_TRAIN", "MATCH_VALID", "DEV_REST_REST"]
GROUPS = ["cex_p:ce_rec_c0"]


def fold_of(s1):
    return ((s1.astype(np.uint64) * np.uint64(0x9E3779B1)) % np.uint64(2 ** 32)) >> np.uint64(31)


def main():
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    t0 = time.time()
    cfg, mcfg = io_utils.load_config(), match.load_match_config()
    fp = match.bucketing_fingerprint(cfg)
    base = match.used_features(mcfg)
    Xs, S, P, tag = [], [], [], []
    for name in SETS:
        X, s1, pool, _ = match.load_features(name, fp)
        if name == "DEV_REST_REST":
            k = fold_of(s1) == 1
            X, s1, pool = X[k], s1[k], pool[k]
        G = exp.load_group(name, GROUPS[0], X, s1, pool)
        names = list(match.FEATURES) + exp.group_spec(GROUPS[0])[0]
        XG = np.hstack([X, G])
        cols = [names.index(f) for f in base + exp.group_spec(GROUPS[0])[0]]
        Xs.append(XG[:, cols]); S.append(s1); P.append(pool); tag += [name] * s1.size
        print(f"{name}: {s1.size:,} rows ({time.time() - t0:.0f}s)", flush=True)
    X, s1, pool, tag = np.vstack(Xs), np.concatenate(S), np.concatenate(P), np.array(tag)
    y = match.pair_labels(s1, pool)
    params = dict(mcfg["lgb"], num_threads=io_utils.n_workers(cfg))
    oof, folds = np.zeros(y.size), []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, s1)):
        fi, ei = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42 + k).split(tr, groups=s1[tr]))
        d = lgb.Dataset(X[tr[fi]], y[tr[fi]], free_raw_data=True)
        b = lgb.train(params, d, num_boost_round=12000, valid_sets=[lgb.Dataset(X[tr[ei]], y[tr[ei]], reference=d)],
                      callbacks=[lgb.early_stopping(mcfg["early_stopping_rounds"], verbose=False)])
        oof[te] = b.predict(X[te], num_iteration=b.best_iteration)
        folds.append({"fold": k, "best_iteration": int(b.best_iteration), "cap": 12000,
                      "truncated": bool(b.best_iteration > 12000 - mcfg["early_stopping_rounds"])})
        print(f"fold {k}: best_iteration {b.best_iteration} ({time.time() - t0:.0f}s)", flush=True)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(oof, y)
    p_cal = iso.predict(oof)
    acc = match.apply_decision({"rule": "threshold", "threshold": 0.72, "one_owner": True}, s1, pool, p_cal)
    for name in SETS:
        m = tag == name
        pq.write_table(pa.table({"s1_idx": pa.array(s1[m]), "pool_idx": pa.array(pool[m]),
                                 "p_raw": pa.array(oof[m].astype(np.float32)), "accepted": pa.array(acc[m])}),
                       f"work/retrieval/firstpass_{name}.parquet", compression="zstd")
    io_utils.write_metrics("firstpass_oof", {"folds": folds, "rows": int(y.size), "accepted": int(acc.sum()),
                                             "wall_s": round(time.time() - t0)}, int(np.unique(s1).size), int(y.sum()))
    print(f"done: {acc.sum():,} accepted of {y.size:,} rows ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
