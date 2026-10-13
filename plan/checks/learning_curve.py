"""Learning curve (VALIDATION_PLAN.md §10 step 8): does OOF F0.5 keep rising as MATCH_TRAIN grows?

Trains x1_strsim's frozen 68-feature set on random subsets of MATCH_TRAIN (45k, 90k S1), same 5-fold
GroupKFold / LightGBM params / isotonic calibration / decision-rule search as `match-train`, but reading
already-built features and never touching work/match/models/ or configs/match.json. The 180,000-S1 point
is not rerun -- it is x1_strsim itself (artifacts/metrics/match_train_x1_strsim.json).

Run from the repository root:
  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/learning_curve.py
"""
import time

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ber import io_utils, match

SUBSET_SIZES = (45_000, 90_000)
SEED = 42


def subset_curve_point(n, X_all, s1_all, pool_all, y_all, queries_all, mcfg, threads):
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    sub_q = np.sort(rng.choice(queries_all, size=n, replace=False))
    row_mask = np.isin(s1_all, sub_q)
    X, s1, pool, y = X_all[row_mask], s1_all[row_mask], pool_all[row_mask], y_all[row_mask]

    oof = np.zeros(y.size, dtype=np.float64)
    gkf = GroupKFold(n_splits=mcfg["n_folds"], shuffle=True, random_state=mcfg["fold_seed"])
    fold_info = []
    for k, (tr, te) in enumerate(gkf.split(X, y, s1)):
        gss = GroupShuffleSplit(n_splits=1, test_size=mcfg["early_stop_frac"],
                                random_state=mcfg["fold_seed"] + k)
        fit_i, es_i = next(gss.split(tr, groups=s1[tr]))
        fit_i, es_i = tr[fit_i], tr[es_i]
        dfit = lgb.Dataset(X[fit_i], label=y[fit_i], feature_name=match.FEATURES, free_raw_data=False)
        des = lgb.Dataset(X[es_i], label=y[es_i], reference=dfit, free_raw_data=False)
        params = {k2: v for k2, v in mcfg["lgb"].items()
                  if k2 not in ("num_boost_round", "early_stopping_rounds", "holdout_frac")}
        params["num_threads"] = threads
        booster = lgb.train(params, dfit, num_boost_round=mcfg["num_boost_round"],
                            valid_sets=[des], valid_names=["early_stop"],
                            callbacks=[lgb.early_stopping(mcfg["early_stopping_rounds"], verbose=False),
                                      lgb.log_evaluation(period=0)])
        oof[te] = booster.predict(X[te], num_iteration=booster.best_iteration)
        fold_info.append({"fold": k, "best_iteration": int(booster.best_iteration),
                          "n_fit_rows": int(fit_i.size)})

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, y)
    p_cal = np.interp(oof, iso.X_thresholds_, iso.y_thresholds_).astype(np.float32)

    scorer = match.Scorer(sub_q, s1, y, match.true_counts(sub_q))
    dcfg = mcfg["decision"]
    grid = match.threshold_grid(dcfg)
    candidates = {}
    for owner in (False, True):
        pc = match.one_owner(pool, p_cal) if owner else p_cal
        sweep = {float(t): scorer.f05(pc >= t) for t in grid}
        t_best = max(sweep, key=sweep.get)
        candidates[owner] = (t_best, sweep[t_best])
    use_owner = candidates[True][1] > candidates[False][1]
    t_best, f05 = candidates[use_owner]

    return {"n_queries": int(n), "n_rows": int(y.size), "oof_f05": float(f05),
            "threshold": t_best, "one_owner": bool(use_owner),
            "folds": fold_info, "wall_s": round(time.time() - t0, 1)}


def main():
    cfg, mcfg = io_utils.load_config(), match.load_match_config()
    fp = match.bucketing_fingerprint(cfg)
    X_all, s1_all, pool_all, meta = match.load_features("MATCH_TRAIN", fp)
    y_all = match.pair_labels(s1_all, pool_all)
    queries_all = match.set_queries("MATCH_TRAIN")
    threads = io_utils.n_workers(cfg)
    print(f"MATCH_TRAIN loaded: {queries_all.size} S1, {y_all.size} rows, {X_all.shape[1]} features")

    points = []
    for n in SUBSET_SIZES:
        print(f">>> training on {n} S1 ...")
        pt = subset_curve_point(n, X_all, s1_all, pool_all, y_all, queries_all, mcfg, threads)
        print(f"    n={n}: OOF F0.5 {pt['oof_f05']:.5f} (t={pt['threshold']}, one_owner={pt['one_owner']}, "
              f"{pt['wall_s']}s)")
        points.append(pt)

    x1 = io_utils.read_json(io_utils.paths()["metrics_root"] / "match_train_x1_strsim.json")
    x1 = x1.get("payload", x1)
    points.append({"n_queries": 180_000, "n_rows": x1["n_rows"], "oof_f05": x1["oof"]["f05"],
                   "threshold": x1["decision"].get("threshold"), "one_owner": x1["decision"]["one_owner"],
                   "wall_s": x1["wall_s"], "source": "match_train_x1_strsim.json (not rerun)"})

    doc = {"points": points, "note": "45k/90k random subsets of MATCH_TRAIN S1 (seed 42, unstratified); "
           "180k point is x1_strsim itself, not rerun. Same 5-fold GroupKFold/LightGBM params/isotonic "
           "calibration/threshold+one_owner search as match-train. No feature rebuild, no state written "
           "to work/match/models/ or configs/match.json."}
    path = io_utils.write_metrics("learning_curve", doc, 0, 0, cfg)
    print(f"\nwrote {path}")
    print(f"{'n_S1':>10} {'OOF F0.5':>10} {'gain/doubling':>15}")
    prev = None
    for pt in points:
        gain = f"{pt['oof_f05'] - prev:+.5f}" if prev is not None else "-"
        print(f"{pt['n_queries']:>10} {pt['oof_f05']:>10.5f} {gain:>15}")
        prev = pt["oof_f05"]


if __name__ == "__main__":
    main()
