"""XGBoost + LightGBM blend on top of full_ce (ALTERNATIVE_APPROACHES 1.6).

Same data, S1 split, and exclusions as train_final_comp.py full_ce (seed 20260926, MATCH_ALL minus DEV_REST_180K,
71 features incl. the ce:ce_rec group). Trains one XGBoost model (hist, lossguide, 63 leaves, eta 0.05 = the
LightGBM block's shape) on the fit rows with early stopping on the early-stop S1, then on the calibration S1
scores three raw predictors with one identical procedure (isotonic + threshold/expected_f + one-owner search):
  lgb   = the saved full_ce booster
  xgb   = the new XGBoost booster
  blend = (1 - W) * lgb + W * xgb, W fixed at 0.5 before the run (the W sweep is reported, not used to choose)
Paired deltas are vs lgb re-scored by the same procedure, so the comparison is procedure-matched.

Saves work/match/runs/<RUN>/models/ = full_ce's model_fold0.txt + xgb_fold0.ubj + matcher.json with
"xgb_blend": {"files": [...], "weight": W} and the blend's calibration/decision; `exp-predict --run <RUN>` reads it.

Run from the repository root:
  PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/train_xgb_blend.py
"""
import os
import shutil
import sys
import time

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from ber import exp, io_utils, match
from dev_rest_full import pick_remaining_slice

RUN = sys.argv[1] if len(sys.argv) > 1 else "full_ce_xgb"
PARENT = "full_ce"
SEED = 20260926
ES_FRAC, CAL_FRAC = 0.05, 0.10
W = 0.5
NUM_BOOST_ROUND = int(os.environ.get("NUM_BOOST_ROUND", "12000"))


def fit_decision(p_raw, y_c, s1_c, pool_c, scorer, dcfg):
    """Trainer's procedure: isotonic on the slice, threshold sweep, expected_f and one-owner adopted on paired gain."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_raw, y_c)
    calib = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    p_cal = match.calibrate(p_raw, calib)
    grid = match.threshold_grid(dcfg)
    cands = {}
    for owner in (False, True):
        pc = match.one_owner(pool_c, p_cal) if owner else p_cal
        sweep = {float(t): scorer.f05(pc >= t) for t in grid}
        t_best = max(sweep, key=sweep.get)
        for dec in ({"rule": "threshold", "threshold": t_best, "one_owner": owner},
                    {"rule": "expected_f", "threshold": None, "one_owner": owner}):
            cands[(dec["rule"], owner)] = (dec, scorer.per_entity(match.apply_decision(dec, s1_c, pool_c, p_cal))[0])
    ef = match.paired_delta(cands[("expected_f", False)][1], cands[("threshold", False)][1])
    rule = "expected_f" if match.adopt(ef, dcfg["expected_f_min_gain"]) else "threshold"
    oo = match.paired_delta(cands[(rule, True)][1], cands[(rule, False)][1])
    owner = match.adopt(oo, dcfg["one_owner_min_gain"])
    decision, per_entity = cands[(rule, owner)]
    return calib, decision, per_entity


def main():
    t0 = time.time()
    cfg, mcfg = io_utils.load_config(), match.load_match_config()
    fp = match.bucketing_fingerprint(cfg)
    pdir = match.runs_dir(PARENT) / "models"
    pdoc = io_utils.read_json(pdir / "matcher.json")
    groups, excl = pdoc["extra_groups"], ["DEV_REST_180K"]
    names, feats = pdoc["feature_names"], pdoc["features_used"]
    cols = [names.index(f) for f in feats]

    s1_rest, s1_180k = pick_remaining_slice()
    sets = [("MATCH_TRAIN", match.set_queries("MATCH_TRAIN")), ("MATCH_VALID", match.set_queries("MATCH_VALID")),
            ("DEV_REST_180K", s1_180k), ("DEV_REST_REST", s1_rest)]
    Xs, s1s, pools, qs = [], [], [], []
    for name, q in sets:
        qs.append(q)
        if name in excl:
            continue
        X, s1, pool, meta = match.load_features(name, fp)
        if meta["features"] != match.FEATURES or meta.get("limited"):
            raise RuntimeError(f"{name}: features stale or --limit build")
        G = np.hstack([exp.load_group(name, g, X, s1, pool) for g in groups])
        Xs.append(np.hstack([X, G])[:, cols].astype(np.float32)); s1s.append(s1); pools.append(pool)
        print(f">>> {name}: {s1.size} rows ({time.time()-t0:.0f}s)", flush=True)
    X, s1, pool = np.concatenate(Xs), np.concatenate(s1s), np.concatenate(pools)
    queries = np.sort(np.concatenate(qs)).astype(np.int32)
    del Xs
    y = match.pair_labels(s1, pool)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(queries)
    n_cal, n_es = int(CAL_FRAC * queries.size), int(ES_FRAC * queries.size)
    q_cal, q_es = np.sort(perm[:n_cal]), np.sort(perm[n_cal:n_cal + n_es])
    present = np.unique(s1)
    q_cal, q_es = q_cal[np.isin(q_cal, present, assume_unique=True)], q_es[np.isin(q_es, present, assume_unique=True)]
    in_cal, in_es = np.isin(s1, q_cal), np.isin(s1, q_es)
    fit = ~(in_cal | in_es)
    print(f">>> split: fit {fit.sum()} rows, early-stop {q_es.size} S1, calibration {q_cal.size} S1 "
          f"(full_ce: {pdoc.get('train_set')})", flush=True)

    lp = mcfg["lgb"]
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
              "grow_policy": "lossguide", "max_leaves": lp["num_leaves"], "max_depth": 0, "eta": lp["learning_rate"],
              "subsample": lp["bagging_fraction"], "colsample_bytree": lp["feature_fraction"],
              "reg_lambda": lp["lambda_l2"], "max_bin": lp["max_bin"], "min_child_weight": 1.0,
              "seed": 42, "nthread": io_utils.n_workers(cfg)}
    dfit = xgb.QuantileDMatrix(X[fit], label=y[fit], feature_names=feats, max_bin=lp["max_bin"])
    des = xgb.QuantileDMatrix(X[in_es], label=y[in_es], feature_names=feats, ref=dfit, max_bin=lp["max_bin"])
    esr = mcfg["early_stopping_rounds"]
    bst = xgb.train(params, dfit, num_boost_round=NUM_BOOST_ROUND, evals=[(des, "early_stop")],
                    early_stopping_rounds=esr, verbose_eval=250)
    del dfit, des
    best = int(bst.best_iteration) + 1  # xgboost best_iteration is 0-based
    truncated = best > NUM_BOOST_ROUND - esr
    print(f">>> xgb trained: best_iteration {best} of cap {NUM_BOOST_ROUND}"
          f"{'  ** TRUNCATED: did not converge **' if truncated else ' (converged)'} ({time.time()-t0:.0f}s)", flush=True)

    Xc, y_c, s1_c, pool_c = X[in_cal], y[in_cal], s1[in_cal], pool[in_cal]
    del X
    p_x = bst.predict(xgb.DMatrix(Xc, feature_names=feats), iteration_range=(0, best))
    p_l = lgb.Booster(model_file=str(pdir / "model_fold0.txt")).predict(Xc)
    from sklearn.metrics import roc_auc_score
    auc = {"lgb": float(roc_auc_score(y_c, p_l)), "xgb": float(roc_auc_score(y_c, p_x)),
           "corr_raw": float(np.corrcoef(p_l, p_x)[0, 1])}
    scorer = match.Scorer(q_cal, s1_c, y_c, match.true_counts(q_cal))
    dcfg = mcfg["decision"]
    res = {}
    for key, w in (("lgb", 0.0), ("xgb", 1.0), ("w0.25", 0.25), ("w0.5", 0.5), ("w0.75", 0.75)):
        res[key] = fit_decision((1 - w) * p_l + w * p_x, y_c, s1_c, pool_c, scorer, dcfg)
        print(f">>> {key}: F0.5 {res[key][2].mean():.5f} decision {res[key][1]}", flush=True)
    fe_l = res["lgb"][2]
    deltas = {k: match.paired_delta(v[2], fe_l, n_boot=2000) for k, v in res.items() if k != "lgb"}
    for k, d in deltas.items():
        print(f">>> {k} - lgb: {d}", flush=True)
    live_f = float(fe_l.mean())
    print(f">>> lgb re-scored {live_f:.5f} vs full_ce recorded {pdoc['oof_f05']:.5f}; AUC {auc}", flush=True)

    rd = match.runs_dir(RUN) / "models"
    rd.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdir / "model_fold0.txt", rd / "model_fold0.txt")
    bst = bst[:best]
    bst.save_model(str(rd / "xgb_fold0.ubj"))
    calib, decision, fe_b = res[f"w{W}"]
    doc = dict(pdoc, run=RUN, calibration=calib, decision=decision, oof_f05=float(fe_b.mean()),
               xgb_blend={"files": ["xgb_fold0.ubj"], "weight": W, "xgb_best_iteration": best,
                          "xgb_params": params, "parent": PARENT},
               created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    io_utils.atomic_write_json(rd / "matcher.json", doc)
    io_utils.write_metrics(f"match_train_{RUN}", {
        "run": RUN, "parent": PARENT, "weight": W, "n_cal_s1": int(q_cal.size), "n_fit_rows": int(fit.sum()),
        "xgb_best_iteration": best, "num_boost_round": NUM_BOOST_ROUND, "truncated": bool(truncated),
        "auc_cal": auc, "f05": {k: float(v[2].mean()) for k, v in res.items()},
        "decisions": {k: v[1] for k, v in res.items()}, "vs_lgb_same_slice": deltas,
        "full_ce_recorded_f05": pdoc["oof_f05"], "xgb_params": params, "wall_s": round(time.time() - t0, 1)},
        int(q_cal.size), int(scorer.n_true.sum()), cfg)
    print(f">>> saved {rd} (NOT copied to the live dir). total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
