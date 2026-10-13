"""X5c LOCO (leave-one-country-out) diagnostic (plan/FEATURE_EXPERIMENTS.md X5c). Report only -- no
adoption decision. Uses the x1_strsim feature set (already on disk for MATCH_TRAIN, no rebuild).

For each direction (train US / score India, train India / score US):
  - restrict MATCH_TRAIN rows to S1 whose own country_raw is the training country;
  - run the SAME 5-fold GroupKFold(shuffle=True, seed=mcfg["fold_seed"]) scheme as run_match_train,
    but over that restricted row set only, producing OOF predictions and an isotonic calibration fit on
    that restricted OOF (not x1's calibration -- this direction's own fold models see different data);
  - average the 5 fold models' raw predictions (ber.match.predict_raw) over ALL rows of the OTHER
    country's S1 and calibrate with the fit above;
  - apply x1_strsim's decision rule (matcher.json "decision": threshold + one_owner) unchanged;
  - score per-entity F0.5 over every S1 of the scored country (Scorer, so 0-candidate S1 count too);
  - compare, paired by S1, to x1_strsim's OOF per-entity F0.5 on the SAME S1 subset
    (work/match/runs/x1_strsim/{queries.npy,OOF_per_entity.npy}).

Country is used ONLY to split which S1's rows train each direction's folds -- never as a model feature
(the feature matrix and column list are exactly used_features(mcfg) with country3/pool_indic etc., no
country_raw column is added). Read-only w.r.t. work/match/models/ (the live matcher); each direction's
folds are written under work/match/runs/loco_<train_country>/models/.

Run: PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/loco_matcher.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "submission/code/business_entity_resolution/src"))

import lightgbm as lgb  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.model_selection import GroupKFold, GroupShuffleSplit  # noqa: E402

from ber import io_utils, match  # noqa: E402

DIRECTIONS = [("US", "India"), ("India", "US")]


def _country_by_s1():
    """country_raw indexed by s1_idx (row order == s1_idx, per ber.match._error_frame's use of the same
    file with the same indexing)."""
    rec = pd.read_parquet(io_utils.work_path("records", "train_s1.parquet"), columns=["country_raw"])
    return rec["country_raw"].to_numpy()


def _train_folds(X, y, s1, feats, mcfg, threads, run_name, lg):
    """5-fold GroupKFold(shuffle=True, seed=mcfg['fold_seed']) over the rows given (already restricted to
    the training country), same early-stopping / params as run_match_train. Returns (models_dir, calib,
    oof, folds_report)."""
    models_dir = io_utils.work_path("match", "runs", run_name, "models", "x").parent
    models_dir.mkdir(parents=True, exist_ok=True)
    for old in models_dir.glob("model_fold*.txt"):
        old.unlink()
    gkf = GroupKFold(n_splits=mcfg["n_folds"], shuffle=True, random_state=mcfg["fold_seed"])
    oof = np.zeros(y.size, dtype=np.float64)
    folds = []
    for k, (tr, te) in enumerate(gkf.split(X, y, s1)):
        gss = GroupShuffleSplit(n_splits=1, test_size=mcfg["early_stop_frac"],
                                random_state=mcfg["fold_seed"] + k)
        fit_i, es_i = next(gss.split(tr, groups=s1[tr]))
        fit_i, es_i = tr[fit_i], tr[es_i]
        dfit = lgb.Dataset(X[fit_i], label=y[fit_i], feature_name=feats, free_raw_data=False)
        des = lgb.Dataset(X[es_i], label=y[es_i], reference=dfit, free_raw_data=False)
        booster = lgb.train(match._lgb_params(mcfg, threads, feats), dfit,
                            num_boost_round=mcfg["num_boost_round"], valid_sets=[des],
                            valid_names=["early_stop"],
                            callbacks=[lgb.early_stopping(mcfg["early_stopping_rounds"], verbose=False),
                                       lgb.log_evaluation(period=0)])
        oof[te] = booster.predict(X[te], num_iteration=booster.best_iteration)
        booster.save_model(str(models_dir / f"model_fold{k}.txt"), num_iteration=booster.best_iteration)
        folds.append({"fold": k, "n_fit_rows": int(fit_i.size), "n_es_rows": int(es_i.size),
                      "best_iteration": int(booster.best_iteration)})
        lg.info("  [%s] fold %d: best_iteration %d", run_name, k, booster.best_iteration)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, y)
    calib = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    return models_dir, calib, oof, folds


def _run_direction(train_c, score_c, X, y, s1, pool, feats, mcfg, threads, queries, s1_country,
                   decision, x1_queries, x1_per_entity, lg):
    t0 = time.time()
    train_mask = s1_country[s1] == train_c
    score_mask = s1_country[s1] == score_c
    tr_idx = np.flatnonzero(train_mask)
    models_dir, calib, oof, folds = _train_folds(X[tr_idx], y[tr_idx], s1[tr_idx], feats, mcfg, threads,
                                                 f"loco_{train_c.lower()}", lg)

    sc_idx = np.flatnonzero(score_mask)
    p_raw = match.predict_raw(X[sc_idx], models_dir, mcfg["n_folds"])
    p_cal = match.calibrate(p_raw, calib)
    keep = match.apply_decision(decision, s1[sc_idx], pool[sc_idx], p_cal)

    score_queries = queries[s1_country[queries] == score_c]
    scorer = match.Scorer(score_queries, s1[sc_idx], y[sc_idx], match.true_counts(score_queries))
    report, per_entity_loco = scorer.report(keep, p_cal)

    pos = np.searchsorted(x1_queries, score_queries)
    if not np.array_equal(x1_queries[pos], score_queries):
        raise RuntimeError("x1_strsim queries.npy does not cover this direction's scored S1")
    per_entity_x1 = x1_per_entity[pos]
    delta = match.paired_delta(per_entity_loco, per_entity_x1)

    doc = {
        "train_country": train_c, "score_country": score_c,
        "n_s1_train": int(np.unique(s1[tr_idx]).size), "n_s1_score": int(score_queries.size),
        "f05_loco": report["f05"], "f05_all_country_x1_strsim": float(per_entity_x1.mean()),
        "paired_delta_loco_minus_x1": delta,
        "loco_eval": report, "folds": folds, "wall_s": round(time.time() - t0, 1),
    }
    lg.info("direction train=%s score=%s: F0.5 LOCO %.5f vs x1_strsim all-country %.5f, delta %s (%.0fs)",
            train_c, score_c, report["f05"], per_entity_x1.mean(), delta, time.time() - t0)
    return doc


def main():
    lg = io_utils.get_logger("loco-matcher")
    cfg = io_utils.load_config()
    mcfg = match.load_match_config()
    if mcfg.get("objective_mode") or mcfg.get("monotone_increasing"):
        raise RuntimeError("configs/match.json is not at the x1_strsim baseline (objective_mode / "
                           "monotone_increasing set); LOCO must run against the frozen x1_strsim features")
    fp = match.bucketing_fingerprint(cfg)
    X_all, s1, pool, meta = match.load_features("MATCH_TRAIN", fp)
    feats = match.used_features(mcfg)
    X = X_all[:, [match._COL[f] for f in feats]]
    if feats != io_utils.read_json(io_utils.work_path("match", "runs", "x1_strsim", "models", "matcher.json"))["features_used"]:
        raise RuntimeError("current used_features() differs from the frozen x1_strsim feature set")
    y = match.pair_labels(s1, pool)
    queries = match.set_queries("MATCH_TRAIN")
    s1_country = _country_by_s1()
    threads = io_utils.n_workers(cfg)

    x1_doc = io_utils.read_json(io_utils.work_path("match", "runs", "x1_strsim", "models", "matcher.json"))
    decision = x1_doc["decision"]
    x1_queries = np.load(io_utils.work_path("match", "runs", "x1_strsim", "queries.npy"))
    x1_per_entity = np.load(io_utils.work_path("match", "runs", "x1_strsim", "OOF_per_entity.npy"))

    lg.info("LOCO: %d MATCH_TRAIN rows, %d queries, %d features (x1_strsim schema); decision %s",
            y.size, queries.size, len(feats), decision)

    results = [_run_direction(tc, sc, X, y, s1, pool, feats, mcfg, threads, queries, s1_country, decision,
                              x1_queries, x1_per_entity, lg)
              for tc, sc in DIRECTIONS]

    payload = {"note": "report only, no adoption decision (plan/FEATURE_EXPERIMENTS.md X5c); country used "
                       "only to split MATCH_TRAIN rows by the training S1's own country, never as a "
                       "feature", "baseline": "x1_strsim", "directions": results}
    io_utils.write_metrics("loco_x1_strsim", payload, int(queries.size), 0, cfg)
    lg.info("loco-matcher done")


if __name__ == "__main__":
    main()
