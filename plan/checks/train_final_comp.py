"""Submission-3 candidate: train_final_single.py (same S1 split, seed, params, cap) plus the adopted M005
pool-contention group `comp`. Saved in exp-run format (work/match/runs/full_comp/models/, n_folds=1,
extra_groups=["comp"]) so `exp-predict --run full_comp` builds the test submission. It does NOT replace the
live model in work/match/models/; it also scores full_single on the same calibration slice for a paired delta.

Original notes (train_final_single.py):

Why one model, not 5-fold: at 1.65M S1 a single held-out slice is large enough for calibration and the
decision rule, and one model on ~85% of S1 trains ~5x faster than 5 fold models on 80% each.

Split (by S1, seeded): fit ~85%, early-stop ~5%, calibration/decision ~10% (~165k S1). Early stopping and
calibration use different S1, so the calibration map and threshold are fit on predictions the stopping
rule never saw. Hyperparameters = configs/match.json lgb block (unchanged; tuning is done separately).
Round cap 12000 (plan/ROUND_CAP_EVIDENCE.md: every 5-fold MATCH_ALL fold hit 4000); truncation is checked
and reported, never silently accepted.

The previous live model (x1_strsim, 5 folds) stays in work/match/runs/x1_strsim/models/. This script writes
work/match/runs/full_single/models/ and then copies it into work/match/models/ (the live dir).

Run from the repository root:
  PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/train_final_single.py
"""
import os
import sys
import time

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from ber import exp, io_utils, match
from dev_rest_full import pick_remaining_slice

# usage: train_final_comp.py [RUN] [GROUP ...]   (default: full_comp comp)
# EXCLUDE_SETS=DEV_REST_180K drops that set's rows (e.g. the cross-encoder's training S1, where p_ce is in-sample);
# its S1 still enter the seeded permutation, so the fit / early-stop / calibration split of every other S1 is
# unchanged and the calibration slice stays out-of-sample for full_single too.
RUN = sys.argv[1] if len(sys.argv) > 1 else "full_comp"
GROUPS = sys.argv[2:] if len(sys.argv) > 2 else ["comp"]
SEED = 20260926
ES_FRAC, CAL_FRAC = 0.05, 0.10
NUM_BOOST_ROUND = int(os.environ.get("NUM_BOOST_ROUND", "12000"))  # override only for smoke runs
EXCLUDE = [s for s in os.environ.get("EXCLUDE_SETS", "").split(",") if s]
# REST_FOLD=k keeps only DEV_REST_REST rows whose S1 is in cross-encoder fold k (ce/train_continue.py fold_of):
# the fold-(1-k) adapter trained on the other half, so its p_ce is in-sample there.
REST_FOLD = os.environ.get("REST_FOLD")


def main():
    t0 = time.time()
    cfg, mcfg = io_utils.load_config(), match.load_match_config()
    lg = io_utils.get_logger("train_final_single")
    fp = match.bucketing_fingerprint(cfg)
    base = match.used_features(mcfg)
    names = list(match.FEATURES) + [c for g in GROUPS for c in exp.group_spec(g)[0]]
    feats = base + [c for g in GROUPS for c in exp.group_spec(g)[0]]
    cols = [names.index(f) for f in feats]
    base_cols = [feats.index(f) for f in base]

    s1_rest, s1_180k = pick_remaining_slice()
    sets = [("MATCH_TRAIN", match.set_queries("MATCH_TRAIN")), ("MATCH_VALID", match.set_queries("MATCH_VALID")),
            ("DEV_REST_180K", s1_180k), ("DEV_REST_REST", s1_rest)]
    Xs, s1s, pools, qs = [], [], [], []
    for name, q in sets:
        qs.append(q)
        if name in EXCLUDE:
            print(f">>> {name}: excluded ({q.size} S1 kept only in the split permutation)", flush=True)
            continue
        X, s1, pool, meta = match.load_features(name, fp)
        if meta["features"] != match.FEATURES or meta.get("limited"):
            raise RuntimeError(f"{name}: features stale or --limit build")
        if name == "DEV_REST_REST" and REST_FOLD is not None:
            k = ((s1.astype(np.uint64) * np.uint64(0x9E3779B1)) % np.uint64(2**32)) >> np.uint64(31) == int(REST_FOLD)
            X, s1, pool = X[k], s1[k], pool[k]
            print(f">>> DEV_REST_REST: fold {REST_FOLD} only, {np.unique(s1).size} S1", flush=True)
        G = np.hstack([exp.load_group(name, g, X, s1, pool) for g in GROUPS])
        Xs.append(np.hstack([X, G])[:, cols]); s1s.append(s1); pools.append(pool)
        print(f">>> {name}: {s1.size} rows with groups ({time.time()-t0:.0f}s)", flush=True)
    X, s1, pool = np.concatenate(Xs), np.concatenate(s1s), np.concatenate(pools)
    queries = np.sort(np.concatenate(qs)).astype(np.int32)
    del Xs
    y = match.pair_labels(s1, pool)
    print(f">>> MATCH_ALL: {queries.size} S1, {y.size} rows, {len(feats)} features ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(queries)
    n_cal, n_es = int(CAL_FRAC * queries.size), int(ES_FRAC * queries.size)
    q_cal, q_es = np.sort(perm[:n_cal]), np.sort(perm[n_cal:n_cal + n_es])
    if EXCLUDE or REST_FOLD is not None:
        present = np.unique(s1)
        q_cal, q_es = q_cal[np.isin(q_cal, present, assume_unique=True)], q_es[np.isin(q_es, present, assume_unique=True)]
        queries = queries[np.isin(queries, present, assume_unique=True)]
    in_cal, in_es = np.isin(s1, q_cal), np.isin(s1, q_es)
    fit = ~(in_cal | in_es)
    print(f">>> split: fit {fit.sum()} rows, early-stop {in_es.sum()} rows ({q_es.size} S1), "
          f"calibration {in_cal.sum()} rows ({q_cal.size} S1)", flush=True)

    params = dict(mcfg["lgb"], num_threads=io_utils.n_workers(cfg))
    dfit = lgb.Dataset(X[fit], label=y[fit], feature_name=feats, free_raw_data=True)
    des = lgb.Dataset(X[in_es], label=y[in_es], reference=dfit, free_raw_data=True)
    booster = lgb.train(params, dfit, num_boost_round=NUM_BOOST_ROUND, valid_sets=[des], valid_names=["early_stop"],
                        callbacks=[lgb.early_stopping(mcfg["early_stopping_rounds"], verbose=False),
                                   lgb.log_evaluation(period=500)])
    best = int(booster.best_iteration)
    truncated = best > NUM_BOOST_ROUND - mcfg["early_stopping_rounds"]
    print(f">>> trained: best_iteration {best} of cap {NUM_BOOST_ROUND}"
          f"{'  ** TRUNCATED: did not converge **' if truncated else ' (converged)'} ({time.time()-t0:.0f}s)", flush=True)

    # calibration + decision rule on the held-out calibration S1 (never used for fitting or stopping)
    p_raw = booster.predict(X[in_cal], num_iteration=best)
    y_c, s1_c, pool_c = y[in_cal], s1[in_cal], pool[in_cal]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_raw, y_c)
    calib = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    p_cal = match.calibrate(p_raw, calib)
    scorer = match.Scorer(q_cal, s1_c, y_c, match.true_counts(q_cal))
    dcfg = mcfg["decision"]
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
    f05 = float(per_entity.mean())
    p_rr = X[in_cal][:, feats.index("p_rerank")]
    floor = {float(t): scorer.f05(p_rr >= t) for t in grid}
    floor_t = max(floor, key=floor.get)
    print(f">>> held-out F0.5 {f05:.5f} on {q_cal.size} S1 (decision {decision}; p_rerank floor {floor[floor_t]:.5f})",
          flush=True)

    # paired comparison with the live full_single model on the SAME calibration S1 (its own calibration + rule);
    # reporting only, skipped when that model is absent (it does not change this run's model or decision)
    fs_dir = match.runs_dir("full_single") / "models"
    vs_fs, fs_mean = None, None
    if (fs_dir / "matcher.json").exists():
        fs_doc = io_utils.read_json(fs_dir / "matcher.json")
        if fs_doc["features_used"] != base:
            raise RuntimeError("full_single feature list differs from used_features()")
        fs_b = lgb.Booster(model_file=str(fs_dir / "model_fold0.txt"))
        fs_p = match.calibrate(fs_b.predict(X[in_cal][:, base_cols]), fs_doc["calibration"])
        fs_f = scorer.per_entity(match.apply_decision(fs_doc["decision"], s1_c, pool_c, fs_p))[0]
        vs_fs, fs_mean = match.paired_delta(per_entity, fs_f, n_boot=2000), float(fs_f.mean())
        print(f">>> full_single on the same slice {fs_mean:.5f}; {RUN} - full_single {vs_fs}", flush=True)
    else:
        print(f">>> {fs_dir} absent: paired comparison with full_single skipped", flush=True)

    rd = match.runs_dir(RUN) / "models"
    rd.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(rd / "model_fold0.txt"), num_iteration=best)
    doc = {"features": match.FEATURES, "extra_groups": GROUPS, "feature_names": names, "features_used": feats,
           "n_folds": 1, "n_seeds": 1, "bucketing_fingerprint": fp,
           "run": RUN, "train_set": "MATCH_ALL" + (f" minus {EXCLUDE}" if EXCLUDE else "") + (f", DEV_REST_REST fold {REST_FOLD} only" if REST_FOLD is not None else ""), "train_limited": False, "calibration": calib, "decision": decision,
           "floor_threshold": floor_t, "oof_f05": f05, "oof_floor_f05": floor[floor_t],
           "holdout_note": "oof_f05 = F0.5 on a held-out 10% of MATCH_ALL S1 (calibration slice), not 5-fold OOF",
           "best_iteration": best, "num_boost_round": NUM_BOOST_ROUND, "truncated": bool(truncated),
           "match_config": mcfg, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    io_utils.atomic_write_json(rd / "matcher.json", doc)
    io_utils.write_metrics(f"match_train_{RUN}", {
        "run": RUN, "groups": GROUPS, "excluded_sets": EXCLUDE, "n_queries": int(queries.size), "n_rows": int(y.size), "n_fit_rows": int(fit.sum()),
        "n_es_s1": int(q_es.size), "n_cal_s1": int(q_cal.size), "best_iteration": best,
        "num_boost_round": NUM_BOOST_ROUND, "truncated": bool(truncated), "holdout_f05": f05,
        "decision": decision, "rule_comparisons": {"expected_f_vs_threshold": ef, "one_owner": oo},
        "floor": {"threshold": floor_t, "f05": floor[floor_t]}, "vs_full_single_same_slice": vs_fs,
        "full_single_same_slice_f05": fs_mean, "wall_s": round(time.time() - t0, 1)},
        int(q_cal.size), int(scorer.n_true.sum()), cfg)
    print(f">>> saved {rd} (NOT copied to the live dir). total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
