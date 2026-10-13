"""LightGBM hyperparameter search for the stage-3 matcher, run AFTER the data-scaling decision
(plan/checks/dev_rest_full.py) so it tunes on whichever training-set size the team ships.

Never touched in this project before now: match.json's lgb block (learning_rate 0.05, num_leaves 63,
min_data_in_leaf 100, feature_fraction 0.8, bagging_fraction 0.8, lambda_l2 1.0) is the original plan's
starting guess. This does a bounded random search, screens cheaply, confirms the best configs fully, and
applies the SAME adoption rule as every other experiment in plan/FEATURE_EXPERIMENTS.md §2.6 (paired OOF
delta >= +0.0005 with CI lower bound > 0; MATCH_VALID CI upper bound >= 0 where a MATCH_VALID exists; no
guard segment -- country_US, country_India, singleton -- drops by more than 0.002 with CI upper bound < 0).
It does not touch configs/match.json, work/match/models/, or any candidate/feature directory -- it only
reads existing features and writes artifacts/metrics/hparam_search.json.

Out of scope on purpose (X5 already tested and rejected both): monotone constraints (x5_mono, OOF -0.00102)
and the lambdarank objective (x5_rank, OOF -0.01983). The objective stays binary logloss.

Usage (set CASE after the data-scaling result is known):
  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/hparam_search.py

Two cases for the training pack + baseline, decide which applies once dev_rest_full.py finishes:
  A) Full-data (MATCH_ALL) run rejected/not worth it -> tune on MATCH_TRAIN only (180,000 S1).
     TRAIN_SETS = [("MATCH_TRAIN", None)]. MATCH_VALID is left over and is scored as the confirmation set
     (rule §2.6 bullet 2). BASELINE_RUN = "x1_strsim" is used as a REPRODUCTION CHECK only: the in-script
     default-config baseline must reproduce its OOF_per_entity.npy (deterministic=True, same folds, no
     x1_strsim fold reached the 3000-round cap: best_iteration 1574-2011, match_train_x1_strsim.json).
  B) Full-data run adopted -> tune on MATCH_ALL (MATCH_TRAIN+MATCH_VALID+DEV_REST_180K+DEV_REST_REST,
     1,655,458 S1), TRAIN_SETS = match_all_train_sets(). No confirmation set is left over (MATCH_VALID is in
     training), so bullet 2 is not evaluable and is reported as such. BASELINE_RUN = None: dev_rest_full.py
     does not save per-entity arrays, so there is nothing on disk to reproduce.

Change from the first version (review 2026-09-26): the baseline is now ALWAYS the shipped match.json lgb
params trained inside this script, through the same folds, the same round-cap rule and the same
decision-selection code as every candidate config. The first version read the baseline's per-entity
array from disk; that paired a candidate selected by a max over (one_owner x 91 thresholds) against a
baseline selected by run_match_train's CI-gated rule, with a different round cap -- a delta that mixes
the hyperparameter effect with the selection-rule and cap differences. np.array_equal on the query arrays
only proved the two arrays index the same S1 in the same (sorted) order; it could not detect those.
"""
import math
import time

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ber import io_utils, match, splits

CASE = "A"   # "A" or "B", see module docstring; set once dev_rest_full.py's result is known


def match_all_train_sets():
    """Case B pack, built exactly like dev_rest_full.py's `packs` (same slices, same order)."""
    from dev_rest_full import SET_NAME_REST, pick_remaining_slice  # plan/checks is sys.path[0]
    s1_rest, s1_already = pick_remaining_slice()
    return [("MATCH_TRAIN", None), ("MATCH_VALID", None), ("DEV_REST_180K", s1_already),
            (SET_NAME_REST, s1_rest)]


def case_settings(case):
    """(TRAIN_SETS, BASELINE_RUN, confirm_set) of a case."""
    if case == "A":
        return [("MATCH_TRAIN", None)], "x1_strsim", "MATCH_VALID"
    if case == "B":
        return match_all_train_sets(), None, None
    raise ValueError(f"CASE must be 'A' or 'B', got {case!r}")


N_SCREEN_TRIALS = 16      # random configs, plus the shipped default as screen anchor (trial 0)
N_CONFIRM_TOP = 3         # was 2, see the screening note in select_for_confirmation
SEED = 20260926
SCREEN_FOLDS = 3          # cheaper than the full 5 for ranking configs only
EARLY_STOP = 100          # same patience as match.json early_stopping_rounds

# Round cap. Was a fixed NUM_BOOST_ROUND = 3000 for every config: at 180,000 S1 and lr 0.05 the folds
# stopped at 1574-2011 rounds (match_train_x1_strsim.json), and at 360,000 S1 two of five folds reached
# 3000 without stopping (dev_rest_full.py docstring). Rounds-to-converge scale roughly as 1/learning_rate,
# so lr 0.02 needs ~2.5x the rounds of lr 0.05: a fixed cap truncates exactly the low-lr configs, and the
# search would rank them by where the cap cut them off, not by what they converge to. The cap is now a
# budget in (learning_rate x rounds): 400 = 8000 rounds at lr 0.05 (2x dev_rest_full.py's 4000), 20000 at
# lr 0.02. Early stopping makes an unused cap free. A fold whose best_iteration lands inside the last
# EARLY_STOP rounds (patience never exhausted = truncated) is retrained once at 2x the cap; if it is still
# truncated the trial is flagged and cannot be adopted.
ROUND_BUDGET = 400.0
MIN_ROUND_CAP = 4000
MAX_ROUND_CAP = 60000

# Search space (review notes, evidence labels per AGENTS.md):
# - learning_rate 0.02-0.10 log: unchanged. Cost warning: lr 0.02 at MATCH_ALL (~9M rows) may need >10,000
#   rounds per fold; the wall time of such trials is not measured yet (d).
# - num_leaves: 255 added. At MATCH_ALL ~7M fit rows per fold, 127 leaves average ~55,000 rows per leaf;
#   capacity above 127 is plausible at that row count (c). Unmeasured.
# - min_data_in_leaf: was {50,100,150,250}; widened log-spaced to {50,100,200,500,1000}. At ~7M fit rows
#   (case B) 250 is 0.004% of the fold, so the old range barely constrains any leaf and the knob was close
#   to inert; 500/1000 regularize rare-pattern leaves (c). Still 0.14% of the ~700k fit rows in case A.
# - feature_fraction / bagging_fraction 0.6-0.95: kept. 0.6 of 68 columns = 41 per tree; 0.6 of >=700k fit
#   rows is >=420k rows per tree, far above any leaf-size bound. Positive rate 0.62
#   (match_train_x1_strsim.json positive_rate) means row subsampling does not starve either class (c).
# - lambda_l2 0.1-5.0 log: kept.
# Fixed on purpose: bagging_freq 1 (with bagging_fraction searched, freq only sets how often the row sample
# is redrawn; 1 = redraw every tree, most decorrelation, no extra cost) (c). max_bin 255: the columns are
# mostly [0,1] similarities, 255 quantile bins ~ 0.004 resolution; 511 doubles histogram build time for a
# gain nobody measured here (d). Not added: min_sum_hessian_in_leaf (default 1e-3; per-row hessian
# p(1-p) >= ~1e-3 except in near-pure leaves, so min_data_in_leaf binds first) (c); path_smooth and
# extra_trees (variance reducers for small leaves / small data; leaves here are large) (d).
SEARCH_SPACE = {
    "learning_rate": (0.02, 0.10, "log"),
    "num_leaves": [31, 63, 95, 127, 255],
    "min_data_in_leaf": [50, 100, 200, 500, 1000],
    "feature_fraction": (0.6, 0.95, "linear"),
    "bagging_fraction": (0.6, 0.95, "linear"),
    "lambda_l2": (0.1, 5.0, "log"),
}
FIXED = {"objective": "binary", "bagging_freq": 1, "max_bin": 255, "seed": 42,
         "deterministic": True, "force_row_wise": True, "verbosity": -1}
MIN_GAIN = 0.0005         # team adoption threshold, FEATURE_EXPERIMENTS.md §2.6 -- not tuned here
GUARD_SEGMENTS = ("country_US", "country_India", "singleton")
GUARD_DROP = 0.002


def round_cap(learning_rate):
    return int(min(MAX_ROUND_CAP, max(MIN_ROUND_CAP, math.ceil(ROUND_BUDGET / learning_rate))))


def default_config(mcfg):
    """The shipped lgb block (match.json), i.e. the baseline every trial is paired against."""
    return dict(mcfg["lgb"])


def sample_config(rng):
    cfg = dict(FIXED)
    for k, spec in SEARCH_SPACE.items():
        if isinstance(spec, list):
            cfg[k] = int(spec[rng.integers(0, len(spec))])
        else:
            lo, hi, kind = spec
            cfg[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if kind == "log" \
                else float(rng.uniform(lo, hi))
    return cfg


def load_pack(fp, train_sets, feats):
    Xs, s1s, pools, ys, qs = [], [], [], [], []
    for set_name, s1_override in train_sets:
        X, s1, pool, meta = match.load_features(set_name, fp)
        if meta.get("limited"):
            raise RuntimeError(f"{set_name} features are a --limit smoke build; rebuild before tuning")
        Xs.append(X[:, [match._COL[f] for f in feats]]); s1s.append(s1); pools.append(pool)
        ys.append(match.pair_labels(s1, pool))
        qs.append(s1_override if s1_override is not None else match.set_queries(set_name))
    queries = np.sort(np.concatenate(qs)).astype(np.int32)
    # A duplicated S1 across sets would be one group in GroupKFold but two entities in Scorer's mean.
    if np.unique(queries).size != queries.size:
        raise RuntimeError("TRAIN_SETS overlap: some S1 appear in two sets")
    # Validation S1 (HOLDOUT_TEST) is scoring-only (AGENTS.md rule 5); fail if any slipped into the pack.
    if np.isin(queries, splits.load_query_sets()["HOLDOUT_TEST"]).any():
        raise RuntimeError("TRAIN_SETS contain HOLDOUT_TEST (validation-S1) queries")
    return (np.concatenate(Xs), np.concatenate(s1s), np.concatenate(pools), np.concatenate(ys), queries)


def _fit_fold(params, X, y, fit_i, es_i, feats, cap):
    dfit = lgb.Dataset(X[fit_i], label=y[fit_i], feature_name=feats, free_raw_data=False)
    des = lgb.Dataset(X[es_i], label=y[es_i], reference=dfit, free_raw_data=False)
    return lgb.train(params, dfit, num_boost_round=cap, valid_sets=[des], valid_names=["es"],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(period=0)])


def run_fold_cv(params, X, s1, y, n_folds, threads, feats, fold_seed, es_frac, keep_models=False):
    """OOF raw scores with the same fold/early-stop split construction as match.run_match_train.

    fold_seed / es_frac were hardcoded (42, 0.1); they now come from match.json so the folds stay
    identical to production if the config changes. GroupKFold(shuffle=True) permutes the unique S1 and
    array_splits them (sklearn 1.9.1), so fold membership depends only on the set of S1: every config
    sees identical folds and per-entity scores are paired across configs."""
    oof = np.zeros(y.size, dtype=np.float64)
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=fold_seed)
    folds, models = [], []
    cap0 = round_cap(params["learning_rate"])
    for k, (tr, te) in enumerate(gkf.split(X, y, s1)):
        gss = GroupShuffleSplit(n_splits=1, test_size=es_frac, random_state=fold_seed + k)
        fit_i, es_i = next(gss.split(tr, groups=s1[tr]))
        fit_i, es_i = tr[fit_i], tr[es_i]
        p = dict(params, num_threads=threads)
        cap = cap0
        booster = _fit_fold(p, X, y, fit_i, es_i, feats, cap)
        truncated = booster.best_iteration > cap - EARLY_STOP
        if truncated and cap < MAX_ROUND_CAP:
            cap = min(MAX_ROUND_CAP, 2 * cap)
            booster = _fit_fold(p, X, y, fit_i, es_i, feats, cap)
            truncated = booster.best_iteration > cap - EARLY_STOP
        oof[te] = booster.predict(X[te], num_iteration=booster.best_iteration)
        folds.append({"fold": k, "best_iteration": int(booster.best_iteration), "cap": int(cap),
                      "truncated": bool(truncated)})
        if keep_models:
            models.append(booster)
    return oof, folds, models


def calibrate_oof(oof, y):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, y)
    calib = {"x": iso.X_thresholds_, "y": iso.y_thresholds_}
    return match.calibrate(oof, calib), calib


def select_decision(p_cal, s1, pool, scorer, dcfg):
    """Decision rule chosen exactly as match.run_match_train does it, so a trial is scored the way it would
    ship: per one_owner setting, the best threshold on the grid (first max) and the expected-F0.5 rule;
    expected_f replaces threshold only if its paired gain >= expected_f_min_gain with CI lower bound > 0;
    then one_owner is adopted by the same paired rule. The first version took the plain max over
    (one_owner x threshold) and skipped expected_f: that picks one_owner on any positive noise and scores a
    config under a rule production might not choose (x1_strsim: expected_f vs threshold +0.00007
    [-0.00014, +0.00025], rejected; one_owner +0.00019 [+0.00015, +0.00023], adopted)."""
    grid = match.threshold_grid(dcfg)
    cands = {}
    for owner in (False, True):
        pc = match.one_owner(pool, p_cal) if owner else p_cal
        sweep = {float(t): scorer.f05(pc >= t) for t in grid}
        t_best = max(sweep, key=sweep.get)
        for dec in ({"rule": "threshold", "threshold": t_best, "one_owner": owner},
                    {"rule": "expected_f", "threshold": None, "one_owner": owner}):
            cands[(dec["rule"], owner)] = (dec, scorer.per_entity(match.apply_decision(dec, s1, pool, p_cal))[0])
    ef = match.paired_delta(cands[("expected_f", False)][1], cands[("threshold", False)][1])
    base_rule = "expected_f" if match.adopt(ef, dcfg["expected_f_min_gain"]) else "threshold"
    oo = match.paired_delta(cands[(base_rule, True)][1], cands[(base_rule, False)][1])
    use_owner = match.adopt(oo, dcfg["one_owner_min_gain"])
    decision, per_entity = cands[(base_rule, use_owner)]
    return decision, per_entity


def evaluate(oof, y, s1, pool, scorer, dcfg):
    p_cal, calib = calibrate_oof(oof, y)
    decision, per_entity = select_decision(p_cal, s1, pool, scorer, dcfg)
    return {"f05": float(per_entity.mean()), "decision": decision, "per_entity": per_entity, "calib": calib}


def guard_check(pe, base_pe, masks):
    """Rule §2.6 bullet 3: breach if a guard segment's paired delta < -GUARD_DROP with CI upper bound < 0."""
    out = {}
    for name in GUARD_SEGMENTS:
        m = masks.get(name)
        if m is None or not m.any():
            out[name] = {"missing": True}
            continue
        d = match.paired_delta(pe[m], base_pe[m], n_boot=500)
        d["breach"] = bool(d["delta"] < -GUARD_DROP and d["ci95"][1] < 0)
        out[name] = d
    return out


def score_confirm_set(models, calib, decision, feats, fp, confirm_set):
    """Per-entity F0.5 on the confirmation set with fold-average raw p, as match.score_set does."""
    X, s1, pool, _ = match.load_features(confirm_set, fp)
    X = X[:, [match._COL[f] for f in feats]]
    raw = np.mean([b.predict(X, num_iteration=b.best_iteration) for b in models], axis=0)
    p_cal = match.calibrate(raw.astype(np.float32), calib)
    queries = match.set_queries(confirm_set)
    y = match.pair_labels(s1, pool)  # scoring only
    scorer = match.Scorer(queries, s1, y, match.true_counts(queries))
    return scorer.per_entity(match.apply_decision(decision, s1, pool, p_cal))[0]


def select_for_confirmation(screened, n_top):
    """VALIDATION_PLAN.md §6.4 T1 screen rule: promote only trials whose paired screen delta vs the default
    anchor is > 0, drop any whose CI upper bound < 0; then the top n_top by screen F0.5.

    Why 3, not 2: paired SE of an OOF delta at 180,000 S1 is ~0.0001 (x1_strsim CI [0.00060, 0.00097]
    -> half-width 0.00019, FEATURE_EXPERIMENTS.md §4) (a), ~0.00004 at 1.655M by 1/sqrt(n) (c). Two
    configs closer than ~2 SE x sqrt(2) ~ 0.0003 (case A) can swap rank on fold noise alone. Screening at 3
    folds also trains on 67% of S1 vs 80% at 5 folds, a systematic shift that favours lower-capacity configs
    in the screen (c), which variance-based widening does not fix; one extra confirmation slot hedges it."""
    promoted = [r for r in screened if r["trial"] != 0 and r["vs_default"]["delta"] > 0
                and not r["truncated"]]
    promoted.sort(key=lambda r: -r["screen_f05"])
    return promoted[:n_top]


def _public(r):
    return {k: v for k, v in r.items() if k not in ("per_entity", "confirm_per_entity", "calib", "models")}


def main():
    t0 = time.time()
    cfg, mcfg = io_utils.load_config(), match.load_match_config()
    fp = match.bucketing_fingerprint(cfg)
    threads = io_utils.n_workers(cfg)
    lg = io_utils.get_logger("hparam_search")
    train_sets, baseline_run, confirm_set = case_settings(CASE)
    feats = match.used_features(mcfg)  # was match.FEATURES; production drops match.json drop_features
    dcfg = mcfg["decision"]
    cv = dict(threads=threads, feats=feats, fold_seed=mcfg["fold_seed"], es_frac=mcfg["early_stop_frac"])

    X, s1, pool, y, queries = load_pack(fp, train_sets, feats)
    lg.info("case %s: %d S1, %d rows, %d features (sets=%s)", CASE, queries.size, y.size, len(feats),
            [s for s, _ in train_sets])
    scorer = match.Scorer(queries, s1, y, match.true_counts(queries))  # built once, reused by every trial
    from ber.evaluate import segment_data
    masks = dict(segment_data(queries, np.zeros(0, dtype=np.uint32))["query"])
    masks["singleton"] = scorer.n_true == 0

    # ---- screen: trial 0 = shipped default (anchor for paired screen deltas), then random configs
    rng = np.random.default_rng(SEED)
    configs = [default_config(mcfg)] + [sample_config(rng) for _ in range(N_SCREEN_TRIALS)]
    screened, anchor = [], None
    for i, cfg_i in enumerate(configs):
        oof, folds, _ = run_fold_cv(cfg_i, X, s1, y, SCREEN_FOLDS, **cv)
        ev = evaluate(oof, y, s1, pool, scorer, dcfg)
        if i == 0:
            anchor = ev["per_entity"]
        r = {"trial": i, "config": cfg_i, "screen_f05": ev["f05"], "decision": ev["decision"],
             "folds": folds, "truncated": any(f["truncated"] for f in folds),
             "vs_default": match.paired_delta(ev["per_entity"], anchor)}
        screened.append(r)
        print(f"[{i}/{N_SCREEN_TRIALS}] screen F0.5 {r['screen_f05']:.5f} vs default "
              f"{r['vs_default']['delta']:+.5f}  lr={cfg_i['learning_rate']:.3f} leaves={cfg_i['num_leaves']} "
              f"min_leaf={cfg_i['min_data_in_leaf']} ff={cfg_i['feature_fraction']:.2f} "
              f"bf={cfg_i['bagging_fraction']:.2f} l2={cfg_i['lambda_l2']:.2f} "
              f"iters={[f['best_iteration'] for f in folds]}{' TRUNCATED' if r['truncated'] else ''} "
              f"({time.time()-t0:.0f}s)", flush=True)

    top = select_for_confirmation(screened, N_CONFIRM_TOP)
    print(f"\n>>> {len(top)} trial(s) promoted; confirming with {mcfg['n_folds']}-fold CV + default ...",
          flush=True)

    # ---- confirm: default first (the paired baseline), then the promoted trials, identical pipeline
    def confirm(config):
        oof, folds, models = run_fold_cv(config, X, s1, y, mcfg["n_folds"], keep_models=confirm_set is not None,
                                         **cv)
        ev = evaluate(oof, y, s1, pool, scorer, dcfg)
        out = {"config": config, "oof_f05": ev["f05"], "decision": ev["decision"], "folds": folds,
               "truncated": any(f["truncated"] for f in folds), "per_entity": ev["per_entity"]}
        if confirm_set is not None:
            out["confirm_per_entity"] = score_confirm_set(models, ev["calib"], ev["decision"], feats, fp,
                                                          confirm_set)
        return out

    base = confirm(default_config(mcfg))
    print(f"default: OOF F0.5 {base['oof_f05']:.5f} decision {base['decision']} ({time.time()-t0:.0f}s)",
          flush=True)
    repro = None
    if baseline_run is not None:
        rd = match.runs_dir(baseline_run)
        run_doc = io_utils.read_json(rd / "models" / "matcher.json")
        base_q = np.load(rd / "queries.npy")
        ok = (np.array_equal(base_q, queries) and run_doc["bucketing_fingerprint"] == fp
              and run_doc.get("features_used", match.FEATURES) == feats)
        repro = {"run": baseline_run, "comparable": bool(ok)}
        if ok:
            ref = np.load(rd / "OOF_per_entity.npy")
            repro.update(max_abs_diff=float(np.abs(ref - base["per_entity"]).max()),
                         oof_f05_run=float(ref.mean()), decision_run=run_doc["decision"])
        print(f"reproduction check vs {baseline_run}: {repro}", flush=True)
        if not ok or repro["max_abs_diff"] > 0:
            lg.warning("default config does not reproduce %s exactly; investigate before trusting deltas",
                       baseline_run)

    results = []
    for r in top:
        c = confirm(r["config"])
        c["trial"] = r["trial"]
        c["vs_default"] = match.paired_delta(c["per_entity"], base["per_entity"], n_boot=2000)
        c["guards"] = guard_check(c["per_entity"], base["per_entity"], masks)
        if confirm_set is not None:
            c["confirm_vs_default"] = match.paired_delta(c["confirm_per_entity"], base["confirm_per_entity"],
                                                         n_boot=2000)
        results.append(c)
        v = c["vs_default"]
        print(f"confirm trial {r['trial']}: OOF F0.5 {c['oof_f05']:.5f} delta {v['delta']:+.5f} "
              f"[{v['ci95'][0]:+.5f},{v['ci95'][1]:+.5f}]"
              + (f"  {confirm_set} delta {c['confirm_vs_default']['delta']:+.5f} "
                 f"[{c['confirm_vs_default']['ci95'][0]:+.5f},{c['confirm_vs_default']['ci95'][1]:+.5f}]"
                 if confirm_set else "") + f"  ({time.time()-t0:.0f}s)", flush=True)

    winner, verdict = None, {"adopt": False, "reason": "no trial promoted from screening"}
    if results:
        # Limitation: the winner is the max of len(results) configs on the same OOF, so its delta carries a
        # small selection optimism; the MATCH_VALID bullet (case A) is the out-of-sample check against it.
        winner = max(results, key=lambda c: c["oof_f05"])
        checks = {"oof_gain": match.adopt(winner["vs_default"], MIN_GAIN),
                  "guards": not any(g.get("breach") for g in winner["guards"].values()),
                  "not_truncated": not winner["truncated"] and not base["truncated"],
                  "confirm_set": (winner["confirm_vs_default"]["ci95"][1] >= 0) if confirm_set else None}
        adopt = all(v for v in checks.values() if v is not None)
        verdict = {"adopt": bool(adopt), "checks": checks,
                   "note": None if confirm_set else "case B: no confirmation set left; bullet 2 not evaluable"}
        print(f"\nBEST trial {winner['trial']}: {winner['config']}")
        print(f"OOF F0.5 {winner['oof_f05']:.5f} vs default {base['oof_f05']:.5f}: delta "
              f"{winner['vs_default']['delta']:+.5f} CI {winner['vs_default']['ci95']} checks {checks} -> "
              f"{'ADOPT' if adopt else 'REJECT'} (mechanical rule)")

    doc = {"case": CASE, "train_sets": [s for s, _ in train_sets], "confirm_set": confirm_set,
           "n_screen_trials": N_SCREEN_TRIALS, "screen_folds": SCREEN_FOLDS, "round_budget": ROUND_BUDGET,
           "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()}, "min_gain": MIN_GAIN,
           "screened": screened, "default": _public(base), "reproduction": repro,
           "confirmed": [_public(c) for c in results],
           "winner_trial": winner["trial"] if winner else None, "verdict": verdict,
           "wall_s": round(time.time() - t0, 1)}
    path = io_utils.write_metrics("hparam_search", doc, int(queries.size), int(scorer.n_true.sum()), cfg)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
