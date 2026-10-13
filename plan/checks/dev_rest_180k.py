"""Learning-curve extension (VALIDATION_PLAN.md §10 step 8): generate candidates + features for another
180,000 S1 out of DEV_REST (the 1,455,458 train S1 never used by the matcher), then train x1_strsim's
frozen feature set on MATCH_TRAIN + this new slice (360,000 S1 total) to get a real third curve point,
instead of extrapolating from the 45k/90k/180k curve.

Reuses the exact production functions (rerank.generate_candidates, match._tasks/_feature_chunk/_write_part,
build_count_tables/build_tfidf_idf) so this is not a reimplementation of the pipeline -- it is the same
code, run once on rows nobody has queried yet. Writes to its own directories
(work/candidates/DEV_REST_180K/, work/match/features/DEV_REST_180K/) and never touches
work/match/models/, configs/*.json, or the MATCH_TRAIN/MATCH_VALID candidate or feature directories.

Run from the repository root:
  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/dev_rest_180k.py
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ber import io_utils, match, rerank, retrieve, splits

N_NEW = 180_000
SET_NAME = "DEV_REST_180K"
SEED = 20260926


def pick_dev_rest_slice():
    dev_rest = splits.load_query_sets()["DEV_REST"]
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(dev_rest, size=N_NEW, replace=False)).astype(np.int32)


def generate_candidates(s1_idx, cfg, lg):
    out_dir = io_utils.work_path("candidates", SET_NAME)
    if rerank._candidates_ready(out_dir):
        lg.info("candidates already exist at %s, reusing", out_dir)
        return out_dir
    model = retrieve.load_stage1_model(cfg)
    meta = io_utils.read_json(io_utils.work_path("rerank", "model_v1_meta.json"))
    feat_names = meta["features"]
    cut = cfg["rerank"]["cut"]
    country_raw = pd.read_parquet(io_utils.work_path("records", "train_s1.parquet"),
                                  columns=["country_raw"])["country_raw"].to_numpy()[s1_idx]
    t0 = time.time()
    rerank.generate_candidates("train", s1_idx, country_raw, cfg, model, feat_names, cut, out_dir)
    lg.info("candidates for %s (%d S1) generated in %.1fs", SET_NAME, s1_idx.size, time.time() - t0)
    return out_dir


def build_features(s1_idx, cfg, mcfg, lg):
    out_dir = match.features_dir(SET_NAME)
    if (out_dir / "meta.json").exists():
        meta = io_utils.read_json(out_dir / "meta.json")
        if meta.get("features") == match.FEATURES and meta.get("n_queries") == int(s1_idx.size):
            lg.info("features already exist at %s, reusing", out_dir)
            return
    t0 = time.time()
    cand = match.read_candidates(SET_NAME, s1_idx)
    names, addrs, country, chain = match._s1_inputs("train", s1_idx)
    match.build_count_tables("train", lg)
    match.build_tfidf_idf("train", lg)
    ncnt, acnt = match.query_counts("train", "name", names), match.query_counts("train", "addr", addrs)
    uniq, start, stop = match.group_bounds(cand["s1_idx"])
    qpos = np.searchsorted(s1_idx, uniq)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").unlink(missing_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    workers = io_utils.n_workers(cfg)
    tasks = match._tasks(cand, qpos, names, addrs, country, chain, ncnt, acnt, uniq, start, stop,
                         mcfg["chunk_queries"])
    part_rows = mcfg["part_queries"] // mcfg["chunk_queries"]
    buf, n_part, row0 = [], 0, 0
    with io_utils.spawn_pool(workers, match._init_worker, ("train",)) as pool:
        for X in pool.imap(match._feature_chunk, tasks):
            buf.append(X)
            if len(buf) >= part_rows:
                row0 = match._write_part(out_dir, n_part, cand, row0, buf)
                n_part, buf = n_part + 1, []
        if buf:
            row0 = match._write_part(out_dir, n_part, cand, row0, buf)
            n_part += 1
    meta = {"set": SET_NAME, "split": "train", "features": match.FEATURES, "n_queries": int(s1_idx.size),
            "n_rows": int(row0), "n_parts": n_part, "limited": False,
            "bucketing_fingerprint": match.bucketing_fingerprint(cfg), "match_config_version": mcfg["version"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    io_utils.atomic_write_json(out_dir / "meta.json", meta)
    lg.info("features for %s: %d queries, %d rows, %.1fs", SET_NAME, s1_idx.size, row0, time.time() - t0)


def combined_curve_point(mt_pack, extra_pack, mcfg, threads):
    X = np.concatenate([mt_pack[0], extra_pack[0]])
    s1 = np.concatenate([mt_pack[1], extra_pack[1]])
    pool = np.concatenate([mt_pack[2], extra_pack[2]])
    y = np.concatenate([mt_pack[3], extra_pack[3]])
    queries = np.sort(np.concatenate([mt_pack[4], extra_pack[4]]))
    t0 = time.time()

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
        fold_info.append({"fold": k, "best_iteration": int(booster.best_iteration), "n_fit_rows": int(fit_i.size)})
        print(f"    fold {k}: best_iteration {booster.best_iteration} ({time.time()-t0:.0f}s)")

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, y)
    p_cal = np.interp(oof, iso.X_thresholds_, iso.y_thresholds_).astype(np.float32)

    scorer = match.Scorer(queries, s1, y, match.true_counts(queries))
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
    return {"n_queries": int(queries.size), "n_rows": int(y.size), "oof_f05": float(f05),
            "threshold": t_best, "one_owner": bool(use_owner), "folds": fold_info,
            "wall_s": round(time.time() - t0, 1)}


def main():
    cfg, mcfg = io_utils.load_config(), match.load_match_config()
    lg = io_utils.get_logger("dev_rest_180k")

    s1_new = pick_dev_rest_slice()
    print(f">>> {time.strftime('%H:%M:%S')} generating candidates for {N_NEW} new DEV_REST S1 ...")
    generate_candidates(s1_new, cfg, lg)
    print(f">>> {time.strftime('%H:%M:%S')} building features ...")
    build_features(s1_new, cfg, mcfg, lg)

    fp = match.bucketing_fingerprint(cfg)
    print(f">>> {time.strftime('%H:%M:%S')} loading features and training on 360,000 S1 ...")
    mt = match.load_features("MATCH_TRAIN", fp)
    extra = match.load_features(SET_NAME, fp)
    mt_pack = (mt[0], mt[1], mt[2], match.pair_labels(mt[1], mt[2]), match.set_queries("MATCH_TRAIN"))
    extra_pack = (extra[0], extra[1], extra[2], match.pair_labels(extra[1], extra[2]), s1_new)

    pt = combined_curve_point(mt_pack, extra_pack, mcfg, io_utils.n_workers(cfg))
    print(f"n=360000: OOF F0.5 {pt['oof_f05']:.5f} (t={pt['threshold']}, one_owner={pt['one_owner']}, "
          f"{pt['wall_s']}s)")

    curve_path = io_utils.paths()["metrics_root"] / "learning_curve.json"
    doc = io_utils.read_json(curve_path)  # write_metrics flattens payload into the top-level dict
    points = doc["points"]
    points.append(dict(pt, source="dev_rest_180k.py: MATCH_TRAIN + 180k new DEV_REST S1"))
    io_utils.write_metrics("learning_curve", {"points": points, "note": doc.get("note")}, 0, 0, cfg)
    print(f"updated {curve_path}")


if __name__ == "__main__":
    main()
