"""Test-like decision for MATCH_TRAIN OOF runs: one-owner over ALL train S1, as match-predict does over all test S1.

match-train / exp-train apply one-owner only among the 180,000 MATCH_TRAIN S1, so a pool record owned by an S1
outside MATCH_TRAIN never suppresses a MATCH_TRAIN row. On test every S1 competes. N05b measured that
universe one-owner is worth +0.00094 on the reference slice, so any feature that carries the same information
(pool-side contention, M005) looks better on MATCH_TRAIN OOF than it can be on test.

Competitor scores: full_single (live model) calibrated p for every train feature row OUTSIDE MATCH_TRAIN
(MATCH_VALID, DEV_REST_180K, DEV_REST_REST), reduced to max p per pool record and cached in
work/exp/outside_max_pcal.parquet. full_single was fitted on those rows, so competitors are sharper than
test-time ones; the bias is the same for parent and child. HOLDOUT_TEST / DEV_B S1 have no feature rows and
are not competitors here.

Usage (repository root):
  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/universe_owner_eval.py \
      --cache                       # build the competitor cache (~45 min, full_single over ~8.5M rows)
  ... universe_owner_eval.py PARENT CHILD [CHILD ...]   # paired comparison, writes universe_owner_<child>.json
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber import exp, io_utils, match

OUTSIDE = ["MATCH_VALID", "DEV_REST_180K", "DEV_REST_REST"]
CACHE = io_utils.work_path("exp", "outside_max_pcal.parquet")


def build_cache():
    doc = io_utils.read_json(io_utils.work_path("match", "models", "matcher.json"))
    booster = lgb.Booster(model_file=str(io_utils.work_path("match", "models", "model_fold0.txt")))
    parts = []
    for name in OUTSIDE:
        for p in sorted(match.features_dir(name).glob("part-*.parquet")):
            t = pq.read_table(p, columns=["pool_idx"] + doc["features_used"])
            X = np.column_stack([t.column(f).to_numpy() for f in doc["features_used"]]).astype(np.float32)
            pc = match.calibrate(booster.predict(X, num_threads=3), doc["calibration"])
            parts.append(pd.Series(pc).groupby(t.column("pool_idx").to_numpy()).max())
            print(name, p.name, t.num_rows, flush=True)
    s = pd.concat(parts).groupby(level=0).max()
    pool, val = s.index.to_numpy().astype(np.int64), s.to_numpy().astype(np.float64)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"pool_idx": pool, "max_p_cal": val}), CACHE)


def universe_keep(s1, pool, p_cal, thr, outside):
    """Threshold decision after one-owner over MATCH_TRAIN rows plus the outside competitors."""
    inside_best = pd.Series(p_cal).groupby(pool).transform("max").to_numpy()
    out_best = outside.reindex(pool).fillna(-1.0).to_numpy()
    p = np.where((p_cal >= inside_best) & (p_cal >= out_best), p_cal, 0.0)
    return p >= thr, p


def score(run, outside, queries):
    t = pq.read_table(match.runs_dir(run) / "oof.parquet").to_pandas()
    s1, pool, y, pc = (t[c].to_numpy() for c in ("s1_idx", "pool_idx", "y", "p_cal"))
    sc = match.Scorer(queries, s1, y, match.true_counts(queries))
    doc = io_utils.read_json(match.runs_dir(run) / "models" / "matcher.json")
    thr = doc["decision"]["threshold"]
    keep, p = universe_keep(s1, pool, pc, thr if thr is not None else 0.5, outside)
    fixed = sc.per_entity(keep)[0]
    sweep = {round(float(x), 2): sc.f05(p >= x) for x in np.arange(0.40, 0.951, 0.01)}
    tb = max(sweep, key=sweep.get)
    best = sc.per_entity(p >= tb)[0]
    return {"threshold_live": thr, "f05_fixed": float(fixed.mean()), "threshold_best": tb,
            "f05_best": float(best.mean())}, fixed, best


def main():
    if sys.argv[1] == "--cache":
        build_cache()
        return
    cfg = io_utils.load_config()
    outside = pq.read_table(CACHE).to_pandas().set_index("pool_idx")["max_p_cal"]
    parent, children = sys.argv[1], sys.argv[2:]
    queries = match.set_queries("MATCH_TRAIN")
    band = exp.band_mask(queries)
    rp, fa, ba = score(parent, outside, queries)
    for child in children:
        rc, fb, bb = score(child, outside, queries)
        out = {"script": "plan/checks/universe_owner_eval.py", "parent": parent, "child": child,
               "competitors": OUTSIDE, "parent_scores": rp, "child_scores": rc,
               "delta_fixed_threshold": match.paired_delta(fb, fa, n_boot=2000),
               "delta_best_threshold": match.paired_delta(bb, ba, n_boot=2000),
               "delta_fixed_band_s1": match.paired_delta(fb[band], fa[band]),
               "delta_fixed_non_band_s1": match.paired_delta(fb[~band], fa[~band])}
        io_utils.write_metrics(f"universe_owner_{child}", out, queries.size, int(match.true_counts(queries).sum()), cfg)
        print(child, rp, rc, out["delta_fixed_threshold"], out["delta_fixed_band_s1"], flush=True)


if __name__ == "__main__":
    main()
