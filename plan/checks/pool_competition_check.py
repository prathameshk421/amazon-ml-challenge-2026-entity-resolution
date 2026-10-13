"""N05: is pool-side competition (reverse retrieval) a signal the live matcher does not already use?

For every candidate row (q, p) we look at pool record p from the pool side: how many S1 lists contain p, and
how q's p_rerank for p compares with the best p_rerank p gets from any OTHER S1. Only candidate lists and
p_rerank are used for the competitors (inputs, no labels). Competitor universe on train = every train-split S1
that has stage-2 candidates (train S1 sets + validation S1 inputs; validation labels are never read). On test =
all test S1 (inputs only).

Labeled reference = full_single's calibration slice (165,545 train S1, never fitted on), scored with the live
model. Questions:
  A  coverage: share of reference rows / S1 whose pool record sits in >= 2 S1 lists; same on test
  B  where the live model's errors sit: FP and missed-in-candidate rows split by "q is p's best S1 by p_rerank"
  C  within the uncertain band p_cal in [0.1, 0.9): P(match) by competition bucket (signal left over after p_cal)
  D  no-training rule: drop a predicted positive when another S1 beats q's p_rerank for p by >= m; paired delta

No model is trained here.

Run: PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/pool_competition_check.py
"""
import glob

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from test_shift_diag import cal_queries, read_rows, BAND

TRAIN_SETS = ["MATCH_TRAIN", "MATCH_VALID", "DEV_REST_180K", "DEV_REST_REST", "DEV_B", "HOLDOUT_TEST"]
MARGINS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def load_universe(sets):
    parts = []
    for s in sets:
        for f in sorted(glob.glob(str(io_utils.work_path("candidates", s, "x").parent / "part-*.parquet"))):
            parts.append(pq.read_table(f, columns=["s1_idx", "pool_idx", "p_rerank"]).to_pandas())
    u = pd.concat(parts, ignore_index=True).drop_duplicates(["s1_idx", "pool_idx"])
    return u


def pool_stats_fast(u):
    u = u.sort_values(["pool_idx", "p_rerank"], ascending=[True, False], kind="stable").reset_index(drop=True)
    pool = u["pool_idx"].to_numpy()
    start = np.flatnonzero(np.r_[True, pool[1:] != pool[:-1]])
    cnt = np.diff(np.r_[start, pool.size])
    pr = u["p_rerank"].to_numpy()
    second = np.full(start.size, np.nan)
    has2 = cnt > 1
    second[has2] = pr[start[has2] + 1]
    return pd.DataFrame({"n_own": cnt, "best": pr[start], "best_s1": u["s1_idx"].to_numpy()[start],
                         "second": second}, index=pool[start])


def comp_features(s1, pool, prr, st):
    s = st.loc[pool]
    n_own = s["n_own"].to_numpy()
    is_best = s["best_s1"].to_numpy() == s1
    best_other = np.where(is_best, s["second"].to_numpy(), s["best"].to_numpy())
    best_other = np.where(n_own > 1, best_other, np.nan)
    return n_own, is_best, best_other, prr - np.nan_to_num(best_other, nan=-1.0)


def main():
    cfg = io_utils.load_config()
    doc = io_utils.read_json(io_utils.work_path("match", "models", "matcher.json"))
    booster = lgb.Booster(model_file=str(io_utils.work_path("match", "models", "model_fold0.txt")))
    thr = doc["decision"]["threshold"]
    out = {"script": "plan/checks/pool_competition_check.py", "live_threshold": thr,
           "competitor_sets_train": TRAIN_SETS, "competitor_score": "p_rerank"}

    u = load_universe(TRAIN_SETS)
    n_train_s1 = pq.ParquetFile(io_utils.work_path("records", "train_s1.parquet")).metadata.num_rows
    out["universe_train"] = {"rows": int(len(u)), "s1_with_candidates": int(u["s1_idx"].nunique()),
                             "train_split_s1_total": int(n_train_s1)}
    st = pool_stats_fast(u)
    del u

    q = cal_queries()
    ref = read_rows(["MATCH_TRAIN", "MATCH_VALID", "DEV_REST_180K", "DEV_REST_REST"], q)
    s1, pool = ref["s1_idx"].to_numpy(), ref["pool_idx"].to_numpy()
    prr = ref["p_rerank"].to_numpy().astype(np.float64)
    p = match.calibrate(booster.predict(ref[doc["features_used"]].to_numpy(np.float32)), doc["calibration"])
    y = match.pair_labels(s1, pool)
    sc = match.Scorer(q, s1, y, match.true_counts(q))
    n_own, is_best, best_other, margin = comp_features(s1, pool, prr, st)
    contested = n_own > 1
    keep = p >= thr
    fp, fn_in = keep & ~y, ~keep & y
    band = (p >= BAND[0]) & (p < BAND[1])

    # A coverage
    qpos = sc.qpos
    out["A_coverage_ref"] = {"rows": int(s1.size), "row_share_contested": float(contested.mean()),
                             "s1_share_any_contested": float((np.bincount(qpos, contested, q.size) > 0).mean()),
                             "row_share_not_best": float((~is_best).mean()),
                             "n_own_mean": float(n_own.mean())}

    # B error location
    def split(m):
        return {"n": int(m.sum()), "share_contested": float(contested[m].mean()),
                "share_q_not_best": float((~is_best[m]).mean())}
    out["B_errors"] = {"fp": split(fp), "fn_in_candidates": split(fn_in), "tp": split(keep & y),
                       "tn": split(~keep & ~y), "band_all": split(band)}

    # C residual signal inside the band
    buckets = {"uncontested": ~contested, "best_margin_ge0.2": is_best & contested & (margin >= 0.2),
               "best_margin_lt0.2": is_best & contested & (margin < 0.2),
               "not_best_gap_lt0.2": ~is_best & (margin > -0.2), "not_best_gap_ge0.2": ~is_best & (margin <= -0.2)}
    out["C_band_by_bucket"] = {k: {"n": int((band & m).sum()), "p_match": float(y[band & m].mean()) if (band & m).any() else None,
                                   "mean_p_cal": float(p[band & m].mean()) if (band & m).any() else None}
                               for k, m in buckets.items()}
    out["C_all_by_bucket"] = {k: {"n": int(m.sum()), "p_match": float(y[m].mean()),
                                  "mean_p_cal": float(p[m].mean())} for k, m in buckets.items()}

    # D no-training suppression rule
    f_live = sc.per_entity(keep)[0]
    out["f05_live_ref"] = float(f_live.mean())
    rules = {}
    for m in MARGINS:
        drop = keep & ~is_best & (margin <= -m)
        f_new = sc.per_entity(keep & ~drop)[0]
        rules[str(m)] = {"rows_dropped": int(drop.sum()), "tp_dropped": int((drop & y).sum()),
                         "f05": float(f_new.mean()), "vs_live": match.paired_delta(f_new, f_live)}
        band_s1 = np.bincount(qpos, band, q.size) > 0
        rules[str(m)]["vs_live_band_s1"] = match.paired_delta(f_new[band_s1], f_live[band_s1])
    out["D_suppress_rule"] = rules

    # A on test (inputs only)
    ut = load_universe(["TEST"])
    stt = pool_stats_fast(ut)
    n_own_t, is_best_t, _, _ = comp_features(ut["s1_idx"].to_numpy(), ut["pool_idx"].to_numpy(),
                                             ut["p_rerank"].to_numpy(), stt)
    out["A_coverage_test"] = {"rows": int(len(ut)), "row_share_contested": float((n_own_t > 1).mean()),
                              "row_share_not_best": float((~is_best_t).mean()), "n_own_mean": float(n_own_t.mean())}
    out["A_pool_records_listed"] = {"train": int(len(st)), "test": int(len(stt))}

    io_utils.write_metrics("pool_competition_check", out, int(q.size), int(sc.n_true.sum()), cfg)
    import json
    print(json.dumps({k: v for k, v in out.items() if k != "script"}, indent=1, default=str))


if __name__ == "__main__":
    main()
