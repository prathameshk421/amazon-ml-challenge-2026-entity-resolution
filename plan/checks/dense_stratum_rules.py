"""Decision-rule behaviour on the test-like stratum of the full_single calibration slice (train labels only).

test_shift_diag.json showed test lists carry ~1.9x more uncertain-band candidates per S1 (p_cal in [0.1, 0.9))
than the train reference. This script asks, on TRAIN data only: in the reference S1 whose lists already look
like that (>= 1 band candidate, or list size > 5), (1) how far below the all-S1 score they sit, (2) which
threshold is best there, (3) whether the expected-F rule beats the threshold rule there. Nothing is tuned on
test; the stratum is defined from train list statistics.

Run: PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/dense_stratum_rules.py
"""
import numpy as np
import lightgbm as lgb

from ber import io_utils, match
from test_shift_diag import cal_queries, read_rows, BAND


def main():
    cfg = io_utils.load_config()
    doc = io_utils.read_json(io_utils.work_path("match", "models", "matcher.json"))
    booster = lgb.Booster(model_file=str(io_utils.work_path("match", "models", "model_fold0.txt")))
    q = cal_queries()
    ref = read_rows(["MATCH_TRAIN", "MATCH_VALID", "DEV_REST_180K", "DEV_REST_REST"], q)
    s1, pool = ref["s1_idx"].to_numpy(), ref["pool_idx"].to_numpy()
    p = match.calibrate(booster.predict(ref[doc["features_used"]].to_numpy(np.float32)), doc["calibration"])
    y = match.pair_labels(s1, pool)
    sc = match.Scorer(q, s1, y, match.true_counts(q))
    qpos = sc.qpos
    n_band = np.bincount(qpos, weights=(p >= BAND[0]) & (p < BAND[1]), minlength=q.size)
    size = np.bincount(qpos, minlength=q.size)
    strata = {"all": np.ones(q.size, bool), "band_ge1": n_band >= 1, "band_0": n_band == 0,
              "size_gt5": size > 5, "size_le5": size <= 5}
    thr0 = doc["decision"]["threshold"]
    grid = np.round(np.arange(0.40, 0.951, 0.01), 2)
    per_t = {float(t): sc.per_entity(p >= t)[0] for t in grid}
    f_ef = sc.per_entity(match.decide_expected_f(s1, p))[0]
    out = {}
    for name, m in strata.items():
        f_by_t = {t: float(v[m].mean()) for t, v in per_t.items()}
        t_best = max(f_by_t, key=f_by_t.get)
        out[name] = {"share_s1": float(m.mean()), "f05_at_live_threshold": f_by_t[thr0],
                     "best_threshold": t_best, "f05_at_best": f_by_t[t_best],
                     "best_minus_live": match.paired_delta(per_t[t_best][m], per_t[thr0][m]),
                     "expected_f_minus_live": match.paired_delta(f_ef[m], per_t[thr0][m]),
                     "curve": {str(t): round(v, 5) for t, v in f_by_t.items() if round(t * 100) % 5 == 0}}
    io_utils.write_metrics("dense_stratum_rules", {"script": "plan/checks/dense_stratum_rules.py",
                                                   "live_threshold": thr0, "one_owner": "not applied",
                                                   "strata": out}, int(q.size), int(sc.n_true.sum()), cfg)
    for k, v in out.items():
        print(k, round(v["share_s1"], 3), "live", round(v["f05_at_live_threshold"], 5), "best_t", v["best_threshold"],
              round(v["best_minus_live"]["delta"], 5), v["best_minus_live"]["ci95"],
              "EF-live", round(v["expected_f_minus_live"]["delta"], 5), v["expected_f_minus_live"]["ci95"])


if __name__ == "__main__":
    main()
