"""N05b: does the p_rerank pool-competition rule add anything ON TOP of the live one-owner rule when one-owner
sees every S1, as it does at test time?

N05 (pool_competition_check.py) compared against a live decision whose one-owner only saw the 165,545-S1
reference slice. On TEST, match-predict applies one-owner across all 1.73M test S1 at once, so most of N05's
gain may already be in the submission. Here:

Train (labels on the reference slice only):
  competitors' p_cal = full_single's predictions on every train feature row (MATCH_TRAIN, MATCH_VALID,
  DEV_REST_180K, DEV_REST_REST). These are IN-SAMPLE for full_single except on the reference slice, so they are
  sharper than test-time p_cal: one-owner here is an optimistic stand-in for test-time one-owner.
  HOLDOUT_TEST (validation S1) and DEV_B have no feature rows, so they cannot take part in p_cal one-owner. The
  increment is therefore measured only on reference rows whose pool record appears in no HOLDOUT_TEST/DEV_B list
  ("complete" rows); the rule is not applied elsewhere.
  Decisions: live = threshold on universe one-owner p_cal; live+rule = also drop kept rows whose p_rerank is
  beaten by another S1's p_rerank by >= m.
Test (inputs only): full_single p_cal for every test row, live decision, then how many kept rows the rule
  would additionally drop, per country.

Run: PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/pool_competition_increment.py
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from test_shift_diag import cal_queries
from pool_competition_check import load_universe, pool_stats_fast, comp_features, MARGINS

FEAT_SETS = ["MATCH_TRAIN", "MATCH_VALID", "DEV_REST_180K", "DEV_REST_REST"]


def scored_rows(sets, booster, doc):
    """s1_idx, pool_idx, p_rerank, p_cal for every feature row of the given sets (deduplicated)."""
    cols = ["s1_idx", "pool_idx"] + doc["features_used"]
    out = []
    for name in sets:
        for p in sorted(match.features_dir(name).glob("part-*.parquet")):
            df = pq.read_table(p, columns=cols).to_pandas()
            pc = match.calibrate(booster.predict(df[doc["features_used"]].to_numpy(np.float32), num_threads=0),
                                 doc["calibration"])
            out.append(pd.DataFrame({"s1_idx": df["s1_idx"].to_numpy(), "pool_idx": df["pool_idx"].to_numpy(),
                                     "p_rerank": df["p_rerank"].to_numpy(), "p_cal": pc}))
            print(name, p.name, len(df), flush=True)
    return pd.concat(out, ignore_index=True).drop_duplicates(["s1_idx", "pool_idx"])


def best_other_cal(pool, p_cal):
    """Live one-owner keep factor: row survives iff its p_cal is the max for its pool record."""
    return match.one_owner(pool, p_cal)


def main():
    t0 = time.time()
    cfg = io_utils.load_config()
    doc = io_utils.read_json(io_utils.work_path("match", "models", "matcher.json"))
    booster = lgb.Booster(model_file=str(io_utils.work_path("match", "models", "model_fold0.txt")))
    thr = doc["decision"]["threshold"]
    out = {"script": "plan/checks/pool_competition_increment.py", "live_decision": doc["decision"]}

    # ---- train
    st = pool_stats_fast(load_universe(["MATCH_TRAIN", "MATCH_VALID", "DEV_REST_180K", "DEV_REST_REST",
                                        "DEV_B", "HOLDOUT_TEST"]))
    nofeat_pools = np.unique(load_universe(["DEV_B", "HOLDOUT_TEST"])["pool_idx"].to_numpy())
    allrows = scored_rows(FEAT_SETS, booster, doc)
    p_oo = best_other_cal(allrows["pool_idx"].to_numpy(), allrows["p_cal"].to_numpy())
    allrows["p_oo"] = p_oo
    q = cal_queries()
    ref = allrows[allrows["s1_idx"].isin(q)].reset_index(drop=True)
    del allrows
    s1, pool = ref["s1_idx"].to_numpy(), ref["pool_idx"].to_numpy()
    prr, p_cal, p_oo = ref["p_rerank"].to_numpy().astype(np.float64), ref["p_cal"].to_numpy(), ref["p_oo"].to_numpy()
    y = match.pair_labels(s1, pool)
    sc = match.Scorer(q, s1, y, match.true_counts(q))
    complete = ~np.isin(pool, nofeat_pools)
    _, is_best, _, margin = comp_features(s1, pool, prr, st)

    keep_plain = p_cal >= thr
    keep_live = p_oo >= thr
    f_plain, f_live = sc.per_entity(keep_plain)[0], sc.per_entity(keep_live)[0]
    out["train"] = {"ref_rows": int(s1.size), "row_share_complete": float(complete.mean()),
                    "f05_no_one_owner": float(f_plain.mean()),
                    "f05_universe_one_owner_insample": float(f_live.mean()),
                    "universe_one_owner_vs_none": match.paired_delta(f_live, f_plain),
                    "kept_rows_not_prr_best_after_one_owner": int((keep_live & ~is_best).sum()),
                    "kept_rows_not_prr_best_after_one_owner_complete": int((keep_live & ~is_best & complete).sum())}
    rules = {}
    for m in MARGINS:
        drop = keep_live & ~is_best & (margin <= -m) & complete
        f_new = sc.per_entity(keep_live & ~drop)[0]
        rules[str(m)] = {"rows_dropped": int(drop.sum()), "tp_dropped": int((drop & y).sum()),
                         "vs_universe_one_owner": match.paired_delta(f_new, f_live)}
    out["train"]["rule_on_top"] = rules
    print(json.dumps(out["train"], indent=1), flush=True)

    # ---- test (inputs only)
    tr = scored_rows(["TEST"], booster, doc)
    tr["p_oo"] = match.one_owner(tr["pool_idx"].to_numpy(), tr["p_cal"].to_numpy())
    stt = pool_stats_fast(tr[["s1_idx", "pool_idx", "p_rerank"]])
    _, is_best_t, _, margin_t = comp_features(tr["s1_idx"].to_numpy(), tr["pool_idx"].to_numpy(),
                                              tr["p_rerank"].to_numpy().astype(np.float64), stt)
    keep_t = tr["p_oo"].to_numpy() >= thr
    country = pq.read_table(io_utils.work_path("records", "test_s1.parquet"),
                            columns=["country_raw"]).column("country_raw").to_numpy()
    c = country[tr["s1_idx"].to_numpy()]
    test = {"rows": int(len(tr)), "kept_rows": int(keep_t.sum()),
            "kept_rows_before_one_owner": int((tr["p_cal"].to_numpy() >= thr).sum())}
    for m in MARGINS:
        drop = keep_t & ~is_best_t & (margin_t <= -m)
        per = {}
        for cc in np.unique(c):
            mm = c == cc
            n_s1 = np.unique(tr["s1_idx"].to_numpy()[mm]).size
            per[str(cc)] = {"rows_dropped": int((drop & mm).sum()),
                            "s1_affected_share": float(np.unique(tr["s1_idx"].to_numpy()[drop & mm]).size / max(n_s1, 1))}
        test[str(m)] = {"rows_dropped": int(drop.sum()),
                        "s1_affected_share": float(np.unique(tr["s1_idx"].to_numpy()[drop]).size / 1732544),
                        "per_country": per}
    out["test_inputs_only"] = test
    out["wall_s"] = round(time.time() - t0, 1)
    io_utils.write_metrics("pool_competition_increment", out, int(q.size), int(sc.n_true.sum()), cfg)
    print(json.dumps(test, indent=1), flush=True)


if __name__ == "__main__":
    main()
