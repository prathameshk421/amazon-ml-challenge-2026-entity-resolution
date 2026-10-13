"""Submission 3: contested-only hybrid of full_comp (M005 contention features, MATCH_ALL) and full_single (live).

N07: M005's gain on contested rows is robust; on uncontested rows it depends on train's orphan density, and test
US/India rows are less contested than train. Hybrid: rows whose pool record appears in >= 2 S1 candidate lists
(c_log_n_own > 0) take full_comp's calibrated p, all other rows take full_single's. The threshold and one-owner
choice are fitted on the MATCH_ALL calibration slice (165,545 S1 that neither model fitted or early-stopped on;
same seeded split as both training scripts). Also scores full_single and full_comp alone on that slice.

Test (inputs only): writes
  work/outputs/full_hybrid/matching_results.tsv   (recommended)
  work/outputs/full_comp/matching_results.tsv     (pure full_comp, its own decision)
both checked with match.check_matching_file.

Run: PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/final_hybrid.py
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from ber import exp, io_utils, match
from test_shift_diag import cal_queries

SETS = ["MATCH_TRAIN", "MATCH_VALID", "DEV_REST_180K", "DEV_REST_REST"]
# usage: final_hybrid.py [COMP_RUN]  (default full_comp); outputs <COMP_RUN>_hybrid and <COMP_RUN>
COMP_RUN = sys.argv[1] if len(sys.argv) > 1 else "full_comp"


def models():
    cd = match.runs_dir(COMP_RUN) / "models"
    sd = match.runs_dir("full_single") / "models"
    cdoc, sdoc = io_utils.read_json(cd / "matcher.json"), io_utils.read_json(sd / "matcher.json")
    return cd, cdoc, lgb.Booster(model_file=str(cd / "model_fold0.txt")), sdoc, lgb.Booster(model_file=str(sd / "model_fold0.txt"))


def score_rows(Xf, names, cdoc, cb, sdoc, sb):
    ci = {n: i for i, n in enumerate(names)}
    pc = match.calibrate(cb.predict(Xf[:, [ci[f] for f in cdoc["features_used"]]]), cdoc["calibration"])
    ps = match.calibrate(sb.predict(Xf[:, [ci[f] for f in sdoc["features_used"]]]), sdoc["calibration"])
    contested = Xf[:, ci["c_log_n_own"]] > 0
    return pc, ps, contested


def write_file(run_name, s1, pool, p_cal, keep):
    rec = pd.read_parquet(io_utils.work_path("records", "test_s1.parquet"), columns=["s1_idx", "entity_id"])
    if not np.array_equal(rec["s1_idx"].to_numpy(), np.arange(len(rec))):
        raise RuntimeError("records/test_s1.parquet: s1_idx is not 0..n-1")
    s1_ids = rec["entity_id"].tolist()
    store = io_utils.paths()["work_root"] / "norm" / "test_pool_store"
    src, num = np.load(store / "src.npy", mmap_mode="r"), np.load(store / "num.npy", mmap_mode="r")
    out = io_utils.work_path("outputs", run_name, "matching_results.tsv")
    k = np.flatnonzero(keep)
    k = k[np.lexsort((-p_cal[k], s1[k]))]
    match.write_matching_file(out, s1_ids, s1[k], pool[k], src, num)
    issues = match.check_matching_file(out, s1_ids, np.sort(match.pair_keys(s1, np.asarray(src)[pool], np.asarray(num)[pool])))
    country = pd.read_parquet(io_utils.work_path("records", "test_s1.parquet"), columns=["country_raw"])["country_raw"].to_numpy()
    n_pred = np.bincount(s1, weights=keep, minlength=len(s1_ids))
    per = {c: {"mean_pred_size": float(n_pred[country == c].mean()), "share_empty": float((n_pred[country == c] == 0).mean())}
           for c in sorted(set(country.tolist()))}
    if issues:
        raise RuntimeError(f"{out} failed checks: {issues[:5]}")
    return {"file": str(out), "pairs": int(keep.sum()), "per_country": per, "check_passed": True}


def main():
    t0 = time.time()
    cfg, mcfg = io_utils.load_config(), match.load_match_config()
    fp = match.bucketing_fingerprint(cfg)
    cd, cdoc, cb, sdoc, sb = models()
    if cdoc["bucketing_fingerprint"] != fp or sdoc["bucketing_fingerprint"] != fp:
        raise RuntimeError("fingerprint mismatch")
    q = cal_queries()
    parts = []
    for name in SETS:
        Xf, names, s1, pool, _ = exp.full_matrix(name, cdoc["extra_groups"], fp)
        m = np.isin(s1, q)
        pc, ps, con = score_rows(Xf[m], names, cdoc, cb, sdoc, sb)
        parts.append((s1[m], pool[m], pc, ps, con))
        print(name, int(m.sum()), f"{time.time() - t0:.0f}s", flush=True)
    s1, pool, pc, ps, con = (np.concatenate([p[i] for p in parts]) for i in range(5))
    o = np.lexsort((pool, s1))
    s1, pool, pc, ps, con = s1[o], pool[o], pc[o], ps[o], con[o]
    y = match.pair_labels(s1, pool)
    sc = match.Scorer(q, s1, y, match.true_counts(q))
    f_single = sc.per_entity(match.apply_decision(sdoc["decision"], s1, pool, ps))[0]
    f_comp = sc.per_entity(match.apply_decision(cdoc["decision"], s1, pool, pc))[0]
    ph = np.where(con, pc, ps).astype(np.float32)
    grid = match.threshold_grid(mcfg["decision"])
    cands = {}
    for owner in (False, True):
        pp = match.one_owner(pool, ph) if owner else ph
        sw = {float(t): sc.f05(pp >= t) for t in grid}
        tb = max(sw, key=sw.get)
        dec = {"rule": "threshold", "threshold": tb, "one_owner": owner}
        cands[owner] = (dec, sc.per_entity(match.apply_decision(dec, s1, pool, ph))[0])
    oo = match.paired_delta(cands[True][1], cands[False][1])
    dec_h, f_h = cands[match.adopt(oo, mcfg["decision"]["one_owner_min_gain"])]
    out = {"script": "plan/checks/final_hybrid.py", "cal_slice_s1": int(q.size),
           "contested_row_share_slice": float(con.mean()),
           "f05_full_single": float(f_single.mean()), "f05_full_comp": float(f_comp.mean()), "f05_hybrid": float(f_h.mean()),
           "hybrid_decision": dec_h, "hybrid_one_owner_cmp": oo,
           "full_comp_vs_single": match.paired_delta(f_comp, f_single, n_boot=2000),
           "hybrid_vs_single": match.paired_delta(f_h, f_single, n_boot=2000)}
    print({k: v for k, v in out.items() if k != "script"}, flush=True)

    # ---- test (inputs only)
    io_utils.log_test_usage("full_comp + full_single scores over all test candidates, hybrid decision (inputs only)",
                            "submission 3 files (plan/checks/final_hybrid.py)", str(match.features_dir("TEST")))
    Xf, names, s1t, poolt, meta = exp.full_matrix("TEST", cdoc["extra_groups"], fp)
    if meta["limited"]:
        raise RuntimeError("TEST features built with --limit")
    pct, pst, cont = score_rows(Xf, names, cdoc, cb, sdoc, sb)
    del Xf
    pht = np.where(cont, pct, pst).astype(np.float32)
    keep_h = match.apply_decision(dec_h, s1t, poolt, pht)
    keep_c = match.apply_decision(cdoc["decision"], s1t, poolt, pct)
    keep_s = match.apply_decision(sdoc["decision"], s1t, poolt, pst)
    out["test"] = {"contested_row_share": float(cont.mean()), "pairs_full_single_replay": int(keep_s.sum()),
                   f"{COMP_RUN}_hybrid": write_file(f"{COMP_RUN}_hybrid", s1t, poolt, pht, keep_h),
                   COMP_RUN: write_file(COMP_RUN, s1t, poolt, pct, keep_c)}
    out["wall_s"] = round(time.time() - t0, 1)
    io_utils.write_metrics(f"final_hybrid_{COMP_RUN}", out, q.size, int(sc.n_true.sum()), cfg)
    print(out["test"], flush=True)


if __name__ == "__main__":
    main()
