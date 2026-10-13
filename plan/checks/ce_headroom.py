"""Where does full_ce (submission 6, LB 0.976128) still lose F0.5? Oracle headroom per error type.

Rows: full_ce's calibration S1 (same seeded split, never fitted / early-stopped on) that lie in MATCH_TRAIN or
MATCH_VALID (~20K S1; their feature tables are small). Train labels only.

Per S1 each true link is one of: TP | FN_in_list (in the K5 candidate list, rejected) | FN_cut (in a wide stage-1+2
list, K_min 20 / p_min 0.0005, but cut from the live K5 list) | FN_stage1 (not even in the wide list).
Reports the current F0.5 and the F0.5 if one error type were fixed perfectly (oracle), plus the p_ce of each
error type, and writes samples with raw text to work/reports/errors/headroom_full_ce.parquet.

Run: PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/ce_headroom.py
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import exp, io_utils, match, rerank, run_candidates
from error_report import attach_text
from test_shift_diag import cal_queries


def f05(n_pred, n_true, n_tp):
    return match.entity_f05(n_pred, n_true, n_tp)


def main():
    t0 = time.time()
    cfg = io_utils.load_config()
    fp = match.bucketing_fingerprint(cfg)
    md = match.runs_dir("full_ce") / "models"
    doc = io_utils.read_json(md / "matcher.json")
    booster = lgb.Booster(model_file=str(md / "model_fold0.txt"))
    small = np.sort(np.concatenate([match.set_queries("MATCH_TRAIN"), match.set_queries("MATCH_VALID")]))
    q = np.intersect1d(cal_queries(), small).astype(np.int32)
    print(f">>> {q.size} calibration S1 in MATCH_TRAIN/MATCH_VALID", flush=True)

    parts = []
    for name in ("MATCH_TRAIN", "MATCH_VALID"):
        Xf, names, s1, pool, _ = exp.full_matrix(name, doc["extra_groups"], fp)
        m = np.isin(s1, q)
        ci = {n: i for i, n in enumerate(names)}
        X = np.column_stack([Xf[m][:, ci[f]] for f in doc["features_used"]]).astype(np.float32)
        p_cal = match.calibrate(booster.predict(X), doc["calibration"])
        parts.append(pd.DataFrame({"s1_idx": s1[m], "pool_idx": pool[m], "p_cal": p_cal,
                                   "p_ce": Xf[m][:, ci["p_ce_rec"]], "rank": Xf[m][:, ci["rank"]]}))
    df = pd.concat(parts, ignore_index=True).sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
    df["y"] = match.pair_labels(df.s1_idx.to_numpy(), df.pool_idx.to_numpy())
    df["keep"] = match.apply_decision(doc["decision"], df.s1_idx.to_numpy(), df.pool_idx.to_numpy(),
                                      df.p_cal.to_numpy())
    print(f">>> scored {len(df)} rows ({time.time()-t0:.0f}s)", flush=True)

    # wide stage-1+2 lists for the same S1 (train split, no test rows)
    model, feat_names = run_candidates._stage2_model(cfg)
    country = pd.read_parquet(io_utils.work_path("records", "train_s1.parquet"), columns=["country_raw"])["country_raw"].to_numpy()[q]
    out_dir = io_utils.work_path("exp", "headroom", "wide_k20")
    pool_lists, _, _, _ = rerank.generate_candidates("train", q, country, cfg, model, feat_names,
                                                     {"K_min": 20, "K_max": 60, "p_min": 0.0005}, out_dir)
    wide_key = np.concatenate([(np.int64(s) << 32) | np.asarray(pl, dtype=np.int64) for s, pl in zip(q, pool_lists)])
    print(f">>> wide lists: mean {np.mean([len(p) for p in pool_lists]):.1f} ({time.time()-t0:.0f}s)", flush=True)

    gt = pq.read_table(io_utils.work_path("records", "train_gt.parquet"), columns=["s1_idx", "pool_idx"]).to_pandas()
    gt = gt[np.isin(gt.s1_idx.to_numpy(), q)]
    gkey = gt.s1_idx.to_numpy().astype(np.int64) << 32 | gt.pool_idx.to_numpy().astype(np.int64)
    ckey = df.s1_idx.to_numpy().astype(np.int64) << 32 | df.pool_idx.to_numpy().astype(np.int64)
    in_list, in_wide = np.isin(gkey, ckey), np.isin(gkey, wide_key)

    qi = lambda s: np.searchsorted(q, s)
    n_true = np.bincount(qi(gt.s1_idx.to_numpy()), minlength=q.size).astype(float)
    tp = np.bincount(qi(df.s1_idx[df.keep & df.y].to_numpy()), minlength=q.size).astype(float)
    fp_ = np.bincount(qi(df.s1_idx[df.keep & ~df.y].to_numpy()), minlength=q.size).astype(float)
    fn_in = np.bincount(qi(df.s1_idx[~df.keep & df.y].to_numpy()), minlength=q.size).astype(float)
    fn_cut = np.bincount(qi(gt.s1_idx.to_numpy()[~in_list & in_wide]), minlength=q.size).astype(float)
    fn_s1 = np.bincount(qi(gt.s1_idx.to_numpy()[~in_wide]), minlength=q.size).astype(float)
    assert np.allclose(tp + fn_in + fn_cut + fn_s1, n_true)

    base = f05(tp + fp_, n_true, tp).mean()
    ref = match.Scorer(q, df.s1_idx.to_numpy(), df.y.to_numpy(), match.true_counts(q)).f05(df.keep.to_numpy())
    oracle = {
        "no_FP": f05(tp, n_true, tp).mean(),
        "fix_FN_in_list": f05(tp + fn_in + fp_, n_true, tp + fn_in).mean(),
        "fix_FN_cut": f05(tp + fn_cut + fp_, n_true, tp + fn_cut).mean(),
        "fix_FN_stage1": f05(tp + fn_s1 + fp_, n_true, tp + fn_s1).mean(),
        "fix_all_in_list(FP+FN_in)": f05(tp + fn_in, n_true, tp + fn_in).mean(),
    }
    res = {"n_s1": int(q.size), "f05": float(base), "f05_scorer_check": float(ref),
           "counts": {"TP": int(tp.sum()), "FP": int(fp_.sum()), "FN_in_list": int(fn_in.sum()),
                      "FN_cut": int(fn_cut.sum()), "FN_stage1": int(fn_s1.sum()), "true_links": int(n_true.sum())},
           "oracle_f05": {k: float(v) for k, v in oracle.items()},
           "oracle_gain": {k: float(v - base) for k, v in oracle.items()},
           "p_ce_quantiles": {}}
    err = pd.concat([df[df.keep & ~df.y].assign(kind="FP"), df[~df.keep & df.y].assign(kind="FN_in_list")])
    for k, g in err.groupby("kind"):
        res["p_ce_quantiles"][k] = {str(t): float(g.p_ce.quantile(t)) for t in (0.1, 0.25, 0.5, 0.75, 0.9)}
        res["p_ce_quantiles"][k]["share_p_ce_agrees_with_label"] = float(((g.p_ce >= 0.5) == (k == "FN_in_list")).mean())
    res["wall_s"] = round(time.time() - t0, 1)
    io_utils.write_metrics("ce_headroom_full_ce", res, int(q.size), int(n_true.sum()), cfg)
    miss = gt[~in_list].assign(kind=np.where(in_wide[~in_list], "FN_cut", "FN_stage1"), p_cal=np.nan, p_ce=np.nan, rank=np.nan)
    rows = pd.concat([err, miss], ignore_index=True)
    out = attach_text(rows[["s1_idx", "pool_idx", "kind", "p_cal", "p_ce", "rank"]])
    out.to_parquet(io_utils.work_path("reports", "errors", "headroom_full_ce.parquet"))
    print(res, flush=True)


if __name__ == "__main__":
    main()
