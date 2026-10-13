"""RCA diagnostic: where does the live matcher (full_single) lose between internal held-out (0.98161) and the
public leaderboard (0.966797)? Test labels do not exist, so every test-side number here uses test INPUTS only
(candidate lists, p_rerank, matcher scores) and is compared with the same statistics on a labeled train
reference whose labels let us calibrate what each statistic means.

Train reference = full_single's calibration slice: 10% of MATCH_ALL S1 (165,545), seed 20260926, never used for
fitting or early stopping (reconstructed exactly as plan/checks/train_final_single.py does).
Test = random 200,000 test S1 (seed 0), all three countries.

Sections (all written to artifacts/metrics/test_shift_diag.json):
  A  list shape per country: stage-2 list size, p_rerank by rank, matcher p_cal counts / uncertain band
  B  self-estimated F0.5 (Monte-Carlo labels ~ Bernoulli(p_cal), candidates only) vs actual on the reference
  C  performance estimator: entity-level LightGBM regressor trained on the reference (list stats -> per-entity
     F0.5 and -> "has a true match outside the candidate list"), applied to test per country
  D  adversarial validation: row classifier reference(US+India) vs test(US+India), and test France vs test US+India
  E  orphan attribution on the reference: which pool records produce the false positives (true owner = another
     S1 vs orphan with no owner); orphans per S1 are 1.215 on train and ~2.29 on test if cluster sizes are equal

Run from the repository root:
  PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/test_shift_diag.py
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from ber import io_utils, match
from dev_rest_full import pick_remaining_slice

SEED_CAL = 20260926
N_TEST = 200_000
MC_DRAWS = 20
BAND = (0.1, 0.9)


def cal_queries():
    s1_rest, s1_180k = pick_remaining_slice()
    qs = [match.set_queries("MATCH_TRAIN"), match.set_queries("MATCH_VALID"), s1_180k, s1_rest]
    queries = np.sort(np.concatenate(qs)).astype(np.int32)
    perm = np.random.default_rng(SEED_CAL).permutation(queries)
    return np.sort(perm[:int(0.10 * queries.size)])


def read_rows(set_names, want):
    """Feature rows of the S1 in `want`, from the given feature sets, sorted by (s1_idx, rank)."""
    cols = ["s1_idx", "pool_idx"] + match.FEATURES
    out = []
    for name in set_names:
        for p in sorted(match.features_dir(name).glob("part-*.parquet")):
            s1 = pq.read_table(p, columns=["s1_idx"]).column("s1_idx").to_numpy()
            keep = np.isin(s1, want)
            if keep.any():
                out.append(pq.read_table(p, columns=cols).filter(keep).to_pandas())
    df = pd.concat(out, ignore_index=True)
    return df.sort_values(["s1_idx", "rank"], kind="stable").reset_index(drop=True)


def entity_table(df, p_cal, thr):
    """One row per S1: list statistics usable on both labeled and unlabeled sets (no labels, no country).
    df rows are sorted by (s1_idx, rank)."""
    s1 = df["s1_idx"].to_numpy()
    prr = df["p_rerank"].to_numpy().astype(np.float64)
    p = p_cal.astype(np.float64)
    q, start, cnt = np.unique(s1, return_index=True, return_counts=True)
    pos = np.arange(s1.size) - np.repeat(start, cnt)
    red = lambda v: np.add.reduceat(v, start)
    e = pd.DataFrame(index=q)
    e["size"] = cnt
    e["p_max"] = np.maximum.reduceat(p, start)
    e["p_sum"] = red(p)
    e["n_pos"] = red((p >= thr).astype(np.float64))
    e["n_band"] = red(((p >= BAND[0]) & (p < BAND[1])).astype(np.float64))
    e["prr_n05"] = red((prr >= 0.5).astype(np.float64))
    last = start + cnt - 1
    e["p_last"], e["prr_last"] = p[last], prr[last]
    order = np.lexsort((-p, s1))
    p_sorted = p[order]
    for k in range(6):
        has = cnt > k
        a = np.full(q.size, np.nan); b = np.full(q.size, np.nan)
        a[has] = p_sorted[start[has] + k]
        b[has] = prr[start[has] + k]
        e[f"p{k+1}"], e[f"prr{k+1}"] = a, b
    e["gap12"] = e["p1"] - e["p2"]
    e["last_pos"] = (e["p_last"] >= thr).astype(np.int8)
    return e


def mc_expected_f(df, p_cal, thr, rng):
    """Mean per-entity F0.5 of the threshold decision when labels ~ Bernoulli(p_cal), candidates only."""
    s1 = df["s1_idx"].to_numpy()
    q, qpos = np.unique(s1, return_inverse=True)
    keep = p_cal >= thr
    n_pred = np.bincount(qpos, weights=keep, minlength=q.size)
    fs = np.zeros(q.size)
    for _ in range(MC_DRAWS):
        y = rng.random(p_cal.size) < p_cal
        fs += match.entity_f05(n_pred, np.bincount(qpos, weights=y, minlength=q.size),
                               np.bincount(qpos, weights=keep & y, minlength=q.size))
    return pd.Series(fs / MC_DRAWS, index=q)


def summarize(e, per_s1_f=None):
    out = {"n_s1": int(len(e)), "size_mean": float(e["size"].mean()), "share_size_gt5": float((e["size"] > 5).mean()),
           "share_size_ge10": float((e["size"] >= 10).mean()),
           "prr_by_rank_mean": [float(e[f"prr{k+1}"].mean()) for k in range(6)],
           "prr_n05_mean": float(e["prr_n05"].mean()),
           "pred_size_mean": float(e["n_pos"].mean()), "share_pred_empty": float((e["n_pos"] == 0).mean()),
           "n_band_mean": float(e["n_band"].mean()), "share_any_band": float((e["n_band"] > 0).mean()),
           "p_max_mean": float(e["p_max"].mean()), "share_last_slot_pos": float(e["last_pos"].mean()),
           "share_all_slots_pos": float((e["n_pos"] == e["size"]).mean())}
    if per_s1_f is not None:
        out["mc_expected_f05"] = float(per_s1_f.reindex(e.index).mean())
    return out


def main():
    t0 = time.time()
    cfg = io_utils.load_config()
    doc = io_utils.read_json(io_utils.work_path("match", "models", "matcher.json"))
    if doc.get("run") != "full_single":
        raise RuntimeError(f"live matcher is {doc.get('run')}, expected full_single")
    thr = doc["decision"]["threshold"]
    feats = doc["features_used"]
    booster = lgb.Booster(model_file=str(io_utils.work_path("match", "models", "model_fold0.txt")))
    rng = np.random.default_rng(0)
    res = {"script": "plan/checks/test_shift_diag.py", "threshold": thr, "one_owner": "not applied (sampled sets)",
           "model": "work/match/models (full_single)"}

    # ---- reference (labeled) ----
    q_cal = cal_queries()
    ref = read_rows(["MATCH_TRAIN", "MATCH_VALID", "DEV_REST_180K", "DEV_REST_REST"], q_cal)
    ref_p = match.calibrate(booster.predict(ref[feats].to_numpy(np.float32)), doc["calibration"])
    y = match.pair_labels(ref["s1_idx"].to_numpy(), ref["pool_idx"].to_numpy())
    n_true = pd.Series(match.true_counts(q_cal), index=q_cal)
    cty_tr = pd.read_parquet(io_utils.work_path("records", "train_s1.parquet"), columns=["country_raw"])["country_raw"].to_numpy()
    print(f">>> reference: {q_cal.size} S1, {len(ref)} rows ({time.time()-t0:.0f}s)", flush=True)

    # ---- test sample (inputs only) ----
    io_utils.log_test_usage("matcher features + full_single scores for 200,000 random test S1 (inputs only, no labels)",
                            "RCA of the internal-vs-leaderboard gap: list-shape, confidence, adversarial validation "
                            "(plan/checks/test_shift_diag.py)", str(match.features_dir("TEST")))
    cty_te = pd.read_parquet(io_utils.work_path("records", "test_s1.parquet"), columns=["country_raw"])["country_raw"].to_numpy()
    q_te = np.sort(rng.choice(cty_te.size, N_TEST, replace=False)).astype(np.int32)
    te = read_rows(["TEST"], q_te)
    te_p = match.calibrate(booster.predict(te[feats].to_numpy(np.float32)), doc["calibration"])
    print(f">>> test sample: {N_TEST} S1, {len(te)} rows ({time.time()-t0:.0f}s)", flush=True)

    # ---- A/B: list shape and self-estimated F ----
    e_ref, e_te = entity_table(ref, ref_p, thr), entity_table(te, te_p, thr)
    mc_ref, mc_te = mc_expected_f(ref, ref_p, thr, rng), mc_expected_f(te, te_p, thr, rng)

    s1r = ref["s1_idx"].to_numpy()
    qpos = np.searchsorted(q_cal, s1r)
    keep = ref_p >= thr
    n_pred = np.bincount(qpos, weights=keep, minlength=q_cal.size)
    n_tp = np.bincount(qpos, weights=keep & y, minlength=q_cal.size)
    n_cpos = np.bincount(qpos, weights=y, minlength=q_cal.size)
    f_full = pd.Series(match.entity_f05(n_pred, n_true.to_numpy(), n_tp), index=q_cal)
    f_incand = pd.Series(match.entity_f05(n_pred, n_cpos, n_tp), index=q_cal)
    blk_miss = pd.Series((n_true.to_numpy() > n_cpos).astype(np.int8), index=q_cal)

    A = {}
    for c in ["US", "India"]:
        idx = q_cal[cty_tr[q_cal] == c]
        ee = e_ref.reindex(idx).dropna(subset=["size"])
        s = summarize(ee, mc_ref)
        s.update({"actual_f05_full": float(f_full.reindex(idx).mean()),
                  "actual_f05_in_candidates": float(f_incand.reindex(idx).mean()),
                  "share_blocking_miss": float(blk_miss.reindex(idx).mean()),
                  "true_links_per_s1": float(n_true.reindex(idx).mean())})
        A[f"ref_{c}"] = s
    A["ref_all"] = summarize(e_ref, mc_ref)
    A["ref_all"].update({"actual_f05_full": float(f_full.mean()), "actual_f05_in_candidates": float(f_incand.mean()),
                         "share_blocking_miss": float(blk_miss.mean())})
    for c in ["US", "India", "France"]:
        idx = q_te[cty_te[q_te] == c]
        A[f"test_{c}"] = summarize(e_te.reindex(idx).dropna(subset=["size"]), mc_te)
    A["test_all"] = summarize(e_te, mc_te)
    # conditional: when the last kept slot is predicted positive, how often is a true match outside the list?
    lp = e_ref["last_pos"].reindex(q_cal).fillna(0).astype(bool).to_numpy()
    A["ref_blocking_miss_given_last_slot_pos"] = {"last_pos": float(blk_miss.to_numpy()[lp].mean()),
                                                  "not_last_pos": float(blk_miss.to_numpy()[~lp].mean())}
    res["A_list_shape"] = A
    print(f">>> A/B done ({time.time()-t0:.0f}s)", flush=True)

    # ---- C: performance estimator trained on the reference, applied to test ----
    fcols = [c for c in e_ref.columns]
    Xr = e_ref[fcols].to_numpy(np.float32)
    tgt_f = f_full.reindex(e_ref.index).to_numpy()
    tgt_b = blk_miss.reindex(e_ref.index).to_numpy()
    prm = {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200, "verbose": -1, "seed": 0,
           "num_threads": io_utils.n_workers(cfg)}
    oof_f, oof_b = np.zeros(len(Xr)), np.zeros(len(Xr))
    for tr, va in GroupKFold(5).split(Xr, groups=e_ref.index.to_numpy()):
        m1 = lgb.train(dict(prm, objective="regression"), lgb.Dataset(Xr[tr], tgt_f[tr]), 400)
        m2 = lgb.train(dict(prm, objective="binary"), lgb.Dataset(Xr[tr], tgt_b[tr]), 400)
        oof_f[va], oof_b[va] = m1.predict(Xr[va]), m2.predict(Xr[va])
    m1 = lgb.train(dict(prm, objective="regression"), lgb.Dataset(Xr, tgt_f), 400)
    m2 = lgb.train(dict(prm, objective="binary"), lgb.Dataset(Xr, tgt_b), 400)
    Xt = e_te[fcols].to_numpy(np.float32)
    pf_te = pd.Series(m1.predict(Xt), index=e_te.index)
    pb_te = pd.Series(m2.predict(Xt), index=e_te.index)
    C = {"note": "estimator trained on the reference's list stats; valid only under covariate shift "
                 "(same P(label | list stats) on test). S1 with no candidates count as F=0 (none in either set).",
         "ref_oof_check": {}}
    for c in ["US", "India"]:
        m = cty_tr[e_ref.index.to_numpy()] == c
        C["ref_oof_check"][c] = {"actual_f": float(tgt_f[m].mean()), "est_f": float(oof_f[m].mean()),
                                 "actual_blk_miss": float(tgt_b[m].mean()), "est_blk_miss": float(oof_b[m].mean()),
                                 "blk_miss_auc": float(roc_auc_score(tgt_b[m], oof_b[m]))}
    for c in ["US", "India", "France"]:
        m = cty_te[e_te.index.to_numpy()] == c
        C[f"test_{c}"] = {"est_f": float(pf_te[m].mean()), "est_blk_miss": float(pb_te[m].mean())}
    shares = {c: float((cty_te == c).mean()) for c in ["US", "India", "France"]}
    C["test_country_share"] = shares
    C["test_est_f_weighted"] = float(sum(shares[c] * C[f"test_{c}"]["est_f"] for c in shares))
    C["public_leaderboard_sub2"] = 0.966797
    res["C_performance_estimate"] = C
    print(f">>> C done ({time.time()-t0:.0f}s)", flush=True)

    # ---- D: adversarial validation ----
    def adv(a, b, tag):
        X = np.vstack([a[feats].to_numpy(np.float32), b[feats].to_numpy(np.float32)])
        lab = np.r_[np.zeros(len(a)), np.ones(len(b))]
        grp = np.r_[a["s1_idx"].to_numpy().astype(np.int64), b["s1_idx"].to_numpy().astype(np.int64) + 10**8]
        sub = np.random.default_rng(1).random(len(lab)) < min(1.0, 1_200_000 / len(lab))
        X, lab, grp = X[sub], lab[sub], grp[sub]
        oof = np.zeros(len(lab))
        gain = np.zeros(len(feats))
        for tr, va in GroupKFold(3).split(X, groups=grp):
            m = lgb.train(dict(prm, objective="binary"), lgb.Dataset(X[tr], lab[tr], feature_name=feats), 200)
            oof[va] = m.predict(X[va])
            gain += m.feature_importance(importance_type="gain")
        top = np.argsort(-gain)[:15]
        drift = {}
        for f in feats:
            xa, xb = a[f].to_numpy(np.float64), b[f].to_numpy(np.float64)
            sd = np.nanstd(np.r_[xa, xb]) or 1.0
            drift[f] = {"mean_a": float(np.nanmean(xa)), "mean_b": float(np.nanmean(xb)),
                        "smd": float((np.nanmean(xb) - np.nanmean(xa)) / sd),
                        "nan_a": float(np.isnan(xa).mean()), "nan_b": float(np.isnan(xb).mean())}
        top_smd = sorted(drift, key=lambda f: -abs(drift[f]["smd"]))[:15]
        return {"auc": float(roc_auc_score(lab, oof)), "n_rows": int(lab.size),
                "top_gain": [[feats[i], float(gain[i] / gain.sum())] for i in top],
                "top_smd": [[f, drift[f]] for f in top_smd]}

    ref_c = cty_tr[ref["s1_idx"].to_numpy()]
    te_c = cty_te[te["s1_idx"].to_numpy()]
    res["D_adversarial"] = {
        "ref_USIndia_vs_test_USIndia": adv(ref, te[te_c != "France"], "ui"),
        "test_USIndia_vs_test_France": adv(te[te_c != "France"], te[te_c == "France"], "fr"),
        "ref_US_vs_ref_India": adv(ref[ref_c == "US"], ref[ref_c == "India"], "c"),
    }
    print(f">>> D done ({time.time()-t0:.0f}s)", flush=True)

    # ---- E: orphan attribution on the reference ----
    gt = pq.read_table(io_utils.work_path("records", "train_gt.parquet"))
    n_pool = pq.ParquetFile(io_utils.work_path("records", "train_pool.parquet")).metadata.num_rows
    owner = np.full(n_pool, -1, dtype=np.int64)
    owner[gt.column("pool_idx").to_numpy().astype(np.int64)] = gt.column("s1_idx").to_numpy()
    own = owner[ref["pool_idx"].to_numpy().astype(np.int64)]
    status = np.where(y, "true", np.where(own < 0, "orphan", "other_s1"))
    fp = keep & ~y
    fp_orphan_s1 = np.bincount(qpos, weights=fp & (status == "orphan"), minlength=q_cal.size) > 0
    fp_other_s1 = np.bincount(qpos, weights=fp & (status == "other_s1"), minlength=q_cal.size) > 0
    loss = 1 - f_full.to_numpy()
    E = {"candidate_row_share": {s: float((status == s).mean()) for s in ["true", "orphan", "other_s1"]},
         "fp_row_share": {s: float((status[fp] == s).mean()) for s in ["orphan", "other_s1"]},
         "n_fp_rows": int(fp.sum()),
         "loss_s1_with_orphan_fp_only": float(loss[fp_orphan_s1 & ~fp_other_s1].sum() / loss.size),
         "loss_s1_with_other_fp_only": float(loss[fp_other_s1 & ~fp_orphan_s1].sum() / loss.size),
         "loss_s1_with_both": float(loss[fp_orphan_s1 & fp_other_s1].sum() / loss.size),
         "loss_total": float(loss.mean()),
         "orphan_share_of_train_pool": float((owner < 0).mean()),
         "orphan_per_s1_train": float((owner < 0).sum() / cty_tr.size),
         "orphan_per_s1_test_if_equal_cluster_size": float(9969589 / 1732544 - 3.461)}
    res["E_orphans"] = E
    res["wall_s"] = round(time.time() - t0, 1)
    io_utils.write_metrics("test_shift_diag", res, int(q_cal.size), int(n_true.sum()), cfg)
    print(f">>> wrote artifacts/metrics/test_shift_diag.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
