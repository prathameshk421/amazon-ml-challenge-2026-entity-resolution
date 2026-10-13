"""CE-derived feature ablation summary (M020..M026 vs the 68-column base x1_strsim and vs A = M020).

  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/cefe_ablation.py M018 M020 ...

Reads each run's OOF (work/match/runs/<id>/) and its exp_<id>.json decision; writes
artifacts/metrics/cefe_ablation.json with, per run: entity F0.5 on all MATCH_TRAIN S1 and on the segments below,
paired delta vs x1_strsim and vs M020 with 95% bootstrap CI, in-list FP / FN pair counts, fold convergence.
Also: per CE column standalone AUC, |Spearman| with ce_p and with the closest base column, and mutual information
with the label on a 200K-row sample (redundancy check; the decision is the paired F0.5, not these).

Segments (S1-level, label-free except singleton):
  band        S1 with >= 1 candidate whose x1_strsim OOF p_cal is in [0.1, 0.9)  (exp.band_mask)
  empty_addr  S1 with >= 1 candidate whose pool address is empty (a_empty == 1)
  hi_name     S1 with >= 2 candidates at name token-set >= 0.95 (same-name competition)
  singleton   S1 with no true match
"""
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import exp, io_utils, match

BASE, REF = "x1_strsim", "M020"


def load_run(run):
    rd = match.runs_dir(run)
    t = pq.read_table(rd / "oof.parquet").to_pandas()
    m = json.load(open(io_utils.paths()["metrics_root"] / f"exp_{run}.json")) \
        if run != BASE else None
    return rd, t, m


def main(runs):
    cfg, fp = io_utils.load_config(), None
    fp = match.bucketing_fingerprint(cfg)
    X, names, s1, pool, _ = exp.full_matrix("MATCH_TRAIN", ["cex_all:ce_rec"], fp)
    col = {n: i for i, n in enumerate(names)}
    queries = np.load(match.runs_dir(BASE) / "queries.npy")

    def s1_mask(rows):
        return np.isin(queries, np.unique(s1[rows]))
    same = X[:, col["n_tset"]] >= 0.95
    n_same = pd.Series(same.astype(int)).groupby(s1).transform("sum").to_numpy()
    segs = {"band": exp.band_mask(queries), "empty_addr": s1_mask(X[:, col["a_empty"]] == 1),
            "hi_name": s1_mask(n_same >= 2), "singleton": match.true_counts(queries) == 0}
    fbase = np.load(match.runs_dir(BASE) / "OOF_per_entity.npy")
    fref = np.load(match.runs_dir(REF) / "OOF_per_entity.npy") if REF in runs else None

    out = {"base": BASE, "ref": REF, "segments_share": {k: float(v.mean()) for k, v in segs.items()}, "runs": {}}
    for run in [BASE] + runs:
        rd, t, m = load_run(run)
        if not np.array_equal(np.load(rd / "queries.npy"), queries):
            raise RuntimeError(f"{run}: different queries")
        f = np.load(rd / "OOF_per_entity.npy")
        dec = m["decision"] if m else json.load(open(rd / "models" / "matcher.json"))["decision"]
        pred = match.apply_decision(dec, t["s1_idx"].to_numpy(), t["pool_idx"].to_numpy(), t["p_cal"].to_numpy())
        y = t["y"].to_numpy()
        r = {"f05": float(f.mean()), "decision": dec, "fp": int((pred & ~y).sum()), "fn_in_list": int((~pred & y).sum()),
             "seg_f05": {k: float(f[v].mean()) for k, v in segs.items()}}
        if run != BASE:
            r["vs_base"] = match.paired_delta(f, fbase, n_boot=2000)
            r["vs_base_seg"] = {k: match.paired_delta(f[v], fbase[v], n_boot=1000) for k, v in segs.items()}
            if fref is not None and run != REF:
                r["vs_ref"] = match.paired_delta(f, fref, n_boot=2000)
                r["vs_ref_seg"] = {k: match.paired_delta(f[v], fref[v], n_boot=1000) for k, v in segs.items()}
            r["folds_best_iteration"] = [x["best_iteration"] for x in m["folds"]]
            r["any_fold_truncated"] = m["any_fold_truncated"]
            r["n_features"] = len(m["features_used"])
        out["runs"][run] = r
        print(f"{run:10s} F0.5 {r['f05']:.5f}  FP {r['fp']:6d}  FN {r['fn_in_list']:6d}  "
              + "  ".join(f"{k} {v:.5f}" for k, v in r["seg_f05"].items())
              + (f"  Δbase {r['vs_base']['delta']:+.5f} {np.round(r['vs_base']['ci95'], 5).tolist()}" if 'vs_base' in r else "")
              + (f"  ΔA {r['vs_ref']['delta']:+.5f} {np.round(r['vs_ref']['ci95'], 5).tolist()}" if 'vs_ref' in r else ""),
              flush=True)

    # redundancy: per CE column
    from scipy.stats import spearmanr
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import roc_auc_score
    t = pq.read_table(match.runs_dir(BASE) / "oof.parquet", columns=["s1_idx", "pool_idx", "y"]).to_pandas()
    key = lambda a, b: a.astype(np.int64) << 32 | b.astype(np.int64)
    yy = pd.Series(t["y"].to_numpy(), index=key(t["s1_idx"].to_numpy(), t["pool_idx"].to_numpy())) \
        .reindex(key(s1, pool)).to_numpy().astype(bool)
    rng = np.random.default_rng(0)
    samp = rng.choice(yy.size, size=200_000, replace=False)
    base_idx = [col[n] for n in match.FEATURES]
    cex = exp.CEX_FAMILIES["all"]
    Xs = X[samp].astype(np.float64)
    mi = mutual_info_classif(np.nan_to_num(Xs[:, [col[c] for c in cex]], nan=-9), yy[samp], random_state=0)
    red = {}
    for j, c in enumerate(cex):
        v = Xs[:, col[c]]
        ok = ~np.isnan(v)
        auc = roc_auc_score(yy[samp][ok], v[ok]) if ok.sum() > 1000 else float("nan")
        rho_p = abs(spearmanr(v[ok], Xs[ok, col["ce_p"]]).statistic)
        rb = [abs(spearmanr(v[ok], Xs[ok, b]).statistic) for b in base_idx]
        rb = np.nan_to_num(np.array(rb))
        red[c] = {"auc": float(max(auc, 1 - auc)), "abs_spearman_ce_p": float(rho_p),
                  "max_abs_spearman_base": float(rb.max()), "closest_base": match.FEATURES[int(rb.argmax())],
                  "mi": float(mi[j])}
    if "M026" in out["runs"]:
        gain = json.load(open(io_utils.paths()["metrics_root"] / "exp_M026.json"))["feature_gain"]
        gd = dict(gain) if not isinstance(gain, dict) else gain
        for c in cex:
            red[c]["gain_M026"] = float(gd.get(c, 0.0))
    out["redundancy"] = red
    io_utils.write_metrics("cefe_ablation", out, queries.size, int(match.true_counts(queries).sum()), cfg)
    print("wrote artifacts/metrics/cefe_ablation.json")


if __name__ == "__main__":
    main(sys.argv[1:])
