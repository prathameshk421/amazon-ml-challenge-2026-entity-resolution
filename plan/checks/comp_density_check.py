"""Robustness of M005 (pool-contention features) to fewer competing S1 lists.

Test US/India rows are less contested than train rows (mean c_log_n_own 0.26 / 0.32 vs 0.46 on MATCH_TRAIN,
work/exp/groups/*/comp.parquet), because the test pool has more unclaimed records per S1. A model trained with
denser contention could over-trust "uncontested" rows on test. Check on train labels: rebuild `comp` for
MATCH_TRAIN with the competitor universe thinned (all MATCH_TRAIN S1 kept, a random share of the OTHER train
S1 lists dropped), score each MATCH_TRAIN row with the M005 fold model that did not see it, apply M005's
decision, and compare with M005's own OOF (full universe) and x1_strsim (no contention).

No model is trained. Run:
  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/comp_density_check.py
"""
import lightgbm as lgb
import numpy as np

from ber import exp, io_utils, match

KEEP_OTHER = [0.5, 0.25]


def main():
    cfg, mcfg = io_utils.load_config(), match.load_match_config()
    fp = match.bucketing_fingerprint(cfg)
    X, s1, pool, _ = match.load_features("MATCH_TRAIN", fp)
    y = match.pair_labels(s1, pool)
    queries = match.set_queries("MATCH_TRAIN")
    sc = match.Scorer(queries, s1, y, match.true_counts(queries))
    md = match.runs_dir("M005") / "models"
    doc = io_utils.read_json(md / "matcher.json")
    boosters = [lgb.Booster(model_file=str(md / f"model_fold{k}.txt")) for k in range(doc["n_folds"])]
    folds = exp._folds(y, s1, mcfg)
    base_u = exp._universe
    mt = set(queries.tolist())
    out = {"script": "plan/checks/comp_density_check.py",
           "f05_M005_full_universe": float(np.load(match.runs_dir("M005") / "OOF_per_entity.npy").mean()),
           "f05_x1_strsim": float(np.load(match.runs_dir("x1_strsim") / "OOF_per_entity.npy").mean())}
    base = np.load(match.runs_dir("x1_strsim") / "OOF_per_entity.npy")
    for keep in KEEP_OTHER:
        rng = np.random.default_rng(0)

        def thin(set_name, keep=keep, rng=rng):
            u = base_u(set_name)
            other = ~u["s1"].isin(mt)
            s = u.loc[other, "s1"].unique()
            kept = set(rng.choice(s, int(keep * s.size), replace=False).tolist())
            return u[~other | u["s1"].isin(kept)]

        exp._universe = thin
        G = exp.group_comp("MATCH_TRAIN", X, s1, pool)
        exp._universe = base_u
        Xf = np.hstack([X, G])
        names = list(match.FEATURES) + exp.GROUPS["comp"][0]
        Xu = Xf[:, [names.index(f) for f in doc["features_used"]]]
        raw = np.zeros(y.size)
        for k, tr, te, _, _ in folds:
            raw[te] = boosters[k].predict(Xu[te])
        p = match.calibrate(raw, doc["calibration"])
        f = sc.per_entity(match.apply_decision(doc["decision"], s1, pool, p))[0]
        out[f"keep_other_{keep}"] = {"f05": float(f.mean()), "mean_c_log_n_own": float(np.nanmean(G[:, 0])),
                                     "vs_x1_strsim": match.paired_delta(f, base)}
        print(keep, out[f"keep_other_{keep}"], flush=True)
    io_utils.write_metrics("comp_density_check", out, queries.size, int(sc.n_true.sum()), cfg)


if __name__ == "__main__":
    main()
