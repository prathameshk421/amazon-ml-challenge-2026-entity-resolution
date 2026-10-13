"""Item 14 / RCA F1: which feature families help or hurt transfer to an unseen country (France proxy).

For each variant (a feature list) and each direction (train US -> score India, train India -> score US):
one LightGBM on the training country's MATCH_TRAIN S1 (90% fit, 10% early stop, same params and cap as the
matcher), isotonic calibration on the 10% early-stop rows, x1_strsim's threshold (0.67) + one-owner, entity
F0.5 over every MATCH_TRAIN S1 of the scored country. Paired per-entity delta vs the base-68 variant.
Country only selects rows; it is never a feature. This is a diagnostic: the fitted LOCO models are not
submission candidates (trained on one country only), so they are not kept.

Run after the MATCH_TRAIN group caches exist (comp, dinv, sib, num):
  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/loco_groups.py
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupShuffleSplit

from ber import exp, io_utils, match

DIRECTIONS = [("US", "India"), ("India", "US")]
FAMILIES = {
    "name_str": lambda f: f.startswith("n_") and f not in ("n_idf_jacc", "n_idf_qcov", "n_idf_pcov",
                                                           "n_q_miss_maxidf", "n_p_miss_maxidf", "n_shared_maxidf"),
    "idf": lambda f: "idf" in f,
    "addr_str": lambda f: f.startswith("a_") and "idf" not in f,
    "list": lambda f: f.startswith("l_"),
    "counts": lambda f: f.endswith("_cnt") or f in ("log_chain", "q_ntok", "p_ntok"),
    "context": lambda f: f in ("src", "country3", "pool_indic"),
}
GROUPS = ["comp", "dinv", "sib", "num"]


def main():
    t0 = time.time()
    lg = io_utils.get_logger("loco-groups")
    cfg, mcfg = io_utils.load_config(), match.load_match_config()
    Xf, names, s1, pool, _ = exp.full_matrix("MATCH_TRAIN", GROUPS, match.bucketing_fingerprint(cfg), lg)
    ci = {n: i for i, n in enumerate(names)}
    y = match.pair_labels(s1, pool)
    queries = match.set_queries("MATCH_TRAIN")
    country = pd.read_parquet(io_utils.work_path("records", "train_s1.parquet"), columns=["country_raw"])["country_raw"].to_numpy()
    base = list(match.FEATURES)
    variants = {"base68": base}
    for g in GROUPS:
        variants[f"+{g}"] = base + exp.GROUPS[g][0]
    for fam, fn in FAMILIES.items():
        variants[f"-{fam}"] = [f for f in base if not fn(f)]
    thr = io_utils.read_json(match.runs_dir("x1_strsim") / "models" / "matcher.json")["decision"]["threshold"]
    params = dict(mcfg["lgb"], num_threads=io_utils.n_workers(cfg))
    out = {}
    per = {}
    for vname, feats in variants.items():
        X = Xf[:, [ci[f] for f in feats]]
        out[vname] = {"n_features": len(feats)}
        for tc, sc in DIRECTIONS:
            tr = np.flatnonzero(country[s1] == tc)
            te = np.flatnonzero(country[s1] == sc)
            gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=mcfg["fold_seed"])
            fi, ei = next(gss.split(tr, groups=s1[tr]))
            fi, ei = tr[fi], tr[ei]
            dfit = lgb.Dataset(X[fi], label=y[fi], feature_name=feats, free_raw_data=False)
            des = lgb.Dataset(X[ei], label=y[ei], reference=dfit, free_raw_data=False)
            b = lgb.train(params, dfit, num_boost_round=mcfg["num_boost_round"], valid_sets=[des],
                          valid_names=["es"], callbacks=[lgb.early_stopping(mcfg["early_stopping_rounds"], verbose=False)])
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(b.predict(X[ei], num_iteration=b.best_iteration), y[ei])
            p = iso.predict(b.predict(X[te], num_iteration=b.best_iteration))
            keep = match.one_owner(pool[te], p) >= thr
            q = queries[country[queries] == sc]
            f = match.Scorer(q, s1[te], y[te], match.true_counts(q)).per_entity(keep)[0]
            per[(vname, tc)] = f
            out[vname][f"train_{tc}"] = {"f05": float(f.mean()), "best_iteration": int(b.best_iteration),
                                         "truncated": bool(b.best_iteration >= mcfg["num_boost_round"] - mcfg["early_stopping_rounds"])}
            if vname != "base68":
                out[vname][f"train_{tc}"]["vs_base68"] = match.paired_delta(f, per[("base68", tc)])
            lg.info("%s train %s: F0.5 %.5f (it %d) %s (%.0fs)", vname, tc, f.mean(), b.best_iteration,
                    out[vname][f"train_{tc}"].get("vs_base68", {}).get("delta", ""), time.time() - t0)
    io_utils.write_metrics("loco_groups", {"script": "plan/checks/loco_groups.py", "threshold": thr,
                                           "variants": out}, queries.size, int(match.true_counts(queries).sum()), cfg)


if __name__ == "__main__":
    main()
