"""Does swapping exact search for an ANN index in reverse-name retrieval change the live matcher's output?

Both top-k files must come from the SAME embeddings (retrieve_job.py RETR_INDEX=hnsw RETR_COMPARE=1), so the only
difference is the search. Steps: rev features for <SET> from each file (rev_prep.feats) -> rows whose features differ ->
if any, predict <SET> with the run's models twice (rev from each file), apply its decision rule, and count S1 whose
predicted set changes. |ΔF0.5| <= changed S1 / all S1 (macro-average over S1; a changed S1 moves by at most 1).

  PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/ann_rev_check.py \
      --exact <x_exact_topk.parquet> --ann <x_topk.parquet> --set TEST --run full_c0_rgrp --work <scratch dir>
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import rev_prep
from ber import exp, io_utils, match


def feats(topk, set_name, d):
    d.mkdir(parents=True, exist_ok=True)
    split = "test" if set_name == "TEST" else "train"
    link = d / f"rev_{split}_topk.parquet"
    link.unlink(missing_ok=True)
    link.symlink_to(Path(topk).resolve())
    rev_prep.R = d
    rev_prep.feats(set_name)
    return pq.read_table(d / f"rev_{set_name}.parquet").to_pandas()


def predict(set_name, run, d):
    cols, _ = exp.GROUPS["rev"]

    def group_rev(set_name, X, s1, pool):
        t = pq.read_table(d / f"rev_{set_name}.parquet")
        if not (np.array_equal(t.column("s1_idx").to_numpy(), s1) and np.array_equal(t.column("pool_idx").to_numpy(), pool)):
            raise RuntimeError("rev rows not in feature-row order")
        return np.column_stack([t.column(c).to_numpy() for c in cols])

    exp.GROUPS["rev"] = (cols, group_rev)
    models = match.runs_dir(run) / "models"
    doc = io_utils.read_json(models / "matcher.json")
    _, _, s1, pool, _, p_cal, _ = exp.predict_run(set_name, doc, models)
    return s1, pool, p_cal, match.apply_decision(doc["decision"], s1, pool, p_cal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--set", default="TEST")
    ap.add_argument("--run", default="full_c0_rgrp")
    ap.add_argument("--work", required=True)
    ap.add_argument("--tag", default="hnsw")
    a = ap.parse_args()
    t0 = time.time()
    cfg = io_utils.load_config()
    w = Path(a.work)
    fe, fa = feats(a.exact, a.set, w / "exact"), feats(a.ann, a.set, w / "ann")
    assert np.array_equal(fe.s1_idx, fa.s1_idx) and np.array_equal(fe.pool_idx, fa.pool_idx)
    cols = exp.GROUPS["rev"][0]
    diff = np.zeros(len(fe), bool)
    per_col = {}
    for c in cols:
        x, y = fe[c].to_numpy(), fa[c].to_numpy()
        dc = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        per_col[c] = int(dc.sum())
        diff |= dc
    n_s1 = int(fe.s1_idx.max()) + 1 if a.set != "TEST" else len(pq.read_table("work/records/test_s1.parquet", columns=["s1_idx"]))
    out = {"set": a.set, "run": a.run, "exact_topk": a.exact, "ann_topk": a.ann, "rows": len(fe),
           "rows_name_only": int(fe.rv_rank.notna().sum()), "rows_feature_differs": int(diff.sum()),
           "rows_differ_per_col": per_col, "s1_with_feature_diff": int(np.unique(fe.s1_idx.to_numpy()[diff]).size),
           "n_s1": n_s1}
    print(f"{a.set}: {diff.sum():,} of {len(fe):,} feature rows differ ({out['rows_name_only']:,} name-only); "
          f"per column {per_col}", flush=True)
    if diff.any():
        if a.set == "TEST":
            io_utils.log_test_usage(f"ann_rev_check {a.run}: matcher inference over test candidates with exact vs "
                                    f"{a.tag} reverse-name features (inputs only)", "ANN retrieval safety check",
                                    str(match.features_dir("TEST")))
        s1, pool, pe, ke = predict(a.set, a.run, w / "exact")
        _, _, pa_, ka = predict(a.set, a.run, w / "ann")
        ch = ke != ka
        s1_ch = np.unique(s1[ch])
        out.update({"pairs_accepted_exact": int(ke.sum()), "pairs_accepted_ann": int(ka.sum()),
                    "pairs_changed": int(ch.sum()), "s1_prediction_changed": int(s1_ch.size),
                    "max_abs_dp_cal": float(np.abs(pe - pa_).max()),
                    "f05_change_upper_bound": s1_ch.size / n_s1})
        print(f"accepted pairs exact {ke.sum():,} ann {ka.sum():,}; changed pairs {ch.sum():,}; "
              f"S1 with a changed prediction {s1_ch.size:,} of {n_s1:,} -> |dF0.5| <= {s1_ch.size / n_s1:.2e}", flush=True)
    else:
        out.update({"s1_prediction_changed": 0, "f05_change_upper_bound": 0.0})
    out["wall_s"] = round(time.time() - t0, 1)
    io_utils.write_metrics(f"ann_rev_check_{a.set}_{a.tag}", out, n_s1, 0, cfg)


if __name__ == "__main__":
    main()
