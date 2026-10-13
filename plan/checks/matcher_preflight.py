"""Matcher preflight: facts the stage-3 matcher plan (plan/MATCHER_PLAN.md) depends on.

Reads only train-S1 work slices MATCH_TRAIN / MATCH_VALID (labels allowed by split.md rule 2).
Never reads val-S1 (HOLDOUT_TEST) labels. Writes artifacts/metrics/matcher_preflight.json.

Measures:
  1. positive rate inside the stage-2 candidate lists
  2. "re-ranker-score floor": entity F0.5 of a global threshold on p_rerank (MATCH_VALID, step 0.01)
  3. F0.5 ceiling (perfect matcher on these candidates)
  4. one-owner rule: a pool record listed by >= 2 S1 queries (inside MATCH_TRAIN u MATCH_VALID)
     is kept only for the S1 with the highest p_rerank; effect on MATCH_VALID F0.5 at the best threshold.
     Only 200,000 of 2,206,821 train S1 are in these sets, so collision counts are a LOWER bound.

Run from repo root: .venv/bin/python plan/checks/matcher_preflight.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
OUT = ROOT / "artifacts" / "metrics" / "matcher_preflight.json"


def load_cands(name):
    parts = sorted((WORK / "candidates" / name).glob("part-*.parquet"))
    return pd.concat([pq.read_table(p).to_pandas() for p in parts], ignore_index=True)


def f05(n_pred, n_true, n_tp):
    """Vectorised per-entity F0.5 with the contest singleton rule."""
    n_pred, n_true, n_tp = (np.asarray(a, dtype=np.float64) for a in (n_pred, n_true, n_tp))
    out = np.zeros_like(n_pred)
    single = n_true == 0
    out[single] = (n_pred[single] == 0).astype(np.float64)
    m = (~single) & (n_tp > 0)
    out[m] = 1.25 * n_tp[m] / (n_pred[m] + 0.25 * n_true[m])  # == 1.25PR/(0.25P+R)
    return out


def entity_scores(queries, df, keep):
    """df rows = candidates with column y; keep = boolean mask of predicted rows."""
    g = pd.DataFrame({"s1_idx": df.s1_idx.values, "pred": keep, "tp": keep & df.y.values})
    agg = g.groupby("s1_idx")[["pred", "tp"]].sum()
    agg = agg.reindex(queries, fill_value=0)
    return f05(agg["pred"].values, n_true[queries], agg["tp"].values)


t0 = time.time()
qs = json.load(open(WORK / "splits" / "query_sets.json"))["sets"]
gt = pq.read_table(WORK / "records" / "train_gt.parquet", columns=["s1_idx", "pool_idx"]).to_pandas()
n_s1 = int(gt.s1_idx.max()) + 1
n_true = np.bincount(gt.s1_idx.values, minlength=max(n_s1, 2_206_821))
gt_key = set((gt.s1_idx.values.astype(np.int64) << 32 | gt.pool_idx.values.astype(np.int64)).tolist())

res = {"script": "plan/checks/matcher_preflight.py", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
frames = {}
for name in ("MATCH_TRAIN", "MATCH_VALID"):
    df = load_cands(name)
    key = df.s1_idx.values.astype(np.int64) << 32 | df.pool_idx.values.astype(np.int64)
    df["y"] = np.fromiter((k in gt_key for k in key.tolist()), dtype=bool, count=len(key))
    q = np.array(sorted(qs[name]), dtype=np.int64)
    frames[name] = (q, df)
    ceil = entity_scores(q, df, df.y.values)
    res[name] = {
        "n_queries": int(q.size), "n_rows": int(len(df)), "n_pos": int(df.y.sum()),
        "pos_rate": float(df.y.mean()), "n_singletons": int((n_true[q] == 0).sum()),
        "n_true_pairs": int(n_true[q].sum()),
        "pair_recall": float(df.y.sum() / n_true[q].sum()),
        "f05_ceiling": float(ceil.mean()),
        "size_mean": float(len(df) / q.size),
    }

# 2. global p_rerank threshold on MATCH_VALID
qv, dv = frames["MATCH_VALID"]
ths = np.round(np.arange(0.05, 0.96, 0.01), 2)
sweep = {float(t): float(entity_scores(qv, dv, dv.p_rerank.values >= t).mean()) for t in ths}
t_best = max(sweep, key=sweep.get)
res["p_rerank_threshold_sweep_MATCH_VALID"] = sweep
res["p_rerank_floor"] = {"threshold": t_best, "f05": sweep[t_best],
                         "f05_empty_all": float(entity_scores(qv, dv, np.zeros(len(dv), bool)).mean())}

# 4. one-owner rule over MATCH_TRAIN u MATCH_VALID candidate rows
both = pd.concat([frames["MATCH_TRAIN"][1].assign(set="T"), frames["MATCH_VALID"][1].assign(set="V")],
                 ignore_index=True)
cnt = both.groupby("pool_idx").s1_idx.transform("size").values
best_p = both.groupby("pool_idx").p_rerank.transform("max").values
is_owner = both.p_rerank.values >= best_p  # ties keep all tied rows
coll = cnt >= 2
res["one_owner"] = {
    "note": "collisions measured inside 200,000 of 2,206,821 train S1 only -> lower bound",
    "rows_in_collision": int(coll.sum()), "share_rows_in_collision": float(coll.mean()),
    "rows_removed": int((~is_owner).sum()),
    "removed_pos": int((~is_owner & both.y.values).sum()),
    "removed_neg": int((~is_owner & ~both.y.values).sum()),
}
vmask = both.set.values == "V"
keep_v = (both.p_rerank.values >= t_best) & is_owner
res["one_owner"]["MATCH_VALID_f05_at_floor_threshold"] = {
    "without_rule": sweep[t_best],
    "with_rule": float(entity_scores(qv, both[vmask], keep_v[vmask]).mean()),
}

# 5. where the floor loses F0.5 on MATCH_VALID (loss = 1 - F0.5 per entity, summed / n_queries)
kf = dv.p_rerank.values >= t_best
g = pd.DataFrame({"s1_idx": dv.s1_idx.values, "pred": kf, "tp": kf & dv.y.values, "cand_pos": dv.y.values})
agg = g.groupby("s1_idx")[["pred", "tp", "cand_pos"]].sum().reindex(qv, fill_value=0)
nt = n_true[qv]
loss = 1.0 - f05(agg["pred"].values, nt, agg["tp"].values)
fp = agg["pred"].values - agg["tp"].values
fn_in = agg["cand_pos"].values - agg["tp"].values           # true pairs present in candidates but not predicted
fn_out = nt - agg["cand_pos"].values                         # true pairs outside candidates (blocking miss)
chain = np.load(WORK / "norm" / "train_s1_chain.npy")[qv]
buckets = {
    "singleton_with_fp": (nt == 0) & (fp > 0),
    "matched_fp_only": (nt > 0) & (fp > 0) & (fn_in == 0),
    "matched_fn_only": (nt > 0) & (fp == 0) & (fn_in > 0),
    "matched_fp_and_fn": (nt > 0) & (fp > 0) & (fn_in > 0),
    "blocking_miss_only": (nt > 0) & (fp == 0) & (fn_in == 0) & (fn_out > 0),
}
res["floor_loss_breakdown_MATCH_VALID"] = {
    "total_loss": float(loss.sum() / qv.size),
    **{k: {"n_entities": int(m.sum()), "loss_share_of_f05": float(loss[m].sum() / qv.size)}
       for k, m in buckets.items()},
    "by_chain": {lab: {"n_entities": int(m.sum()), "loss_share_of_f05": float(loss[m].sum() / qv.size)}
                 for lab, m in (("chain_1", chain <= 1), ("chain_2_9", (chain >= 2) & (chain <= 9)),
                                ("chain_ge10", chain >= 10))},
}
res["wall_s"] = round(time.time() - t0, 1)
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "p_rerank_threshold_sweep_MATCH_VALID"}, indent=1))
