"""D1: DRACE-specific scorer vs the fixed c0 >= 0.995 rule (Phase 8), MATCH_TRAIN, cross-fitted by S1 hash half.

  PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/drace_calib.py

Candidates: gate-2 new pairs restricted to what test used (S1 rank <= 5 or seed rank <= 2), minus pool records M027
already accepts. Features (all available on test): p_ce (c0), logit p_ce, cos_s1, cos_seed, s1_rank, seed_rank, from_s1,
from_seed, n_seed_paths, best seed p_cal, name token-set / address token-set (rapidfuzz), pool address empty.
Model: LightGBM binary, trained on one half's S1, scores the other half; then the add-rule threshold on the model score
is chosen on the training half by mean F0.5 and applied to the scored half. Baseline: c0 >= 0.995 on the same pairs.
Writes artifacts/metrics/drace_calib.json.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from ber import io_utils, match
from gate2_eval import f05

SB = 10 ** 9


def main():
    q = np.load(match.runs_dir("M027") / "queries.npy")
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas()
    gt = gt[gt.s1_idx.isin(q)]
    T = gt.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    oof = pq.read_table(match.runs_dir("M027") / "oof.parquet").to_pandas()
    pred = match.apply_decision({"rule": "threshold", "threshold": 0.73, "one_owner": True},
                                oof.s1_idx.to_numpy(), oof.pool_idx.to_numpy(), oof.p_cal.to_numpy())
    acc = oof[pred]
    P0 = acc.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    TP0 = acc[acc.y].groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    base = f05(T, P0, TP0)

    m = pd.read_parquet("work/retrieval/gate2_meta.parquet")
    m["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
    m = m[((m.s1_rank <= 5) | (m.seed_rank <= 2)) & ~m.pool_idx.isin(set(acc.pool_idx))].reset_index(drop=True)
    # seed provenance: number of accepted seeds reaching the pair within rank 2, best seed p_cal
    top = pq.read_table("work/retrieval/gate1_topk.parquet").to_pandas()
    seeds = pd.read_parquet("work/retrieval/gate1_seeds.parquet")
    sd = top[(top.qid >= SB) & (top["rank"] <= 2)].merge(seeds, on="qid", suffixes=("", "_seed"))
    seed_p = oof.set_index(["s1_idx", "pool_idx"]).p_cal
    sd["seed_p"] = seed_p.reindex(pd.MultiIndex.from_arrays([sd.s1_idx, sd.pool_idx_seed])).to_numpy()
    agg = sd.groupby(["s1_idx", "pool_idx"]).agg(n_paths=("qid", "nunique"), seed_p=("seed_p", "max")).reset_index()
    m = m.merge(agg, on=["s1_idx", "pool_idx"], how="left")
    s1t = pq.read_table("work/records/train_s1.parquet", columns=["name_raw", "addr_raw"]).to_pandas()
    pt = pq.read_table("work/records/train_pool.parquet", columns=["name_raw", "addr_raw"]).to_pandas()
    sn, sa = s1t.name_raw.fillna("").to_numpy()[m.s1_idx], s1t.addr_raw.fillna("").to_numpy()[m.s1_idx]
    pn, pa_ = pt.name_raw.fillna("").to_numpy()[m.pool_idx], pt.addr_raw.fillna("").to_numpy()[m.pool_idx]
    m["n_tset"] = [fuzz.token_set_ratio(a, b) / 100 for a, b in zip(sn, pn)]
    m["a_tset"] = [fuzz.token_set_ratio(a, b) / 100 if b.strip() else np.nan for a, b in zip(sa, pa_)]
    m["a_empty"] = [float(not b.strip()) for b in pa_]
    pc = np.clip(m.p.to_numpy(), 1e-6, 1 - 1e-6)
    m["logit"] = np.log(pc / (1 - pc))
    m["from_s1"], m["from_seed"] = m.s1_rank.notna().astype(float), m.seed_rank.notna().astype(float)
    feats = ["p", "logit", "cos_s1", "cos_seed", "s1_rank", "seed_rank", "from_s1", "from_seed", "n_paths", "seed_p",
             "n_tset", "a_tset", "a_empty"]
    half_q = ((q.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2 ** 32) >> np.uint64(31)).astype(int)
    hq = pd.Series(half_q, index=q)
    m["half"] = hq.reindex(m.s1_idx).to_numpy()
    y = m.label.to_numpy().astype(int)

    import lightgbm as lgb

    def add_f(sel):
        a = m[sel].sort_values("p", ascending=False).drop_duplicates("pool_idx")
        ad = a.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        tp = a[a.label == 1].groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        return f05(T, P0 + ad, TP0 + tp)
    rule = add_f(m.p >= 0.995)
    score = np.zeros(len(m))
    for h in (0, 1):
        tr, te = m.half != h, m.half == h
        b = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
                       "feature_fraction": 0.9, "verbose": -1, "seed": 42},
                      lgb.Dataset(m.loc[tr, feats], y[tr.to_numpy()]), num_boost_round=400)
        score[te.to_numpy()] = b.predict(m.loc[te, feats])
    m["s"] = score
    f_cf = base.copy()
    chosen = {}
    for h in (0, 1):  # threshold chosen on half h (whose scores came from a model trained on half 1-h), applied to 1-h
        grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        best = max(grid, key=lambda t: (add_f(m.s >= t) - base)[half_q == h].mean())
        chosen[h] = best
        f_cf[half_q == 1 - h] = add_f(m.s >= best)[half_q == 1 - h]
    d_rule = match.paired_delta(rule, base, n_boot=1000)
    d_model = match.paired_delta(f_cf, base, n_boot=1000)
    d_vs = match.paired_delta(f_cf, rule, n_boot=2000)
    from sklearn.metrics import roc_auc_score
    out = {"pairs": int(len(m)), "true": int(y.sum()), "auc_c0": float(roc_auc_score(y, m.p)),
           "auc_model": float(roc_auc_score(y, m.s)), "rule_c0_0995_vs_M027": d_rule, "model_vs_M027": d_model,
           "model_vs_rule": d_vs, "chosen_threshold": chosen}
    io_utils.write_metrics("drace_calib", out, q.size, int(T.sum()))
    print(f"pairs {len(m):,} ({y.sum():,} true); AUC c0 {out['auc_c0']:.5f}, model {out['auc_model']:.5f}")
    print(f"rule c0>=0.995 vs M027 {d_rule['delta']:+.5f}; model vs M027 {d_model['delta']:+.5f}; "
          f"MODEL vs RULE {d_vs['delta']:+.5f} [{d_vs['ci95'][0]:+.5f}, {d_vs['ci95'][1]:+.5f}]; thresholds {chosen}")


if __name__ == "__main__":
    main()
