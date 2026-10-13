"""Gate 1 evaluation (retrieval recall channel), MATCH_TRAIN S1 vs the full train pool. Labels used for scoring only.

  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/gate1_eval.py [--tag gate1]

"Missed" = a ground-truth (S1, pool) pair of a MATCH_TRAIN S1 that is not in its final candidate list
(work/candidates/MATCH_TRAIN; stage-1 miss or stage-2 cut). Reports, for S1-query top-k and seed (copy-expansion)
top-k: share of missed pairs retrieved, new pairs per S1 and their positive rate, and the ORACLE per-S1 F0.5 gain of
adding only the retrieved true pairs to M027's OOF predictions (upper bound; a real filter keeps fewer and adds FPs).
Writes artifacts/metrics/gate1_<tag>.json.
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match

SEED_BASE = 10 ** 9


def key(a, b):
    return np.asarray(a, dtype=np.int64) << 32 | np.asarray(b, dtype=np.int64)


def f05(T, P, TP):
    T, P, TP = (np.asarray(x, dtype=np.float64) for x in (T, P, TP))
    den = 1.25 * TP + 0.25 * (T - TP) + (P - TP)
    return np.where((T == 0) & (P == 0), 1.0, np.where(TP > 0, 1.25 * TP / np.maximum(den, 1e-12), 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gate1")
    args = ap.parse_args()
    d = io_utils.paths()["work_root"] / "retrieval"
    queries = np.load(match.runs_dir("M027") / "queries.npy")
    top = pq.read_table(d / f"{args.tag}_topk.parquet").to_pandas()
    seeds = pd.read_parquet(d / f"{args.tag}_seeds.parquet")
    qset = set(top.loc[top.qid < SEED_BASE, "qid"].unique())
    queries = queries[np.isin(queries, list(qset))]
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas()
    gt = gt[gt.s1_idx.isin(queries)]
    cand = pq.read_table("work/candidates/MATCH_TRAIN/part-00000.parquet", columns=["s1_idx", "pool_idx"]).to_pandas()
    cand = cand[cand.s1_idx.isin(queries)]
    in_list = set(key(cand.s1_idx, cand.pool_idx).tolist())
    gt_k = key(gt.s1_idx, gt.pool_idx)
    true_set = set(gt_k.tolist())
    missed = np.array([k not in in_list for k in gt_k.tolist()])
    missed_set = set(gt_k[missed].tolist())
    s1_country = pq.read_table("work/records/train_s1.parquet", columns=["s1_idx", "country_raw"]).to_pandas() \
        .set_index("s1_idx").country_raw.fillna("NA").astype(str).str.strip()

    # M027 OOF predictions and per-S1 counts
    oof = pq.read_table(match.runs_dir("M027") / "oof.parquet").to_pandas()
    oof = oof[oof.s1_idx.isin(queries)]
    pred = match.apply_decision({"rule": "threshold", "threshold": 0.73, "one_owner": True},
                                oof.s1_idx.to_numpy(), oof.pool_idx.to_numpy(), oof.p_cal.to_numpy())
    T = gt.groupby("s1_idx").size().reindex(queries, fill_value=0).to_numpy()
    P = pd.Series(pred.astype(int)).groupby(oof.s1_idx.to_numpy()).sum().reindex(queries, fill_value=0).to_numpy()
    TPs = pd.Series((pred & oof.y.to_numpy()).astype(int)).groupby(oof.s1_idx.to_numpy()).sum() \
        .reindex(queries, fill_value=0).to_numpy()
    base = f05(T, P, TPs)
    out = {"n_s1": int(queries.size), "n_true_pairs": int(len(gt)), "n_missed_pairs": int(missed.sum()),
           "missed_share": float(missed.mean()), "f05_M027": float(base.mean()),
           "s1_with_missed": int(pd.Series(missed).groupby(gt.s1_idx.to_numpy()).any().sum()), "rows": []}
    print(f"S1 {queries.size:,}; true pairs {len(gt):,}; missed {missed.sum():,} ({missed.mean():.2%}); "
          f"M027 F0.5 on these S1 {base.mean():.5f}", flush=True)

    s1q = top[top.qid < SEED_BASE].rename(columns={"qid": "s1_idx"})
    sq = top[top.qid >= SEED_BASE].merge(seeds[["qid", "s1_idx"]], on="qid")
    # sanity: in-list true pairs found by the S1 query at k=50 (prefix/pooling bugs show up here)
    k_all = key(s1q.s1_idx, s1q.pool_idx)
    inl_true = gt_k[~missed]
    out["sanity_inlist_true_recall_at50"] = float(np.isin(inl_true, k_all).mean())
    print(f"sanity: in-list true pairs retrieved by S1 query @50: {out['sanity_inlist_true_recall_at50']:.3f}", flush=True)

    qpos = pd.Series(np.arange(queries.size), index=queries)
    for ks, kd in ((1, 0), (5, 0), (10, 0), (20, 0), (50, 0), (0, 5), (0, 10), (0, 20), (5, 5), (10, 10), (20, 20)):
        parts = []
        if ks:
            a = s1q[s1q["rank"] <= ks]
            parts.append(key(a.s1_idx, a.pool_idx))
        if kd:
            b = sq[sq["rank"] <= kd]
            parts.append(key(b.s1_idx, b.pool_idx))
        kk = np.unique(np.concatenate(parts))
        new = kk[~np.isin(kk, np.fromiter(in_list, dtype=np.int64))]
        is_true = np.isin(new, np.fromiter(true_set, dtype=np.int64))
        found = np.isin(new, np.fromiter(missed_set, dtype=np.int64))
        s_new = (new >> 32).astype(np.int64)
        add_tp = pd.Series(found.astype(int)).groupby(s_new).sum().reindex(queries, fill_value=0).to_numpy()
        oracle = f05(T, P + add_tp, TPs + add_tp)
        n_new = pd.Series(1, index=s_new).groupby(level=0).size().reindex(queries, fill_value=0).to_numpy()
        sing = T == 0
        row = {"s1_k": ks, "seed_k": kd, "new_pairs": int(new.size), "new_per_s1": float(new.size / queries.size),
               "new_pos_rate": float(is_true.mean()) if new.size else 0.0,
               "missed_recovered": int(found.sum()), "missed_recall": float(found.sum() / max(missed.sum(), 1)),
               "oracle_f05": float(oracle.mean()), "oracle_gain": float(oracle.mean() - base.mean()),
               "new_per_singleton_s1": float(n_new[sing].mean()) if sing.any() else 0.0,
               "missed_recall_by_country": {c: float(found[np.isin(s_new, queries[(s1_country.reindex(queries).to_numpy() == c)])].sum()
                                                  / max(1, missed[np.isin(gt.s1_idx.to_numpy(), queries[(s1_country.reindex(queries).to_numpy() == c)])].sum()))
                                            for c in ("US", "India")}}
        out["rows"].append(row)
        print(f"S1@{ks:<2} seed@{kd:<2}: new {row['new_per_s1']:.2f}/S1 (pos {row['new_pos_rate']:.3f}), "
              f"missed recovered {row['missed_recovered']:,} = {row['missed_recall']:.1%} "
              f"(US {row['missed_recall_by_country']['US']:.1%}, India {row['missed_recall_by_country']['India']:.1%}); "
              f"oracle F0.5 gain {row['oracle_gain']:+.5f}", flush=True)
    io_utils.write_metrics(f"gate1_{args.tag}", out, queries.size, int(len(gt)))
    print(f"wrote artifacts/metrics/gate1_{args.tag}.json")


if __name__ == "__main__":
    main()
