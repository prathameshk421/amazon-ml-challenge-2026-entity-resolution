"""Transitive-consistency cleaning (TransClean-style, arXiv 2506.04006) — train check on MATCH_TRAIN.

  PYTHONPATH=submission/code/business_entity_resolution/src:ce:plan/checks .venv/bin/python plan/checks/tc_train.py prep
  (HF) score_job.py CE_SETS=MATCH_VALID,TC_TRAIN (c0), collect
  ... tc_train.py eval

Base prediction per S1 = M027 OOF decision (threshold 0.73 + one-owner) + gate-2 DRACE additions at p_ce >= 0.995
(same one-owner handling as gate2_eval). For every S1 with >= 3 predicted pool records, c0 scores every ordered
pool-pool pair among them (both orders; pool text in both slots). A record's sibling score = mean over its siblings of
the two-order average p. Rule: drop a record when its sibling score < tau and (optionally) its own match score < cap.
(tau, cap) chosen on one hash half of the S1 and scored on the other, swapped; paired delta vs base with CI.
Labels only for choosing/scoring the rule. Writes artifacts/metrics/tc_train.json.
"""
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber import exp, io_utils, match
from gate2_eval import f05, key


def base_sets():
    queries = np.load(match.runs_dir("M027") / "queries.npy")
    oof = pq.read_table(match.runs_dir("M027") / "oof.parquet").to_pandas()
    pred = match.apply_decision({"rule": "threshold", "threshold": 0.73, "one_owner": True},
                                oof.s1_idx.to_numpy(), oof.pool_idx.to_numpy(), oof.p_cal.to_numpy())
    b = oof.loc[pred, ["s1_idx", "pool_idx", "y", "p_cal"]].rename(columns={"p_cal": "score"})
    owned = set(b.pool_idx.tolist())
    m = pd.read_parquet("work/retrieval/gate2_meta.parquet")
    m["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
    m = m[~m.pool_idx.isin(owned) & (m.p >= 0.995)].sort_values("p", ascending=False).drop_duplicates("pool_idx")
    add = pd.DataFrame({"s1_idx": m.s1_idx.to_numpy(), "pool_idx": m.pool_idx.to_numpy(),
                        "y": m.label.to_numpy().astype(bool), "score": m.p.to_numpy()})
    s = pd.concat([b.assign(src="lgb"), add.assign(src="drace")], ignore_index=True)
    s["s1_idx"] = s.s1_idx.astype(np.int64)
    s["pool_idx"] = s.pool_idx.astype(np.int64)
    return queries, s


def prep():
    from prepare_pairs import record_text
    queries, s = base_sets()
    n = s.groupby("s1_idx").size()
    big = s[s.s1_idx.isin(n[n >= 3].index)]
    pairs = big.merge(big, on="s1_idx", suffixes=("_a", "_b"))
    pairs = pairs[pairs.pool_idx_a != pairs.pool_idx_b]
    pairs = pairs.sort_values(["s1_idx", "pool_idx_a", "pool_idx_b"]).reset_index(drop=True)
    pool_txt = record_text("train", "pool")
    a, b = pairs.pool_idx_a.to_numpy(), pairs.pool_idx_b.to_numpy()
    lab = (pairs.y_a & pairs.y_b).to_numpy().astype(np.int8)
    pq.write_table(pa.table({"s1_idx": pa.array(pairs.s1_idx.to_numpy().astype(np.int32)),
                             "pool_idx": pa.array(b.astype(np.uint32)),
                             "rank": pa.array(np.zeros(len(pairs), dtype=np.int16)),
                             "text_a": pa.array(pool_txt[a].tolist(), pa.string()),
                             "text_b": pa.array(pool_txt[b].tolist(), pa.string()),
                             "label": pa.array(lab)}), "work/ce/pairs_TC_TRAIN.parquet", compression="zstd")
    pairs[["s1_idx", "pool_idx_a", "pool_idx_b"]].to_parquet("work/retrieval/tc_train_meta.parquet")
    print(f"base predicted pairs {len(s):,} ({(s.src == 'drace').sum():,} from DRACE); S1 with >=3: {len(n[n >= 3]):,} "
          f"of {queries.size:,}; ordered sibling pairs to score {len(pairs):,}; both-true share {lab.mean():.3f}")


def evaluate():
    queries, s = base_sets()
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas()
    T = gt[gt.s1_idx.isin(queries)].groupby("s1_idx").size().reindex(queries, fill_value=0).to_numpy().astype(float)
    meta = pd.read_parquet("work/retrieval/tc_train_meta.parquet")
    sc = pq.read_table("work/ce/ce_rec_c0/TC_TRAIN.parquet").to_pandas()
    if not np.array_equal(sc.pool_idx.to_numpy().astype(np.int64), meta.pool_idx_b.to_numpy()):
        raise RuntimeError("TC_TRAIN scores do not match meta order")
    meta["p"] = sc.p_ce.to_numpy()
    rev = meta.rename(columns={"pool_idx_a": "pool_idx_b", "pool_idx_b": "pool_idx_a", "p": "p_rev"})
    both = meta.merge(rev, on=["s1_idx", "pool_idx_a", "pool_idx_b"])
    both["p2"] = (both.p + both.p_rev) / 2
    sib = both.groupby(["s1_idx", "pool_idx_a"]).p2.agg(["mean", "min"]).reset_index() \
        .rename(columns={"pool_idx_a": "pool_idx", "mean": "sib_mean", "min": "sib_min"})
    s = s.merge(sib, on=["s1_idx", "pool_idx"], how="left")
    from sklearn.metrics import roc_auc_score
    has = s.sib_mean.notna()
    auc = float(roc_auc_score(s.y[has], s.sib_mean[has]))
    fp_share = float((~s.y[has]).mean())
    qpos = pd.Series(np.arange(queries.size), index=queries)

    def f(sel_keep):
        k = s[sel_keep]
        P = k.groupby("s1_idx").size().reindex(queries, fill_value=0).to_numpy().astype(float)
        TP = k[k.y].groupby("s1_idx").size().reindex(queries, fill_value=0).to_numpy().astype(float)
        return f05(T, P, TP)
    base = f(np.ones(len(s), bool))
    half = ((queries.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2 ** 32) >> np.uint64(31)).astype(int)
    grid = [(tau, cap) for tau in (0.05, 0.1, 0.2, 0.3, 0.5) for cap in (1.01, 0.99, 0.95, 0.9)]
    curve = {}
    for tau, cap in grid:
        drop = has & (s.sib_mean < tau) & (s.score < cap)
        fr = f(~drop.to_numpy())
        curve[f"{tau}|{cap}"] = {"delta": float((fr - base).mean()), "dropped": int(drop.sum()),
                                 "dropped_fp": int((drop & ~s.y).sum()), "dropped_tp": int((drop & s.y).sum()),
                                 "delta_h": [float((fr - base)[half == h].mean()) for h in (0, 1)], "_f": fr}
    f_cf, chosen = base.copy(), {}
    for h in (0, 1):
        bk = max(curve, key=lambda k: curve[k]["delta_h"][h])
        chosen[f"chosen_on_half{h}"] = bk
        f_cf[half == 1 - h] = curve[bk]["_f"][half == 1 - h]
    cf = match.paired_delta(f_cf, base, n_boot=2000)
    s1c = pq.read_table("work/records/train_s1.parquet", columns=["s1_idx", "country_raw"]).to_pandas() \
        .set_index("s1_idx").country_raw.fillna("NA").astype(str).str.strip().reindex(queries).to_numpy()
    segs = {"US": s1c == "US", "India": s1c == "India", "singleton": T == 0, "band": exp.band_mask(queries)}
    seg = {k: match.paired_delta(f_cf[m], base[m], n_boot=1000) for k, m in segs.items()}
    for v in curve.values():
        v.pop("_f")
    out = {"base_f05": float(base.mean()), "records_with_siblings": int(has.sum()), "fp_share_among_them": fp_share,
           "auc_sib_mean_true_vs_fp": auc, "crossfit": {"f05": float(f_cf.mean()), **cf, **chosen},
           "crossfit_segments": seg, "curve": curve}
    io_utils.write_metrics("tc_train", out, queries.size, int(T.sum()))
    print(f"base (M027 + DRACE@0.995) F0.5 {base.mean():.5f}; records with >=2 siblings {has.sum():,} "
          f"(FP share {fp_share:.4f}); sibling-score AUC true-vs-FP {auc:.4f}")
    for k, v in curve.items():
        print(f"  tau|cap {k:>10}: dropped {v['dropped']:>6,} (FP {v['dropped_fp']:>5,} TP {v['dropped_tp']:>5,}) "
              f"delta {v['delta']:+.5f}")
    print(f"CROSS-FIT delta {cf['delta']:+.5f} [{cf['ci95'][0]:+.5f}, {cf['ci95'][1]:+.5f}] {chosen}")
    for k, v in seg.items():
        print(f"  {k:9s} {v['delta']:+.5f} [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]")


if __name__ == "__main__":
    {"prep": prep, "eval": evaluate}[sys.argv[1]]()
