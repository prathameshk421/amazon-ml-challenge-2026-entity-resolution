"""Gate 2: does adding c0-filtered dense-retrieval pairs to M027's OOF predictions raise per-S1 F0.5 on MATCH_TRAIN?

  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/gate2_eval.py

Rule: add a new pair (not in the candidate list) when p_ce(c0) >= t and cos >= c. One-owner kept: a pool record
already accepted for another S1 by M027 is not added, and a pool record wanted by several S1 goes to the highest p_ce.
Honest threshold: S1 split in two halves by hash; (t, c) chosen on one half by mean F0.5, scored on the other; swapped.
Reported: cross-fitted paired delta vs M027 with bootstrap CI, per segment, TP/FP added; plus the in-sample best (t, c)
and the curve. c0 never trained on a MATCH_TRAIN S1. Writes artifacts/metrics/gate2_c0.json.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import exp, io_utils, match


def key(a, b):
    return np.asarray(a, dtype=np.int64) << 32 | np.asarray(b, dtype=np.int64)


def f05(T, P, TP):
    den = 1.25 * TP + 0.25 * (T - TP) + (P - TP)
    return np.where((T == 0) & (P == 0), 1.0, np.where(TP > 0, 1.25 * TP / np.maximum(den, 1e-12), 0.0))


def main():
    queries = np.load(match.runs_dir("M027") / "queries.npy")
    qpos = pd.Series(np.arange(queries.size), index=queries)
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas()
    T = gt[gt.s1_idx.isin(queries)].groupby("s1_idx").size().reindex(queries, fill_value=0).to_numpy().astype(float)
    oof = pq.read_table(match.runs_dir("M027") / "oof.parquet").to_pandas()
    pred = match.apply_decision({"rule": "threshold", "threshold": 0.73, "one_owner": True},
                                oof.s1_idx.to_numpy(), oof.pool_idx.to_numpy(), oof.p_cal.to_numpy())
    g = oof.s1_idx.to_numpy()
    P0 = pd.Series(pred.astype(float)).groupby(g).sum().reindex(queries, fill_value=0).to_numpy()
    TP0 = pd.Series((pred & oof.y.to_numpy()).astype(float)).groupby(g).sum().reindex(queries, fill_value=0).to_numpy()
    base = f05(T, P0, TP0)
    owned = set(oof.pool_idx.to_numpy()[pred].tolist())

    meta = pd.read_parquet("work/retrieval/gate2_meta.parquet")
    sc = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").to_pandas()
    if not (np.array_equal(sc.s1_idx.to_numpy(), meta.s1_idx.to_numpy())
            and np.array_equal(sc.pool_idx.to_numpy(), meta.pool_idx.to_numpy())):
        raise RuntimeError("score rows do not match gate2_meta order")
    meta["p"] = sc.p_ce.to_numpy()
    meta["cos"] = meta[["cos_s1", "cos_seed"]].max(axis=1).astype(float)
    meta = meta[~meta.pool_idx.isin(owned)].reset_index(drop=True)
    y = meta.label.to_numpy().astype(bool)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, meta.p))
    half = ((queries.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2 ** 32) >> np.uint64(31)).astype(int)
    half_of = pd.Series(half, index=queries)

    def apply(t, c, rows=None):
        m = meta if rows is None else meta[rows]
        m = m[(m.p >= t) & (m.cos >= c)]
        m = m.sort_values("p", ascending=False).drop_duplicates("pool_idx")  # one-owner among new pairs
        add = m.groupby("s1_idx").size().reindex(queries, fill_value=0).to_numpy().astype(float)
        add_tp = m[m.label == 1].groupby("s1_idx").size().reindex(queries, fill_value=0).to_numpy().astype(float)
        return f05(T, P0 + add, TP0 + add_tp), int(len(m)), int(m.label.sum())

    grid = [(t, c) for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998) for c in (0.0, 0.85, 0.9)]
    curve = {}
    for t, c in grid:
        f, n, tp = apply(t, c)
        curve[f"{t}|{c}"] = {"f05": float(f.mean()), "delta": float(f.mean() - base.mean()), "added": n, "tp": tp,
                             "fp": n - tp, "delta_h": [float((f - base)[half == h].mean()) for h in (0, 1)]}
    # cross-fit: choose on half h, score on half 1-h
    chosen, f_cf = {}, base.copy()
    for h in (0, 1):
        best = max(grid, key=lambda tc: curve[f"{tc[0]}|{tc[1]}"]["delta_h"][h])
        chosen[f"chosen_on_half{h}"] = {"t": best[0], "c": best[1]}
        f_all, _, _ = apply(*best)
        f_cf[half == 1 - h] = f_all[half == 1 - h]
    cf = match.paired_delta(f_cf, base, n_boot=2000)
    s1c = pq.read_table("work/records/train_s1.parquet", columns=["s1_idx", "country_raw"]).to_pandas() \
        .set_index("s1_idx").country_raw.fillna("NA").astype(str).str.strip().reindex(queries).to_numpy()
    segs = {"US": s1c == "US", "India": s1c == "India", "singleton": T == 0, "band": exp.band_mask(queries)}
    seg = {k: match.paired_delta(f_cf[m], base[m], n_boot=1000) for k, m in segs.items()}
    best_in = max(curve, key=lambda k: curve[k]["delta"])
    out = {"n_s1": int(queries.size), "f05_M027": float(base.mean()), "new_pairs_after_owner": int(len(meta)),
           "new_true": int(y.sum()), "auc_c0_new_pairs": auc, "crossfit": {"f05": float(f_cf.mean()), **cf, **chosen},
           "crossfit_segments": seg, "insample_best": {"rule": best_in, **curve[best_in]}, "curve": curve}
    io_utils.write_metrics("gate2_c0", out, queries.size, int(T.sum()))
    print(f"M027 F0.5 {base.mean():.5f}; new pairs {len(meta):,} ({int(y.sum()):,} true); c0 AUC on them {auc:.4f}")
    for k, v in curve.items():
        print(f"  t|cos {k:>9}: added {v['added']:>7,} (TP {v['tp']:>5,} FP {v['fp']:>6,})  delta {v['delta']:+.5f}")
    print(f"CROSS-FIT (rule picked on the other half): F0.5 {f_cf.mean():.5f}, delta {cf['delta']:+.5f} "
          f"[{cf['ci95'][0]:+.5f}, {cf['ci95'][1]:+.5f}], S1 changed {cf['n_changed']:,}; {chosen}")
    for k, v in seg.items():
        print(f"  {k:9s} delta {v['delta']:+.5f} [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]")
    print("wrote artifacts/metrics/gate2_c0.json")


if __name__ == "__main__":
    main()
