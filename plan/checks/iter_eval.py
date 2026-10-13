"""I1 evaluation: do profile queries / hop-2 recursion reach true pairs that NO current channel retrieves?
MATCH_TRAIN, labels for scoring only.

  PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/iter_eval.py

Current channels (S13 on train): stage-2 lists, untrained-e5 S1@5, seed@2, distilled-e5 S1@5. A pair is NEW only if none
of them retrieves it. Reports per method and depth: new pairs per S1, their true rate, true pairs recovered among the
"missed by every channel" set (split name-only pool address / India), and the oracle F0.5 of adding only the recovered
true pairs to the S13-equivalent accepted set. Writes artifacts/metrics/iter_eval.json.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from gate2_eval import f05, key

SB = 10 ** 9


def main():
    q = np.load(match.runs_dir("M027") / "queries.npy")
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas()
    gt = gt[gt.s1_idx.isin(q)]
    T = gt.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    state = pd.read_parquet("work/retrieval/iter_train_state.parquet")
    gk = key(gt.s1_idx, gt.pool_idx)
    acc_k = key(state.s1_idx, state.pool_idx)
    P = state.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    TP = pd.Series(np.isin(acc_k, gk).astype(float)).groupby(state.s1_idx.to_numpy()).sum() \
        .reindex(q, fill_value=0).to_numpy()
    base = f05(T, P, TP)
    # everything current channels retrieve
    cand = pq.read_table("work/candidates/MATCH_TRAIN/part-00000.parquet", columns=["s1_idx", "pool_idx"]).to_pandas()
    g2 = pd.read_parquet("work/retrieval/gate2_meta.parquet")
    g2 = g2[(g2.s1_rank <= 5) | (g2.seed_rank <= 2)]
    ds = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet")
    known = np.unique(np.concatenate([key(cand.s1_idx, cand.pool_idx), key(g2.s1_idx, g2.pool_idx),
                                      key(ds.s1_idx, ds.pool_idx)]))
    missed = gt[~np.isin(gk, known)]
    mk = key(missed.s1_idx, missed.pool_idx)
    pool = pq.read_table("work/records/train_pool.parquet", columns=["addr_raw"]).column(0).to_pandas()
    empty = pool.fillna("").astype(str).str.strip().eq("").to_numpy()
    s1c = pq.read_table("work/records/train_s1.parquet", columns=["country_raw"]).column(0).to_pandas() \
        .fillna("NA").astype(str).str.strip().to_numpy()
    print(f"S13-equivalent train F0.5 {base.mean():.5f}; true pairs missed by every current channel: {len(missed):,} "
          f"(name-only {empty[missed.pool_idx].mean():.0%}, India {(s1c[missed.s1_idx] == 'India').mean():.0%})")
    top = pq.read_table("work/retrieval/iter_train_topk.parquet").to_pandas()
    qs = pq.read_table("work/retrieval/iter_train_queries.parquet", columns=["qid", "kind", "s1_idx"]).to_pandas()
    top = top.merge(qs, on="qid")
    owned = set(state.pool_idx.tolist())
    out = {"base_f05": float(base.mean()), "missed_by_all": int(len(missed)), "rows": []}
    for kind in ("profile", "hop2", "both"):
        for kmax in (1, 2, 5, 10):
            t = top if kind == "both" else top[top.kind == kind]
            t = t[t["rank"] <= kmax]
            kk = np.unique(key(t.s1_idx, t.pool_idx))
            new = kk[~np.isin(kk, known)]
            new = new[~np.isin(new & 0xFFFFFFFF, np.fromiter(owned, dtype=np.int64))]
            is_true = np.isin(new, gk)
            rec = np.isin(mk, new)
            s_new = new >> 32
            ad = pd.Series(is_true.astype(float)).groupby(s_new).sum().reindex(q, fill_value=0).to_numpy()
            oracle = f05(T, P + ad, TP + ad)
            row = {"kind": kind, "k": kmax, "new_per_s1": float(new.size / q.size), "true_rate": float(is_true.mean()) if new.size else 0.0,
                   "recovered": int(rec.sum()), "recovered_share": float(rec.mean()),
                   "recovered_name_only": int((rec & empty[missed.pool_idx.to_numpy()]).sum()),
                   "recovered_india": int((rec & (s1c[missed.s1_idx.to_numpy()] == "India")).sum()),
                   "oracle_gain": float(oracle.mean() - base.mean())}
            out["rows"].append(row)
            print(f"  {kind:7s} top-{kmax:<2d}: new {row['new_per_s1']:.2f}/S1 (true {row['true_rate']:.3%}); recovered "
                  f"{row['recovered']:,} of {len(missed):,} ({row['recovered_share']:.1%}; name-only {row['recovered_name_only']:,}, "
                  f"India {row['recovered_india']:,}); oracle {row['oracle_gain']:+.5f}", flush=True)
    io_utils.write_metrics("iter_eval", out, q.size, int(T.sum()))


if __name__ == "__main__":
    main()
