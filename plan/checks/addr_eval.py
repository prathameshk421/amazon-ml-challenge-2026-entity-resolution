"""AD1 gate (MATCH_TRAIN, labels for scoring only): new pairs from address-only retrieval beyond every S14 channel, by rank;
recall of the missed true pairs (by N13 category), true rate, oracle F0.5 over the S13-equivalent accepted set.
Optionally writes the new pairs for c0 scoring (--write K). Writes artifacts/metrics/addr_eval.json.
"""
import argparse
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber import io_utils, match
from gate2_eval import f05, key

T = re.compile(r"[a-z0-9]+")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", type=int, default=0); a = ap.parse_args()
    q = np.load(match.runs_dir("M027") / "queries.npy")
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas(); gt = gt[gt.s1_idx.isin(q)]
    gk = key(gt.s1_idx, gt.pool_idx)
    Tn = gt.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    st = pd.read_parquet("work/retrieval/iter_train_state.parquet")
    P = st.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    TP = pd.Series(np.isin(key(st.s1_idx, st.pool_idx), gk).astype(float)).groupby(st.s1_idx.to_numpy()).sum() \
        .reindex(q, fill_value=0).to_numpy()
    base = f05(Tn, P, TP).mean()
    owned = np.unique(st.pool_idx.to_numpy().astype(np.int64))
    cand = pq.read_table("work/candidates/MATCH_TRAIN/part-00000.parquet", columns=["s1_idx", "pool_idx"]).to_pandas()
    g = pd.read_parquet("work/retrieval/gate2_meta.parquet"); d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet")
    e = pd.read_parquet("work/retrieval/gate2d10_train_meta.parquet")
    known = np.unique(np.concatenate([key(cand.s1_idx, cand.pool_idx), key(g.s1_idx, g.pool_idx),
                                      key(d.s1_idx, d.pool_idx), key(e.s1_idx, e.pool_idx)]))
    miss_k = gk[~np.isin(gk, known)]
    t = pd.read_parquet("work/retrieval/addr_train_topk.parquet")
    out = {"base_f05": float(base), "missed": int(miss_k.size), "rows": []}
    for kmax in (1, 2, 3, 5, 10):
        s = t[t["rank"] <= kmax]
        kk = np.unique(key(s.s1_idx, s.pool_idx)); new = kk[~np.isin(kk, known)]
        new = new[~np.isin(new & 0xFFFFFFFF, owned)]
        it = np.isin(new, gk)
        ad = pd.Series(it.astype(float)).groupby(new >> 32).sum().reindex(q, fill_value=0).to_numpy()
        o = f05(Tn, P + ad, TP + ad).mean() - base
        r = {"k": kmax, "new_per_s1": new.size / q.size, "new": int(new.size), "true": int(it.sum()),
             "recovered_missed": int(np.isin(miss_k, new).sum()), "oracle": float(o)}
        out["rows"].append(r)
        print(f"addr top-{kmax}: new {r['new_per_s1']:.2f}/S1 ({new.size:,}) true {r['true']:,} ({it.mean():.3%}); "
              f"missed recovered {r['recovered_missed']:,}/{miss_k.size:,}; oracle {o:+.5f}", flush=True)
        if kmax == a.write:
            import sys; sys.path.insert(0, "ce")
            from prepare_pairs import record_text
            m = pd.DataFrame({"s1_idx": new >> 32, "pool_idx": new & 0xFFFFFFFF, "label": it.astype(np.int8)})
            m = m.sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
            s1t, pt = record_text("train", "s1"), record_text("train", "pool")
            s1a, pa_ = m.s1_idx.to_numpy().astype(np.int32), m.pool_idx.to_numpy().astype(np.uint32)
            pq.write_table(pa.table({"s1_idx": pa.array(s1a), "pool_idx": pa.array(pa_),
                                     "rank": pa.array(np.zeros(len(m), dtype=np.int16)),
                                     "text_a": pa.array(s1t[s1a].tolist(), pa.string()),
                                     "text_b": pa.array(pt[pa_].tolist(), pa.string()),
                                     "label": pa.array(m.label.to_numpy())}), "work/ce/pairs_ADDR_TRAIN.parquet",
                           compression="zstd")
            m.to_parquet("work/retrieval/addr_train_meta.parquet")
            print(f"wrote ADDR_TRAIN {len(m):,} pairs")
    io_utils.write_metrics("addr_eval", out, q.size, int(Tn.sum()))


if __name__ == "__main__":
    main()
