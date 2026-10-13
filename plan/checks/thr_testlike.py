"""TH2: threshold chosen on train under test's candidate-list-length distribution (covariate reweighting; test INPUTS only,
no test labels, no leaderboard feedback). Train S1 (MATCH_TRAIN, M027 OOF + S14 DRACE additions) are weighted by
w(bucket) = share of test S1 in that list-length bucket / share of train S1 in it; the weighted mean F0.5 is swept over
thresholds. Cross-check: unweighted (TH1). Writes artifacts/metrics/thr_testlike.json.
"""
import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from gate2_eval import f05

B = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 10 ** 6]


def main():
    q = np.load(match.runs_dir("M027") / "queries.npy")
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas(); gt = gt[gt.s1_idx.isin(q)]
    Tn = gt.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    o = pq.read_table(match.runs_dir("M027") / "oof.parquet").to_pandas()
    ln_tr = o.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy()
    tc = pd.concat([pq.read_table(p, columns=["s1_idx"]).to_pandas() for p in sorted(glob.glob("work/candidates/TEST/part-*.parquet"))])
    n_test = pq.ParquetFile("work/records/test_s1.parquet").metadata.num_rows
    ln_te = np.concatenate([tc.groupby("s1_idx").size().to_numpy(), np.zeros(n_test - tc.s1_idx.nunique(), int)])
    btr, bte = np.digitize(ln_tr, B) , np.digitize(ln_te, B)
    sh_tr = np.bincount(btr, minlength=len(B) + 1) / len(btr); sh_te = np.bincount(bte, minlength=len(B) + 1) / len(bte)
    w_b = np.where(sh_tr > 0, sh_te / np.maximum(sh_tr, 1e-12), 0)
    w = w_b[btr]
    print("list length mean train %.2f test %.2f" % (ln_tr.mean(), ln_te.mean()))
    print("bucket weights", {f"{B[i-1]}-{B[i]-1}" if i < len(B) else f">={B[-1]}": round(float(w_b[i]), 2) for i in range(1, len(B)) if sh_tr[i] > 0})
    cols = ["s1_idx", "pool_idx", "p", "label"]
    g = pd.read_parquet("work/retrieval/gate2_meta.parquet"); g["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
    d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet"); d["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2DS_TRAIN.parquet").column("p_ce").to_numpy()
    dr = pd.concat([g[cols], d[cols]]).drop_duplicates(["s1_idx", "pool_idx"])
    dr = dr[dr.p >= 0.995].sort_values("p", ascending=False)
    out = {"weights": w_b.tolist(), "rows": []}
    fs = {}
    for t in np.round(np.arange(0.66, 0.901, 0.02), 2):
        k = match.apply_decision({"rule": "threshold", "threshold": float(t), "one_owner": True},
                                 o.s1_idx.to_numpy(), o.pool_idx.to_numpy(), o.p_cal.to_numpy())
        acc = o[k]
        add = dr[~dr.pool_idx.isin(set(acc.pool_idx.tolist()))].drop_duplicates("pool_idx")
        P = (acc.groupby("s1_idx").size().reindex(q, fill_value=0) + add.groupby("s1_idx").size().reindex(q, fill_value=0)).to_numpy().astype(float)
        TP = (acc[acc.y].groupby("s1_idx").size().reindex(q, fill_value=0)
              + add[add.label == 1].groupby("s1_idx").size().reindex(q, fill_value=0)).to_numpy().astype(float)
        fs[t] = f05(Tn, P, TP)
    ref = fs[0.72] if 0.72 in fs else None
    for t, f in fs.items():
        uw, ww = f.mean(), np.average(f, weights=w)
        out["rows"].append({"t": float(t), "f05": float(uw), "f05_testlike": float(ww)})
        print(f"t {t:.2f}: unweighted {uw:.5f}  test-like weighted {ww:.5f}", flush=True)
    best = max(out["rows"], key=lambda r: r["f05_testlike"])
    out["best_testlike"] = best
    print("best test-like threshold (M027 scale, chosen 0.73 = live 0.70):", best)
    io_utils.write_metrics("thr_testlike", out, q.size, int(Tn.sum()))


if __name__ == "__main__":
    main()
