"""A2: DRACE c0 admission cut-off under test's list-length distribution (TH2 weights; test inputs only), S16-equivalent
base (M027 at 0.74 on its own scale ~ live 0.73) + S17 channels (seed<=5, distilled S1@5, distilled S1 ranks 6-7).
Cross-fitted by S1 half. MATCH_TRAIN, labels for scoring only. Writes artifacts/metrics/admit_testlike.json.
"""
import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from gate2_eval import f05

B = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 10 ** 6]
TS = [0.98, 0.99, 0.993, 0.995, 0.997, 0.998, 0.999]


def main():
    q = np.load(match.runs_dir("M027") / "queries.npy")
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas(); gt = gt[gt.s1_idx.isin(q)]
    Tn = gt.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    o = pq.read_table(match.runs_dir("M027") / "oof.parquet").to_pandas()
    ln_tr = o.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy()
    tc = pd.concat([pq.read_table(p, columns=["s1_idx"]).to_pandas() for p in sorted(glob.glob("work/candidates/TEST/part-*.parquet"))])
    n_test = pq.ParquetFile("work/records/test_s1.parquet").metadata.num_rows
    ln_te = np.concatenate([tc.groupby("s1_idx").size().to_numpy(), np.zeros(n_test - tc.s1_idx.nunique(), int)])
    btr, bte = np.digitize(ln_tr, B), np.digitize(ln_te, B)
    sh_tr = np.bincount(btr, minlength=len(B) + 1) / len(btr); sh_te = np.bincount(bte, minlength=len(B) + 1) / len(bte)
    w = np.where(sh_tr > 0, sh_te / np.maximum(sh_tr, 1e-12), 0)[btr]
    k = match.apply_decision({"rule": "threshold", "threshold": 0.74, "one_owner": True},
                             o.s1_idx.to_numpy(), o.pool_idx.to_numpy(), o.p_cal.to_numpy())
    acc = o[k]
    P0 = acc.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    TP0 = acc[acc.y].groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    cols = ["s1_idx", "pool_idx", "p", "label"]
    g = pd.read_parquet("work/retrieval/gate2_meta.parquet"); g["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
    d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet"); d["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2DS_TRAIN.parquet").column("p_ce").to_numpy()
    e = pd.read_parquet("work/retrieval/gate2d10_train_meta.parquet"); e["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2D10_TRAIN.parquet").column("p_ce").to_numpy()
    dr = pd.concat([g[cols], d[cols], e[e.s1_rank <= 7][cols]]).drop_duplicates(["s1_idx", "pool_idx"])
    dr = dr[~dr.pool_idx.isin(set(acc.pool_idx.tolist()))].sort_values("p", ascending=False)
    fs = {}
    for t in TS:
        m = dr[dr.p >= t].drop_duplicates("pool_idx")
        ad = m.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        tp = m[m.label == 1].groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
        fs[t] = f05(Tn, P0 + ad, TP0 + tp)
    out = {"rows": [{"t": t, "f05": float(f.mean()), "f05_testlike": float(np.average(f, weights=w))} for t, f in fs.items()]}
    for r in out["rows"]:
        print(f"T {r['t']}: unweighted {r['f05']:.5f}  test-like {r['f05_testlike']:.5f}")
    half = (q % 2).astype(bool); cf = 0.0
    for h in (True, False):
        best = max(TS, key=lambda t: np.average(fs[t][half == h], weights=w[half == h]))
        ev = half != h
        cf += ((fs[best][ev] - fs[0.995][ev]) * w[ev]).sum(); print(f"half {h}: chose {best}")
    out["crossfit_vs_0995_testlike"] = float(cf / w.sum())
    print(f"cross-fitted test-like vs 0.995: {out['crossfit_vs_0995_testlike']:+.5f}")
    io_utils.write_metrics("admit_testlike", out, q.size, int(Tn.sum()))


if __name__ == "__main__":
    main()
