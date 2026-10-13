"""TH3: per-country LightGBM threshold under the S14/S16 pipeline, test-like weighting (TH2 list-length weights; test inputs
only). Cross-fitted: each country's threshold is chosen on one S1 half and scored on the other; compared with one global
threshold chosen the same way. MATCH_TRAIN, labels for scoring only. Writes artifacts/metrics/thr_country.json.
"""
import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from gate2_eval import f05

B = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 10 ** 6]
TS = [0.70, 0.72, 0.73, 0.74, 0.76, 0.78, 0.80, 0.82]


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
    cols = ["s1_idx", "pool_idx", "p", "label"]
    g = pd.read_parquet("work/retrieval/gate2_meta.parquet"); g["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
    d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet"); d["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2DS_TRAIN.parquet").column("p_ce").to_numpy()
    dr = pd.concat([g[cols], d[cols]]).drop_duplicates(["s1_idx", "pool_idx"])
    dr = dr[dr.p >= 0.995].sort_values("p", ascending=False)
    cty = pq.read_table("work/records/train_s1.parquet", columns=["s1_idx", "country_raw"]).to_pandas() \
        .set_index("s1_idx").country_raw.fillna("NA").str.strip()
    s1c = cty.reindex(q).to_numpy(); oc = cty.reindex(o.s1_idx).to_numpy()
    half = (q % 2).astype(bool)

    def score(thr_by_c):
        thr = np.array([thr_by_c.get(c, thr_by_c["_"]) for c in oc])
        keep = o.p_cal.to_numpy() >= thr
        k = match.apply_decision({"rule": "threshold", "threshold": 0.0, "one_owner": True},
                                 o.s1_idx.to_numpy()[keep], o.pool_idx.to_numpy()[keep], o.p_cal.to_numpy()[keep])
        acc = o[keep][k]
        add = dr[~dr.pool_idx.isin(set(acc.pool_idx.tolist()))].drop_duplicates("pool_idx")
        P = (acc.groupby("s1_idx").size().reindex(q, fill_value=0) + add.groupby("s1_idx").size().reindex(q, fill_value=0)).to_numpy().astype(float)
        TP = (acc[acc.y].groupby("s1_idx").size().reindex(q, fill_value=0)
              + add[add.label == 1].groupby("s1_idx").size().reindex(q, fill_value=0)).to_numpy().astype(float)
        return f05(Tn, P, TP)
    glob_f = {t: score({"_": t}) for t in TS}
    per = {}
    for c in ("US", "India"):
        for t in TS:
            per[(c, t)] = score({"_": 0.74, c: t})
    wm = lambda f, m: np.average(f[m], weights=w[m])
    out = {"global": {str(t): float(np.average(f, weights=w)) for t, f in glob_f.items()}}
    cf_glob = cf_cty = 0.0
    for h in (True, False):
        fit, ev = half == h, half != h
        tg = max(TS, key=lambda t: wm(glob_f[t], fit))
        tc_ = {c: max(TS, key=lambda t: wm(per[(c, t)], fit & (s1c == c))) for c in ("US", "India")}
        f_c = score({"_": tg, **tc_})
        cf_glob += (glob_f[tg][ev] * w[ev]).sum(); cf_cty += (f_c[ev] * w[ev]).sum()
        print(f"half {h}: global {tg}, per-country {tc_}")
        out[f"half_{h}"] = {"global": tg, "per_country": tc_}
    out["crossfit_delta_testlike"] = float((cf_cty - cf_glob) / w.sum())
    print(f"cross-fitted test-like weighted: per-country vs global {out['crossfit_delta_testlike']:+.5f}")
    for c in ("US", "India"):
        print(c, {t: round(wm(per[(c, t)], s1c == c), 5) for t in TS})
    io_utils.write_metrics("thr_country", out, q.size, int(Tn.sum()))


if __name__ == "__main__":
    main()
