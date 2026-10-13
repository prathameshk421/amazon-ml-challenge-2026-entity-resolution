"""TH1: LightGBM decision threshold under the full S14 pipeline (M027 OOF + DRACE additions S1@5 | seed@5 + distilled S1@5,
c0 >= 0.995, one-owner), MATCH_TRAIN, labels for scoring only. M027's own chosen threshold is 0.73 (full_c0_rgrp: 0.70), so
thresholds are also reported as offsets from it. Writes artifacts/metrics/thr_sweep_s14.json.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from gate2_eval import f05


def main():
    q = np.load(match.runs_dir("M027") / "queries.npy")
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas(); gt = gt[gt.s1_idx.isin(q)]
    Tn = gt.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    o = pq.read_table(match.runs_dir("M027") / "oof.parquet").to_pandas()
    cols = ["s1_idx", "pool_idx", "p", "label"]
    g = pd.read_parquet("work/retrieval/gate2_meta.parquet")
    g["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
    d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet")
    d["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2DS_TRAIN.parquet").column("p_ce").to_numpy()
    dr = pd.concat([g[cols], d[cols]]).drop_duplicates(["s1_idx", "pool_idx"])
    dr = dr[dr.p >= 0.995].sort_values("p", ascending=False)
    s1c = pq.read_table("work/records/train_s1.parquet", columns=["s1_idx", "country_raw"]).to_pandas() \
        .set_index("s1_idx").country_raw.fillna("NA").str.strip().reindex(q).to_numpy()
    fs, out = {}, {"rows": []}
    for t in (0.55, 0.60, 0.63, 0.65, 0.68, 0.70, 0.73, 0.75, 0.78):
        k = match.apply_decision({"rule": "threshold", "threshold": t, "one_owner": True},
                                 o.s1_idx.to_numpy(), o.pool_idx.to_numpy(), o.p_cal.to_numpy())
        acc = o[k]
        add = dr[~dr.pool_idx.isin(set(acc.pool_idx.tolist()))].drop_duplicates("pool_idx")
        P = (acc.groupby("s1_idx").size().reindex(q, fill_value=0) + add.groupby("s1_idx").size().reindex(q, fill_value=0)).to_numpy().astype(float)
        TP = (acc[acc.y].groupby("s1_idx").size().reindex(q, fill_value=0)
              + add[add.label == 1].groupby("s1_idx").size().reindex(q, fill_value=0)).to_numpy().astype(float)
        fs[t] = f05(Tn, P, TP)
    ref = fs[0.73]
    for t, f in fs.items():
        dl = match.paired_delta(f, ref, n_boot=500)
        seg = {c: float(f[s1c == c].mean() - ref[s1c == c].mean()) for c in ("US", "India")}
        seg["singleton"] = float(f[Tn == 0].mean() - ref[Tn == 0].mean())
        out["rows"].append({"t": t, "f05": float(f.mean()), "vs_0.73": dl, "seg": seg})
        print(f"t {t:.2f}: F0.5 {f.mean():.5f}  vs chosen(0.73) {dl['delta']:+.5f} [{dl['ci95'][0]:+.5f}, {dl['ci95'][1]:+.5f}]  "
              + " ".join(f"{k} {v:+.5f}" for k, v in seg.items()), flush=True)
    io_utils.write_metrics("thr_sweep_s14", out, q.size, int(Tn.sum()))


if __name__ == "__main__":
    main()
