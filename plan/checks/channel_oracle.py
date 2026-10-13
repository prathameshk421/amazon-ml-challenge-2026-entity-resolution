"""N12: oracle upside of each retrieval channel still unused after S13 + seed ranks 3-5, MATCH_TRAIN (labels for scoring
only). Channels from cached top-50 files: untrained e5 (gate1_topk) S1 / seed ranks, distilled e5 (gate1_distil_topk) S1 /
seed ranks. For each: new pairs per S1 (not in lists or any used channel, pool not owned), true rate, oracle F0.5 gain
(accept only the true new pairs). Realised/oracle calibration: seed ranks 3-5 oracle vs measured +0.00028 (seed5_eval).
Writes artifacts/metrics/channel_oracle.json.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, match
from gate2_eval import f05, key

SB = 10 ** 9


def main():
    q = np.load(match.runs_dir("M027") / "queries.npy")
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas(); gt = gt[gt.s1_idx.isin(q)]
    gk = key(gt.s1_idx, gt.pool_idx)
    Tn = gt.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    st = pd.read_parquet("work/retrieval/iter_train_state.parquet")  # S13-equivalent accepted set
    P = st.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
    TP = pd.Series(np.isin(key(st.s1_idx, st.pool_idx), gk).astype(float)).groupby(st.s1_idx.to_numpy()).sum() \
        .reindex(q, fill_value=0).to_numpy()
    base = f05(Tn, P, TP).mean()
    owned = np.unique(st.pool_idx.to_numpy().astype(np.int64))
    cand = pq.read_table("work/candidates/MATCH_TRAIN/part-00000.parquet", columns=["s1_idx", "pool_idx"]).to_pandas()
    g = pd.read_parquet("work/retrieval/gate2_meta.parquet")
    d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet")
    used = [key(cand.s1_idx, cand.pool_idx), key(g.s1_idx, g.pool_idx), key(d.s1_idx, d.pool_idx)]
    known13 = np.unique(np.concatenate([used[0], key(g[(g.s1_rank <= 5) | (g.seed_rank <= 2)].s1_idx,
                                                          g[(g.s1_rank <= 5) | (g.seed_rank <= 2)].pool_idx), used[2]]))
    known = np.unique(np.concatenate(used))  # after seed ranks 3-5 (all of gate2_meta is seed<=5 | s1<=5)
    seeds = pd.read_parquet("work/retrieval/gate1_seeds.parquet")
    out = {"base_f05": float(base), "rows": []}

    def load(f):
        t = pq.read_table(f"work/retrieval/{f}").to_pandas()
        s = t[t.qid < SB].rename(columns={"qid": "s1_idx"})
        e = t[t.qid >= SB].merge(seeds[["qid", "s1_idx"]], on="qid")
        return s, e
    chans = {}
    us, ue = load("gate1_topk.parquet")
    ds, de = load("gate1_distil_topk.parquet")
    for name, df in (("untrained_seed", ue), ("distil_seed", de), ("distil_s1", ds), ("untrained_s1", us)):
        chans[name] = df

    def orc(df, lo, hi, kn):
        s = df[(df["rank"] >= lo) & (df["rank"] <= hi)]
        kk = np.unique(key(s.s1_idx, s.pool_idx))
        new = kk[~np.isin(kk, kn)]
        new = new[~np.isin(new & 0xFFFFFFFF, owned)]
        it = np.isin(new, gk)
        ad = pd.Series(it.astype(float)).groupby(new >> 32).sum().reindex(q, fill_value=0).to_numpy()
        return new.size, it.sum(), f05(Tn, P + ad, TP + ad).mean() - base
    n, t, o = orc(ue, 3, 5, known13)
    out["calib_seed3_5"] = {"new_per_s1": n / q.size, "true": int(t), "oracle": float(o), "realised": 0.00028}
    print(f"calibration: untrained seed 3-5 vs S13: new {n / q.size:.2f}/S1 true {t:,} oracle {o:+.5f} (realised +0.00028)")
    for name, lo, hi in (("untrained_seed", 6, 10), ("untrained_seed", 6, 20), ("untrained_seed", 6, 50),
                         ("distil_seed", 1, 2), ("distil_seed", 1, 5), ("distil_seed", 1, 10),
                         ("distil_s1", 6, 7), ("distil_s1", 6, 10), ("untrained_s1", 6, 10), ("untrained_s1", 6, 20)):
        n, t, o = orc(chans[name], lo, hi, known)
        out["rows"].append({"channel": name, "lo": lo, "hi": hi, "new_per_s1": n / q.size, "true": int(t), "oracle": float(o)})
        print(f"{name} {lo}-{hi}: new {n / q.size:.2f}/S1 ({n:,}) true {t:,} ({t / max(n, 1):.3%}) oracle {o:+.5f}", flush=True)
    io_utils.write_metrics("channel_oracle", out, q.size, int(Tn.sum()))


if __name__ == "__main__":
    main()
