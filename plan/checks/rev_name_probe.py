"""N11 probes (MATCH_TRAIN, labels for scoring only): name-only c0 separability on DRACE pairs + reverse-name (pool->S1) coverage of missed name-only true pairs. Writes artifacts/metrics/rev_name_probe.json."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
from ber import match
from gate2_eval import key
from sklearn.metrics import roc_auc_score
q = np.load(match.runs_dir("M027") / "queries.npy")
gt = pq.read_table("work/records/train_gt.parquet").to_pandas()
gtq = gt[gt.s1_idx.isin(q)]
pool = pq.read_table("work/records/train_pool.parquet", columns=["addr_raw"]).column(0).to_pandas()
empty = pool.fillna("").astype(str).str.strip().eq("").to_numpy()
a = pd.read_parquet("work/retrieval/gate2_meta.parquet")
a["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2_TRAIN.parquet").column("p_ce").to_numpy()
d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet")
d["p"] = pq.read_table("work/ce/ce_rec_c0/GATE2DS_TRAIN.parquet").column("p_ce").to_numpy()
m = pd.concat([a, d]).drop_duplicates(["s1_idx","pool_idx"])
e = empty[m.pool_idx]
print("DRACE cand name-only:", e.sum(), "true", m.label[e].sum(), "AUC", roc_auc_score(m.label[e], m.p[e]) if m.label[e].nunique()>1 else None)
for t in (0.5,0.9,0.99): print(t, "name-only true p>=t", (m.label[e]&(m.p[e]>=t)).sum(), "false", ((~m.label[e].astype(bool))&(m.p[e]>=t)).sum())
print("pool name-only share", empty.mean(), "true pairs name-only share", empty[gtq.pool_idx].mean())
# reverse name topk coverage of missed pairs
cand = pq.read_table("work/candidates/MATCH_TRAIN/part-00000.parquet", columns=["s1_idx","pool_idx"]).to_pandas()
g2 = a[(a.s1_rank <= 5) | (a.seed_rank <= 2)]
known = np.unique(np.concatenate([key(cand.s1_idx,cand.pool_idx), key(g2.s1_idx,g2.pool_idx), key(d.s1_idx,d.pool_idx)]))
miss = gtq[~np.isin(key(gtq.s1_idx,gtq.pool_idx), known)]
mo = miss[empty[miss.pool_idx]]
rv = pq.read_table("work/retrieval/rev_train_topk.parquet").to_pandas()
print("rev topk rows", len(rv), rv.columns.tolist(), "queries", rv.qid.nunique())
cov = mo.pool_idx.isin(rv.qid)
print("name-only missed", len(mo), "covered by rev queries", cov.sum())
r = rv.merge(mo.rename(columns={"pool_idx":"qid"})[["qid","s1_idx"]], on="qid")
hit = r[r.pool_idx == r.s1_idx]
print("true S1 rank among rev top-20:", hit["rank"].value_counts().sort_index().head(10).to_dict())
# precision of rev rank-1 on all rev queries: is rank-1 S1 the truth?
g_all = key(gt.s1_idx, gt.pool_idx)
r1 = rv[rv["rank"]==1]
t1 = np.isin(key(r1.pool_idx, r1.qid), g_all)
print("rev rank-1 true rate (all rev queries):", t1.mean(), "n", len(r1))
print("pool records with any true S1 among rev queries:", np.isin(rv.qid.unique(), gt.pool_idx).mean())
from ber import io_utils
io_utils.write_metrics("rev_name_probe", {"drace_name_only_pairs": int(e.sum()), "drace_name_only_true": int(m.label[e].sum()),
    "name_only_true_p_ge_0.9": int((m.label[e] & (m.p[e] >= 0.9)).sum()), "name_only_missed": int(len(mo)),
    "covered_by_rev": int(cov.sum()), "true_s1_rank1": int((hit["rank"] == 1).sum()), "rev_rank1_true_rate": float(t1.mean())}, q.size, int(len(gtq)))
