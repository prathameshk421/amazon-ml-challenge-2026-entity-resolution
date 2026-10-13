"""N11 probe: oracle recall of deeper distilled-e5 S1 retrieval (ranks <= 10/20/50, optional min S13 acceptances). Writes artifacts/metrics/deep_probe.json."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
from ber import match
from gate2_eval import key, f05
q = np.load(match.runs_dir("M027") / "queries.npy")
gt = pq.read_table("work/records/train_gt.parquet").to_pandas(); gtq = gt[gt.s1_idx.isin(q)]
Tn = gtq.groupby("s1_idx").size().reindex(q, fill_value=0).to_numpy().astype(float)
pool = pq.read_table("work/records/train_pool.parquet", columns=["addr_raw"]).column(0).to_pandas()
empty = pool.fillna("").astype(str).str.strip().eq("").to_numpy()
t = pq.read_table("work/retrieval/gate1_distil_topk.parquet").to_pandas()
qs = pq.read_table("work/retrieval/gate1_queries.parquet", columns=["qid","kind"]).to_pandas()
print(qs.kind.value_counts().to_dict(), "max rank", t["rank"].max())
t = t.merge(qs, on="qid"); t = t[t.kind=="s1"]; t["s1_idx"]=t.qid
t = t[t.s1_idx.isin(q)]
a = pd.read_parquet("work/retrieval/gate2_meta.parquet"); a = a[(a.s1_rank<=5)|(a.seed_rank<=2)]
d = pd.read_parquet("work/retrieval/gate2ds_train_meta.parquet")
cand = pq.read_table("work/candidates/MATCH_TRAIN/part-00000.parquet", columns=["s1_idx","pool_idx"]).to_pandas()
known = np.unique(np.concatenate([key(cand.s1_idx,cand.pool_idx), key(a.s1_idx,a.pool_idx), key(d.s1_idx,d.pool_idx)]))
gk = key(gtq.s1_idx, gtq.pool_idx)
miss = gtq[~np.isin(gk, known)]; mk = key(miss.s1_idx, miss.pool_idx)
st = pd.read_parquet("work/retrieval/iter_train_state.parquet")
owned = np.unique(st.pool_idx.to_numpy())
nacc = st.groupby("s1_idx").size().reindex(q, fill_value=0)
P = nacc.to_numpy().astype(float)
TP = pd.Series(np.isin(key(st.s1_idx,st.pool_idx), gk).astype(float)).groupby(st.s1_idx.to_numpy()).sum().reindex(q, fill_value=0).to_numpy()
base = f05(Tn,P,TP).mean()
rows=[]
tk = key(t.s1_idx, t.pool_idx)
for kmax in (10, 20, 50):
  for minacc in (0, 3, 5):
    s = t[(t["rank"]<=kmax) & (t.s1_idx.map(nacc).to_numpy()>=minacc)]
    kk = np.unique(key(s.s1_idx, s.pool_idx)); new = kk[~np.isin(kk, known)]
    new = new[~np.isin(new & 0xFFFFFFFF, owned)]
    it = np.isin(new, gk); rec = np.isin(mk, new)
    ad = pd.Series(it.astype(float)).groupby(new>>32).sum().reindex(q, fill_value=0).to_numpy()
    orc = f05(Tn, P+ad, TP+ad).mean()-base
    ne = empty[(new & 0xFFFFFFFF)]
    rows.append({'k':kmax,'minacc':minacc,'new_per_s1':new.size/q.size,'true_rate':float(it.mean()),'recovered':int(rec.sum()),'oracle_gain':float(orc)})
    print(f"distil S1@{kmax} minacc {minacc}: new {new.size/q.size:.2f}/S1 true {it.mean():.2%}; recovered {rec.sum():,}/{len(miss):,} (addr-bearing true {int((it & ~ne).sum()):,}); oracle {orc:+.5f}")

from ber import io_utils
io_utils.write_metrics('deep_probe', {'base_f05': float(base), 'missed_by_all': int(len(miss)), 'rows': rows}, q.size, int(len(gtq)))
