"""HF Job, experiment I1 (iterative retrieval), MATCH_TRAIN only. Embeds the train pool with the distilled e5 (repo
e5_distil/), then per country, exact top-K for:
  profile  normalize(e5(S1 text) + mean e5(accepted pool records of that S1))
  hop2     e5(pool record added by DRACE)   (its own neighbours; self excluded)
Inputs in the dataset repo: retrieval/gate1_pool.parquet (train pool texts), retrieval/gate1_queries.parquet (S1 texts,
kind 's1'), retrieval/iter_train_queries.parquet (plan/checks/iter_prep.py). Output: retrieval/iter_train_topk.parquet.
"""
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModel, AutoTokenizer

REPO = "prathameshk421/ber-ce-scoring"
K = int(os.environ.get("ITER_K", "10"))
t0 = time.time()
POOL_F = os.environ.get("ITER_POOL", "gate1_pool")      # test: drace_test_pool
S1Q_F = os.environ.get("ITER_S1Q", "gate1_queries")     # test: dracd_test_s1_queries
Q_F = os.environ.get("ITER_Q", "iter_train_queries")    # test: iter_test_queries
OUT_F = os.environ.get("ITER_OUT", "iter_train_topk")   # test: iter_test_topk
root = Path(snapshot_download(REPO, repo_type="dataset", local_dir="/tmp/it", allow_patterns=[
    f"retrieval/{POOL_F}.parquet", f"retrieval/{S1Q_F}.parquet", f"retrieval/{Q_F}.parquet", "e5_distil/*"]))
dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(str(root / "e5_distil"))
model = AutoModel.from_pretrained(str(root / "e5_distil"), dtype=torch.bfloat16).to(dev).eval()


@torch.inference_mode()
def embed(texts, tag, batch=1024):
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = torch.empty((len(texts), 384), dtype=torch.float16, device=dev)
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        b = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        out[torch.as_tensor(idx, device=dev)] = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).half()
    print(f"[{tag}] {len(texts):,} embedded ({time.time() - t0:.0f}s)", flush=True)
    return out


pool = pq.read_table(root / f"retrieval/{POOL_F}.parquet")
p_idx = pool.column("pool_idx").to_numpy().astype(np.int64)
p_cty = np.asarray(pool.column("country").to_pylist(), dtype=object)
P = embed(pool.column("text").to_pylist(), "pool")
row_of = np.full(p_idx.max() + 1, -1, dtype=np.int64)
row_of[p_idx] = np.arange(p_idx.size)
g1 = pq.read_table(root / f"retrieval/{S1Q_F}.parquet", columns=["qid", "kind", "text"]).to_pandas()
s1_text = dict(zip(g1.qid[g1.kind == "s1"].astype(np.int64), g1.text[g1.kind == "s1"]))
q = pq.read_table(root / f"retrieval/{Q_F}.parquet").to_pandas()
prof = q[q.kind == "profile"].reset_index(drop=True)
hop = q[q.kind == "hop2"].reset_index(drop=True)
S = embed([s1_text[int(s)] for s in prof.s1_idx], "S1")
lens = prof.members.map(len).to_numpy()
flat = np.concatenate([np.asarray(m, dtype=np.int64) for m in prof.members])
owner = torch.as_tensor(np.repeat(np.arange(len(prof)), lens), device=dev)
M = torch.zeros((len(prof), 384), dtype=torch.float32, device=dev)
M.index_add_(0, owner, P[torch.as_tensor(row_of[flat], device=dev)].float())
V = S.float() + M / torch.as_tensor(lens, dtype=torch.float32, device=dev).unsqueeze(1)
V = torch.nn.functional.normalize(V, dim=-1).half()
H = P[torch.as_tensor(row_of[hop.self_pool.to_numpy()], device=dev)]
Qv = torch.cat([V, H])
qid = np.concatenate([prof.qid.to_numpy(), hop.qid.to_numpy()])
qcty = np.concatenate([prof.country.to_numpy(object), hop.country.to_numpy(object)])
qself = np.concatenate([np.full(len(prof), -1), hop.self_pool.to_numpy()])
cols = {"qid": [], "rank": [], "pool_idx": [], "cos": []}
for c in sorted(set(qcty.tolist())):
    qi, pi = np.flatnonzero(qcty == c), np.flatnonzero(p_cty == c)
    Pc = P[torch.as_tensor(pi, device=dev)]
    for a in range(0, qi.size, 1024):
        rows = qi[a:a + 1024]
        v, j = torch.topk(Qv[torch.as_tensor(rows, device=dev)] @ Pc.T, K + 1, dim=1)
        v, j = v.float().cpu().numpy(), p_idx[pi[j.cpu().numpy()]]
        for r, qq in enumerate(rows):
            keep = j[r] != qself[qq]
            cols["qid"].append(np.full(keep.sum(), qid[qq])); cols["rank"].append(np.arange(1, keep.sum() + 1))
            cols["pool_idx"].append(j[r][keep]); cols["cos"].append(v[r][keep])
    print(f"country {c}: {qi.size:,} queries x {pi.size:,} pool ({time.time() - t0:.0f}s)", flush=True)
out = pa.table({"qid": pa.array(np.concatenate(cols["qid"]).astype(np.int64)),
                "rank": pa.array(np.concatenate(cols["rank"]).astype(np.int16)),
                "pool_idx": pa.array(np.concatenate(cols["pool_idx"]).astype(np.uint32)),
                "cos": pa.array(np.concatenate(cols["cos"]).astype(np.float16))})
pq.write_table(out, f"/tmp/{OUT_F}.parquet", compression="zstd")
HfApi().upload_file(path_or_fileobj=f"/tmp/{OUT_F}.parquet", path_in_repo=f"retrieval/{OUT_F}.parquet",
                    repo_id=REPO, repo_type="dataset", commit_message="iter I1 topk")
print(f"done {out.num_rows:,} rows ({time.time() - t0:.0f}s)", flush=True)
