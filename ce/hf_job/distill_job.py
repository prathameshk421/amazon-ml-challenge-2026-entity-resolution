"""HF Job: distil c0 into multilingual-e5-small (triplets from plan/checks/distill_e5.py --dump), save the model to the
dataset repo as e5_distil/, then run retrieve_job.py on the gate-1 MATCH_TRAIN inputs with it (output gate1_distil_topk).

Local mode (no Hub access): D_LOCAL=work reads work/retrieval/e5_distil_triplets.parquet, saves the model to
work/retrieval/e5_distil/ and stops (no upload, no gate-1 retrieval).

Loss per batch (mean pooling, L2 norm, same as retrieve_job.py): InfoNCE over the batch (q vs every pos and neg, t 0.05)
+ LAM * MarginMSE(cos(q,pos) - cos(q,neg), p_c0(pos) - p_c0(neg)). bf16 autocast on CUDA.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModel, AutoTokenizer

REPO = "prathameshk421/ber-ce-scoring"
BASE = "intfloat/multilingual-e5-small"
BATCH, LR, EPOCHS, LAM = int(os.environ.get("D_BATCH", "128")), float(os.environ.get("D_LR", "2e-5")), 1.0, 1.0
LOCAL = os.environ.get("D_LOCAL")
OUT = Path(LOCAL) / "retrieval" / "e5_distil" if LOCAL else Path("/tmp/e5_distil")

t0 = time.time()
if LOCAL:
    tp = Path(LOCAL) / "retrieval" / "e5_distil_triplets.parquet"
else:
    tp = Path(hf_hub_download(REPO, "retrieval/e5_distil_triplets.parquet", repo_type="dataset", local_dir="/tmp/d_in"))
t = pq.read_table(tp).to_pandas()
dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModel.from_pretrained(BASE).to(dev)
print(f"{len(t):,} triplets, {torch.cuda.get_device_name()}", flush=True)


def enc(texts):
    b = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model(**b).last_hidden_state
    h = h.float()
    m = b["attention_mask"].unsqueeze(-1).float()
    return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)


q, p, n = t.q.tolist(), t.p.tolist(), t.n.tolist()
tpos, tneg = t.tp.to_numpy(np.float32), t.tn.to_numpy(np.float32)
steps = int(len(t) * EPOCHS) // BATCH
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.06, anneal_strategy="linear")
order = np.random.default_rng(1).permutation(len(t))
log = []
model.train()
for st in range(steps):
    idx = order[st * BATCH:(st + 1) * BATCH]
    eq, ep, en = enc([q[i] for i in idx]), enc([p[i] for i in idx]), enc([n[i] for i in idx])
    logits = eq @ torch.cat([ep, en]).T / 0.05
    nce = torch.nn.functional.cross_entropy(logits, torch.arange(len(idx), device=dev))
    mse = torch.nn.functional.mse_loss((eq * ep).sum(1) - (eq * en).sum(1), torch.from_numpy(tpos[idx] - tneg[idx]).to(dev))
    loss = nce + LAM * mse
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if st % 200 == 0 or st == steps - 1:
        log.append({"step": st, "nce": round(nce.item(), 4), "mse": round(mse.item(), 4), "s": round(time.time() - t0)})
        print(f"step {st}/{steps} nce {nce.item():.4f} marginmse {mse.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
model.eval()
OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
json.dump({"base": BASE, "triplets": len(t), "batch": BATCH, "lr": LR, "epochs": EPOCHS, "lam": LAM,
           "train_set": "DEV_REST_REST fold 1 S1 (disjoint from MATCH_TRAIN); teacher ce_rec_c0 p_ce",
           "log": log}, open(OUT / "train_meta.json", "w"), indent=1)
if LOCAL:
    print(f"model saved to {OUT} ({time.time() - t0:.0f}s)", flush=True)
    sys.exit(0)
HfApi().upload_folder(folder_path=str(OUT), path_in_repo="e5_distil", repo_id=REPO, repo_type="dataset",
                      commit_message="e5 distilled from c0")
print(f"model saved to {REPO}/e5_distil ({time.time() - t0:.0f}s); running gate-1 retrieval", flush=True)
rj = hf_hub_download(REPO, "retrieve_job.py", repo_type="dataset", local_dir="/tmp/j")
env = dict(os.environ, RETR_TAG="gate1", RETR_MODEL_DIR="e5_distil", RETR_OUT_TAG="gate1_distil", RETR_K="50")
sys.exit(subprocess.call([sys.executable, "-u", rj], env=env))
