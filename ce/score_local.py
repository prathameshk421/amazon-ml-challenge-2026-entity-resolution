"""Local (MPS) c0 scoring, same model/text/truncation as ce/hf_job/score_job.py: bge-reranker-v2-m3 + LoRA adapter
(work/ce/ce_rec_c0/adapter), merged, fp16 on MPS, max_length 128. Writes work/ce/ce_rec_c0/<OUT>.parquet in the
pairs file's row order (s1_idx, pool_idx, p_ce), checkpointing every chunk so a stopped run can resume.

  PYTHONPATH=ce .venv/bin/python ce/score_local.py <pairs parquet> <OUT name> [--limit N] [--rows a:b]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import common

ap = argparse.ArgumentParser()
ap.add_argument("pairs"); ap.add_argument("out")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--chunk", type=int, default=50_000)
ap.add_argument("--tpb", type=int, default=16384)
a = ap.parse_args()
dev = common.device()
tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", num_labels=1, dtype=torch.float32)
model = PeftModel.from_pretrained(model, "work/ce/ce_rec_c0/adapter").merge_and_unload()
model = model.half().to(dev).eval()
t = pq.read_table(a.pairs, columns=["s1_idx", "pool_idx", "text_a", "text_b"])
if a.limit:
    t = t.slice(0, a.limit)
ck = Path(f"work/ce/ce_rec_c0/_local_{a.out}"); ck.mkdir(parents=True, exist_ok=True)
t0 = time.time()
n = t.num_rows
for j, c in enumerate(range(0, n, a.chunk)):
    f = ck / f"c{j:04d}.parquet"
    if f.exists():
        continue
    s = t.slice(c, a.chunk)
    ids = tok(s.column("text_a").to_pylist(), s.column("text_b").to_pylist(), truncation=True, max_length=128)["input_ids"]
    with torch.inference_mode():
        p = common.score_ids(model, ids, tok.pad_token_id, dev, tokens_per_batch=a.tpb, fp16=False, log_every=10 ** 9)
    pq.write_table(pa.table({"s1_idx": s.column("s1_idx"), "pool_idx": s.column("pool_idx"),
                             "p_ce": pa.array(p.astype(np.float32))}), f)
    done = min(c + a.chunk, n)
    el = time.time() - t0
    print(f"[{a.out}] {done:,}/{n:,} {el / 60:.1f} min, {done / el:.0f} pairs/s (this run), "
          f"non-finite {int((~np.isfinite(p)).sum())}", flush=True)
parts = sorted(ck.glob("c*.parquet"))
out = pa.concat_tables([pq.read_table(x) for x in parts])
assert out.num_rows == n, (out.num_rows, n)
pq.write_table(out, f"work/ce/ce_rec_c0/{a.out}.parquet")
print(f"wrote work/ce/ce_rec_c0/{a.out}.parquet {n:,} rows")
