"""HF Jobs driver: score cross-encoder p_ce for every (S1, pool) pair of MATCH_VALID, MATCH_TRAIN_S20000 and TEST.

Runs inside an HF Job (GPU). Reads pair files + adapter + ce/common.py from the private dataset repo REPO,
loads BAAI/bge-reranker-v2-m3 + the LoRA adapter once (merged, bf16 on CUDA), and writes per 1M-row chunk
  scores/<SET>/c<j>.parquet  (s1_idx:int32, pool_idx:uint32, p_ce:float32)
uploading each chunk back to REPO as soon as it is written, so a timeout keeps finished chunks.
MATCH_VALID runs first and is a gate: AUC < MIN_AUC stops the job before TEST (input-format mismatch).

  hf jobs run --flavor a100-large --timeout 1h50m --secrets HF_TOKEN <image> bash -c "... python score_job.py"

Local mode (no Hub access; same model code): CE_LOCAL=<work dir> CE_ADAPTER=<adapter dir> CE_NAME=ce_rec_c0
reads <work>/ce/pairs_<SET>.parquet and writes <work>/ce/<CE_NAME>/<SET>.parquet (the file collect.py would write).
"""
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, snapshot_download

REPO = os.environ.get("CE_REPO", "prathameshk421/ber-ce-scoring")
BASE = "BAAI/bge-reranker-v2-m3"
ADAPTER = os.environ.get("CE_ADAPTER", "adapter")  # adapter dir inside REPO (c0: adapter_c0)
OUT = os.environ.get("CE_OUT", "scores")  # score prefix inside REPO (c0: scores_c0)
SETS = os.environ.get("CE_SETS", "MATCH_VALID,MATCH_TRAIN_S20000,TEST").split(",")
CHUNK = 1_000_000
MAX_LEN = 128
MIN_AUC = 0.99
TOKENS_PER_BATCH = int(os.environ.get("CE_TOKENS_PER_BATCH", "131072"))

LOCAL = os.environ.get("CE_LOCAL")  # work dir: local run, nothing is downloaded or uploaded
CE_NAME = os.environ.get("CE_NAME", "ce_rec_c0")
if LOCAL:
    api = None
    root = Path(LOCAL) / "ce"
    adapter_dir = Path(ADAPTER)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ce/common.py
else:
    api = HfApi()
    root = Path(snapshot_download(REPO, repo_type="dataset", local_dir="/tmp/ce_in",
                                  allow_patterns=[f"pairs_{s}.parquet" for s in SETS] + [f"{ADAPTER}/*", "common.py"]))
    adapter_dir = root / ADAPTER
    sys.path.insert(0, str(root))
import common  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

dev = common.device()
print(f"device {dev} {torch.cuda.get_device_name() if dev.type == 'cuda' else ''}", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1, dtype=torch.float32)
model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
model = model.to(torch.bfloat16 if bf16 else torch.float32).to(dev).eval()
print(f"model ready, dtype {next(model.parameters()).dtype}", flush=True)

uploads = []


def upload(local, remote):
    if LOCAL:
        return
    th = threading.Thread(target=lambda: api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                                                         repo_id=REPO, repo_type="dataset",
                                                         commit_message=f"scores {remote}"))
    th.start()
    uploads.append(th)


def auc(y, s):
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    npos = y.sum()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


t_job = time.time()
for set_name in SETS:
    src = root / f"pairs_{set_name}.parquet"
    has_label = "label" in pq.read_schema(src).names  # test-side sets carry no labels
    cols = ["s1_idx", "pool_idx", "text_a", "text_b"] + (["label"] if has_label else [])
    table = pq.read_table(src, columns=cols)
    out_dir = (root / CE_NAME / "_chunks" / set_name) if LOCAL else (Path("/tmp/scores") / set_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, preds = [], []
    t_set = time.time()
    for j, c0 in enumerate(range(0, table.num_rows, CHUNK)):
        t = table.slice(c0, CHUNK)
        t0 = time.time()
        ids = tok(t.column("text_a").to_pylist(), t.column("text_b").to_pylist(), truncation=True,
                  max_length=MAX_LEN)["input_ids"]
        t_tok = time.time() - t0
        p = common.score_ids(model, ids, tok.pad_token_id, dev, tokens_per_batch=TOKENS_PER_BATCH, fp16=not bf16)
        path = out_dir / f"c{j:03d}.parquet"
        pq.write_table(pa.table({"s1_idx": pa.array(t.column("s1_idx").to_numpy().astype(np.int32)),
                                 "pool_idx": pa.array(t.column("pool_idx").to_numpy().astype(np.uint32)),
                                 "p_ce": pa.array(p.astype(np.float32))}), path, compression="zstd")
        upload(path, f"{OUT}/{set_name}/{path.name}")
        dt = time.time() - t0
        done = c0 + t.num_rows
        eta = (table.num_rows - done) / (done / (time.time() - t_set))
        print(f"[{set_name}] chunk {j}: {t.num_rows:,} rows in {dt:.0f}s (tokenize {t_tok:.0f}s), "
              f"{t.num_rows / dt:.0f} pairs/s, non-finite {int((~np.isfinite(p)).sum())}, "
              f"{done:,}/{table.num_rows:,}, set ETA {eta / 60:.1f} min, job elapsed {(time.time() - t_job) / 60:.1f} min",
              flush=True)
        if "label" in cols:
            labels.append(t.column("label").to_numpy()); preds.append(p)
    if LOCAL:  # same assembly as collect.py: chunks in order, row order checked against the pair file
        t = pa.concat_tables([pq.read_table(p) for p in sorted(out_dir.glob("c*.parquet"))])
        for c in ("s1_idx", "pool_idx"):
            if not np.array_equal(t.column(c).to_numpy(), table.column(c).to_numpy()):
                raise RuntimeError(f"{set_name}: {c} order differs from pairs_{set_name}.parquet")
        pq.write_table(t, root / CE_NAME / f"{set_name}.parquet", compression="zstd")
        print(f"wrote {root / CE_NAME / f'{set_name}.parquet'}: {t.num_rows:,} rows", flush=True)
    if labels:
        a = auc(np.concatenate(labels).astype(int), np.concatenate(preds))
        print(f"[{set_name}] AUC {a:.5f}", flush=True)
        if set_name == "MATCH_VALID" and a < MIN_AUC:
            for th in uploads:
                th.join()
            sys.exit(f"MATCH_VALID AUC {a:.5f} < {MIN_AUC}: input format differs from training; stopping before TEST")

for th in uploads:
    th.join()
print(f"all sets done in {(time.time() - t_job) / 60:.1f} min", flush=True)
