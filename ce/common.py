"""Shared helpers for ce/train.py, ce/inference.py, ce/evaluate.py. No `ber` import: these run on Kaggle / other
GPUs from the pair parquet files alone."""
import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_pairs(path, columns=None, limit=None):
    t = pq.read_table(path, columns=columns)
    if limit:
        t = t.slice(0, limit)
    return t.to_pandas()


def length_order(tokenizer, text_a, text_b, max_length):
    """Row order by tokenized pair length (short first) and the lengths themselves."""
    enc = tokenizer(list(text_a), list(text_b), truncation=True, max_length=max_length)
    lens = np.fromiter((len(x) for x in enc["input_ids"]), dtype=np.int32, count=len(text_a))
    return np.argsort(lens, kind="stable"), lens, enc["input_ids"]


def collate(ids_list, pad_id):
    n = max(len(x) for x in ids_list)
    ids = torch.full((len(ids_list), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(ids_list), n), dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        att[i, :len(x)] = 1
    return ids, att


@torch.no_grad()
def score_ids(model, ids_all, pad_id, dev, tokens_per_batch=65_536, fp16=True, log_every=500_000):
    """Sigmoid scores for pre-tokenized pairs; batches are built from length-sorted rows with a token budget."""
    model.eval()
    order = np.argsort([len(x) for x in ids_all], kind="stable")
    out = np.empty(len(ids_all), dtype=np.float32)
    i, done, t0, next_log = 0, 0, time.time(), log_every
    while i < len(order):
        j = i
        # rows are sorted, so the batch's padded length is its last row's length
        while j < len(order) and (j - i + 1) * len(ids_all[order[j]]) <= tokens_per_batch:
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        ids, att = collate([ids_all[k] for k in idx], pad_id)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=fp16 and dev.type == "cuda"):
            logit = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits[:, 0]
        out[idx] = torch.sigmoid(logit.float()).cpu().numpy()
        done += len(idx)
        i = j
        if done >= next_log:
            print(f"  scored {done:,}/{len(order):,} ({done / (time.time() - t0):.0f} pairs/s)", flush=True)
            next_log += log_every
    return out


def write_json(path, doc):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, default=str)
