"""K0 throughput probe: measured train and scoring rates on this GPU, and the hours they imply.

  python ce/probe.py --pairs <dir> --model <checkpoint> --out <dir> [--steps 200] [--batch 64]

Trains `--steps` full fine-tune steps on DEV_REST_180K pairs (weights discarded), scores the first 20,000
MATCH_VALID pairs with the untrained model (pairs/s + untrained ROC-AUC), and extrapolates to the row counts
in ce_pairs_manifest (DEV_REST_180K 976,902 train; MATCH_TRAIN+VALID 1,084,192; TEST 10,613,056;
DEV_REST_REST 6,918,889). Writes <out>/probe.json.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import common

ROWS = {"train_1_epoch": 976_902 * 0.95, "MATCH_TRAIN+VALID": 1_084_192, "TEST": 10_613_056,
        "DEV_REST_REST": 6_918_889}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--n-gpus-scoring", type=int, default=max(torch.cuda.device_count(), 1))
    args = ap.parse_args()
    dev = common.device()
    tok = AutoTokenizer.from_pretrained(args.model)
    use_amp = dev.type == "cuda"

    # scoring rate + untrained AUC (fp16 on CUDA, as in inference.py)
    va = common.read_pairs(Path(args.pairs) / "pairs_MATCH_VALID.parquet", ["text_a", "text_b", "label"], 20_000)
    ids = tok(list(va.text_a), list(va.text_b), truncation=True, max_length=args.max_length)["input_ids"]
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
    model = (model.half() if use_amp else model).to(dev)
    common.score_ids(model, ids[:2000], tok.pad_token_id, dev, log_every=10**12)  # warm-up
    t0 = time.time()
    p = common.score_ids(model, ids, tok.pad_token_id, dev, log_every=10**12)
    if use_amp:
        torch.cuda.synchronize()
    score_rate = len(ids) / (time.time() - t0)
    auc_untrained = float(roc_auc_score(va.label, p))
    del model

    # training rate (fp32 master weights, fp16 autocast)
    tr = common.read_pairs(Path(args.pairs) / "pairs_DEV_REST_180K.parquet", ["text_a", "text_b", "label"],
                           args.steps * args.batch)
    tids = tok(list(tr.text_a), list(tr.text_b), truncation=True, max_length=args.max_length)["input_ids"]
    y = tr.label.to_numpy().astype(np.float32)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lossf = torch.nn.BCEWithLogitsLoss()
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    model.train()
    t_train, losses = 0.0, []
    for s in range(args.steps):
        rows = np.arange(s * args.batch, (s + 1) * args.batch)
        ids_b, att = common.collate([tids[i] for i in rows], tok.pad_token_id)
        ts = time.time()
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            logit = model(input_ids=ids_b.to(dev), attention_mask=att.to(dev)).logits[:, 0]
        loss = lossf(logit.float(), torch.from_numpy(y[rows]).to(dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if use_amp:
            torch.cuda.synchronize()
        if s >= 10:  # skip warm-up steps
            t_train += time.time() - ts
        losses.append(float(loss))
    train_rate = (args.steps - 10) * args.batch / t_train

    hours = {"train_1_epoch": ROWS["train_1_epoch"] / train_rate / 3600}
    for k in ("MATCH_TRAIN+VALID", "TEST", "DEV_REST_REST"):
        hours[k] = ROWS[k] / (score_rate * args.n_gpus_scoring) / 3600
    doc = {"gpu": torch.cuda.get_device_name(0) if use_amp else dev.type, "n_gpus": torch.cuda.device_count(),
           "model": args.model, "max_length": args.max_length, "batch": args.batch,
           "score_pairs_per_s_one_gpu": round(score_rate, 1), "train_pairs_per_s_one_gpu": round(train_rate, 1),
           "peak_vram_train_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if use_amp else None,
           "untrained_auc_match_valid_20k": round(auc_untrained, 4),
           "loss_first10_mean": float(np.mean(losses[:10])), "loss_last10_mean": float(np.mean(losses[-10:])),
           "hours": {k: round(v, 2) for k, v in hours.items()},
           "hours_without_dev_rest_rest": round(sum(v for k, v in hours.items() if k != "DEV_REST_REST"), 2),
           "note": "scoring hours assume --n-gpus-scoring processes in parallel; untrained score is sorted-length "
                   "batches without tokenization time"}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    common.write_json(Path(args.out) / "probe.json", doc)
    print(doc)


if __name__ == "__main__":
    main()
