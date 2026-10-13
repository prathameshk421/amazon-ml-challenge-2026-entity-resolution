"""Fine-tune a cross-encoder on (S1 record, pool record) pairs of DEV_REST_180K.

  python ce/train.py --pairs <dir with pairs_DEV_REST_180K.parquet> --model <checkpoint dir or hub id> \
      --out <dir> [--mode full|lora] [--epochs 1] [--batch 64] [--lr 3e-5] [--max-steps N] [--limit N]

Writes <out>/model/ (weights fp16 safetensors + tokenizer; LoRA mode: merged weights) and <out>/train_meta.json
(settings, library versions, steps, held-out logloss/AUC per eval, train pairs/s).

Held-out slice: 5% of DEV_REST_180K S1 by a fixed hash of s1_idx (grouped by S1, never MATCH_VALID).
Mixed precision (--precision auto): bf16 autocast on CUDA GPUs with native bf16 (A10G/L4/L40S), else fp16 autocast +
GradScaler (T4); fp32 off CUDA.
"""
import argparse
import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from sklearn.metrics import log_loss, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import common


def heldout_mask(s1_idx, pct=5):
    return (s1_idx.astype(np.uint64) * np.uint64(2654435761) % np.uint64(100)) < np.uint64(pct)


def bucketed_batches(lens, batch, rng, chunk_mult=50):
    """Shuffled batches of similar-length rows: sort a random permutation by length inside chunks."""
    perm = rng.permutation(lens.size)
    chunk = batch * chunk_mult
    batches = []
    for s in range(0, perm.size, chunk):
        c = perm[s:s + chunk]
        c = c[np.argsort(lens[c], kind="stable")]
        batches += [c[i:i + batch] for i in range(0, c.size, batch)]
    rng.shuffle(batches)
    return batches


def evaluate(model, ids, y, pad_id, dev):
    p = common.score_ids(model, ids, pad_id, dev, log_every=10**12)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {"logloss": float(log_loss(y, pc, labels=[0, 1])), "auc": float(roc_auc_score(y, p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["full", "lora"], default="full")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (probe/smoke)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N pairs (smoke)")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = common.device()
    out = Path(args.out)
    (out / "model").mkdir(parents=True, exist_ok=True)

    df = common.read_pairs(Path(args.pairs) / "pairs_DEV_REST_180K.parquet",
                           ["s1_idx", "text_a", "text_b", "label"], args.limit or None)
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    enc = tok(list(df.text_a), list(df.text_b), truncation=True, max_length=args.max_length)["input_ids"]
    print(f"tokenized {len(enc):,} pairs in {time.time() - t0:.0f}s", flush=True)
    y = df.label.to_numpy().astype(np.float32)
    ho = heldout_mask(df.s1_idx.to_numpy())
    tr_idx, ho_idx = np.flatnonzero(~ho), np.flatnonzero(ho)
    if args.limit:  # smoke runs: cap the held-out scoring cost
        ho_idx = ho_idx[:5000]
    ho_ids, ho_y = [enc[i] for i in ho_idx], y[ho_idx]
    lens = np.array([len(enc[i]) for i in tr_idx], dtype=np.int32)
    print(f"train {tr_idx.size:,} pairs (pos {y[tr_idx].mean():.3f}), held-out {ho_idx.size:,}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model  # optional dependency, LoRA mode only
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, task_type="SEQ_CLS",
                                                 target_modules=["query", "key", "value", "dense"]))
    model.to(dev)
    prec = args.precision
    if prec == "auto":
        prec = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if dev.type == "cuda" else "fp32"
    use_amp = dev.type == "cuda" and prec != "fp32"
    amp_dtype = torch.bfloat16 if prec == "bf16" else torch.float16
    steps_per_epoch = int(np.ceil(tr_idx.size / args.batch))
    total = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total = min(total, args.max_steps)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and prec == "fp16")  # bf16 needs no loss scaling
    lossf = torch.nn.BCEWithLogitsLoss()

    before = evaluate(model, ho_ids, ho_y, tok.pad_token_id, dev)
    print("held-out before training", before, flush=True)
    evals, best, bad, step, seen, t_train = [], None, 0, 0, 0, 0.0
    done = False
    while not done:
        for b in bucketed_batches(lens, args.batch, rng):
            model.train()
            rows = tr_idx[b]
            ids, att = common.collate([enc[i] for i in rows], tok.pad_token_id)
            tgt = torch.from_numpy(y[rows]).to(dev)
            ts = time.time()
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                logit = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits[:, 0]
            loss = lossf(logit.float(), tgt)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_train += time.time() - ts
            step += 1
            seen += len(rows)
            if step % 100 == 0:
                print(f"step {step}/{total} loss {loss.item():.4f} {seen / t_train:.0f} pairs/s", flush=True)
            last = step >= total
            if step % args.eval_every == 0 or last:
                m = evaluate(model, ho_ids, ho_y, tok.pad_token_id, dev)
                m["step"] = step
                evals.append(m)
                print("held-out", m, flush=True)
                if best is None or m["logloss"] < best["logloss"]:
                    best, bad = m, 0
                    save(model, tok, out / "model", args.mode)
                else:
                    bad += 1
                    if bad >= args.patience:
                        print("early stop", flush=True)
                        last = True
            if last:
                done = True
                break

    meta = {"model": args.model, "mode": args.mode, "train_set": "DEV_REST_180K", "heldout": "5% S1 hash",
            "input_format": "text_a='{s1 name} | {s1 addr}', text_b='{pool name} | {pool addr}'",
            "max_length": args.max_length, "precision": prec, "seed": args.seed, "limit": args.limit, "batch": args.batch, "lr": args.lr, "epochs": args.epochs,
            "steps": step, "total_steps_planned": total, "train_pairs": int(tr_idx.size),
            "heldout_pairs": int(ho_idx.size), "train_pairs_per_s": round(seen / max(t_train, 1e-9), 1),
            "train_seconds": round(t_train, 1), "before": before, "evals": evals, "best": best,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None,
            "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else dev.type,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "kaggle_kernel": os.environ.get("KAGGLE_KERNEL_RUN_TYPE")}
    common.write_json(out / "train_meta.json", meta)
    print("best", best, "->", out / "model", flush=True)


def save(model, tok, path, mode):
    """fp16 copy on CPU (half the size); training continues on the original. LoRA: merge the copy, not the
    model being trained (merge_and_unload mutates in place)."""
    c = copy.deepcopy(model).to("cpu")
    if mode == "lora":
        c = c.merge_and_unload()
    c = c.half()
    c.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    del c


if __name__ == "__main__":
    main()
