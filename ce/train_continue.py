"""Continue training the ce_rec LoRA adapter (bge-reranker-v2-m3) on one half of DEV_REST_REST, under a wall-clock budget.

  python ce/train_continue.py --pairs <dir with pairs_DEV_REST_REST.parquet> --base <v2-m3 dir or hub id> \
      --init-adapter <ce_rec adapter dir> --fold 0 --out <dir> [--budget-hours 9.5] [--lr 5e-5] [--batch 32]

Cross-fit design: fold 0 and fold 1 split DEV_REST_REST S1 in half by a hash of s1_idx (independent of the 5% held-out
hash). The fold-k adapter trains on half k only, so its scores on half 1-k, MATCH_TRAIN, MATCH_VALID and TEST are
out-of-sample. Both start from ce_rec (trained on DEV_REST_180K), so DEV_REST_180K stays in-sample for both.

Held-out: 5% of the fold's S1 (train.heldout_mask), capped at --heldout-cap pairs, grouped by S1.
Schedule: linear warmup over 3% of planned steps, then lr * (1 - max(step/steps, elapsed/budget)), so the decay
reaches 0 whether the epoch or the budget ends first. Non-finite loss (fp16 overflow on T4) skips the batch; the run
stops only if more than --max-skips batches are skipped.
Writes <out>/adapter/ (PEFT adapter of the best held-out logloss, base_model_name_or_path = --base-id) and
<out>/train_meta.json (rewritten at every eval, so a killed run still leaves its numbers).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import transformers
from sklearn.metrics import log_loss, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import common
from train import bucketed_batches, heldout_mask


def fold_of(s1_idx):
    """0/1 per S1: top bit of a 32-bit multiplicative hash (Knuth), independent of heldout_mask's mod-100 hash."""
    return ((s1_idx.astype(np.uint64) * np.uint64(0x9E3779B1)) % np.uint64(2**32)) >> np.uint64(31)


def tokenize(tok, a, b, max_length, step=100_000):
    """Token ids as compact int32 arrays (Python int lists for millions of pairs exhausted Kaggle RAM before)."""
    ids = []
    for s in range(0, len(a), step):
        enc = tok(a[s:s + step], b[s:s + step], truncation=True, max_length=max_length)["input_ids"]
        ids += [np.asarray(x, dtype=np.int32) for x in enc]
        del enc
    return ids


def evaluate(model, ids, y, pad_id, dev, fp16):
    p = common.score_ids(model, ids, pad_id, dev, fp16=fp16, log_every=10**12)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {"logloss": float(log_loss(y, pc, labels=[0, 1])), "auc": float(roc_auc_score(y, p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--set", default="DEV_REST_REST")
    ap.add_argument("--base", required=True, help="local dir (or hub id) of BAAI/bge-reranker-v2-m3")
    ap.add_argument("--base-id", default="BAAI/bge-reranker-v2-m3", help="written into the saved adapter_config")
    ap.add_argument("--init-adapter", required=True)
    ap.add_argument("--fold", type=int, choices=[0, 1], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-hours", type=float, default=9.5, help="training wall clock (excludes tokenization)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--heldout-cap", type=int, default=30_000)
    ap.add_argument("--max-skips", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="first N pairs only (smoke)")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + args.fold)
    dev = common.device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    pf = pq.ParquetFile(Path(args.pairs) / f"pairs_{args.set}.parquet")
    t = pf.read(columns=["s1_idx", "text_a", "text_b", "label"])
    if args.limit:
        t = t.slice(0, args.limit)
    s1 = t.column("s1_idx").to_numpy()
    keep = np.flatnonzero(fold_of(s1) == args.fold)
    ho = heldout_mask(s1[keep])
    tr_rows, ho_rows = keep[~ho], keep[ho]
    if ho_rows.size > args.heldout_cap:  # whole S1 groups, chosen by a seeded shuffle until the cap
        ho_s1, cnt = np.unique(s1[ho_rows], return_counts=True)
        order = np.random.default_rng(0).permutation(ho_s1.size)
        ho_rows = ho_rows[np.isin(s1[ho_rows], ho_s1[order[np.cumsum(cnt[order]) <= args.heldout_cap]])]
    y_all = t.column("label").to_numpy().astype(np.float32)
    a_all, b_all = t.column("text_a"), t.column("text_b")
    print(f"fold {args.fold}: {tr_rows.size:,} train pairs ({np.unique(s1[tr_rows]).size:,} S1, "
          f"pos {y_all[tr_rows].mean():.3f}), held-out {ho_rows.size:,} pairs", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    t0 = time.time()
    take = lambda col, rows: col.take(rows).to_pylist()
    tr_ids = tokenize(tok, take(a_all, tr_rows), take(b_all, tr_rows), args.max_length)
    ho_ids = tokenize(tok, take(a_all, ho_rows), take(b_all, ho_rows), args.max_length)
    y, ho_y = y_all[tr_rows], y_all[ho_rows]
    del t, a_all, b_all
    lens = np.array([len(x) for x in tr_ids], dtype=np.int32)
    print(f"tokenized in {time.time() - t0:.0f}s, mean length {lens.mean():.1f}", flush=True)

    from peft import PeftModel
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1)
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    model.to(dev)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp16 = dev.type == "cuda" and torch.cuda.get_device_capability()[0] < 8  # T4: fp16 + GradScaler
    amp_dtype = torch.float16 if fp16 else torch.bfloat16
    use_amp = dev.type == "cuda"
    print(f"device {dev}, amp {amp_dtype if use_amp else 'off'}, trainable params {n_tr:,}", flush=True)

    steps = int(np.ceil(tr_rows.size / args.batch) * args.epochs)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    warm = max(1, int(0.03 * steps))
    budget = args.budget_hours * 3600
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and fp16)
    lossf = torch.nn.BCEWithLogitsLoss()

    meta = {"script": "ce/train_continue.py", "base": args.base_id, "init_adapter": args.init_adapter,
            "set": args.set, "fold": args.fold, "fold_rule": "top bit of s1_idx * 0x9E3779B1 mod 2^32",
            "heldout": f"5% S1 hash within fold, cap {args.heldout_cap} pairs",
            "input_format": "text_a='{s1 name} | {s1 addr}', text_b='{pool name} | {pool addr}'",
            "max_length": args.max_length, "batch": args.batch, "lr": args.lr, "warmup_steps": warm,
            "steps_planned": steps, "budget_hours": args.budget_hours, "limit": args.limit,
            "train_pairs_available": int(tr_rows.size), "heldout_pairs": int(ho_rows.size),
            "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else dev.type,
            "torch": torch.__version__, "transformers": transformers.__version__, "evals": []}

    def checkpoint(step, seen, t_train, skips, final=False):
        m = evaluate(model, ho_ids, ho_y, tok.pad_token_id, dev, fp16)
        m.update(step=step, pairs_seen=seen, hours=round((time.time() - t_start) / 3600, 3))
        meta["evals"].append(m)
        print("held-out", m, flush=True)
        best = meta.get("best")
        if best is None or m["logloss"] < best["logloss"]:
            meta["best"] = m
            model.save_pretrained(out / "adapter")
            tok.save_pretrained(out / "adapter")
            cfg = out / "adapter" / "adapter_config.json"
            c = json.loads(cfg.read_text())
            c["base_model_name_or_path"] = args.base_id
            common.write_json(cfg, c)
        meta.update(steps_done=step, pairs_seen=seen, skipped_batches=skips, final=final,
                    train_pairs_per_s=round(seen / max(t_train, 1e-9), 1), train_seconds=round(t_train, 1),
                    peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None)
        common.write_json(out / "train_meta.json", meta)

    meta["before"] = evaluate(model, ho_ids, ho_y, tok.pad_token_id, dev, fp16)
    print("held-out before (ce_rec as is)", meta["before"], flush=True)
    step, seen, skips, t_train, t_loop = 0, 0, 0, 0.0, time.time()
    stop = False
    for b in bucketed_batches(lens, args.batch, rng):
        frac = max(step / steps, (time.time() - t_loop) / budget)
        if frac >= 1 or step >= steps:
            break
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (step + 1) / warm) * (1 - frac)
        model.train()
        ids, att = common.collate([tr_ids[i] for i in b], tok.pad_token_id)
        tgt = torch.from_numpy(y[b]).to(dev)
        ts = time.time()
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
            logit = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits[:, 0]
        loss = lossf(logit.float(), tgt)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            skips += 1
            print(f"step {step}: non-finite loss, batch skipped ({skips})", flush=True)
            if skips > args.max_skips:
                print("too many skipped batches, stopping", flush=True)
                stop = True
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - ts
        step += 1
        seen += len(b)
        if step % 200 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{seen / t_train:.0f} pairs/s, {(time.time() - t_loop) / 3600:.2f} h", flush=True)
        if step % args.eval_every == 0:
            checkpoint(step, seen, t_train, skips)
        if stop:
            break
    checkpoint(step, seen, t_train, skips, final=True)
    print(f"done fold {args.fold}: {step} steps, {seen:,} pairs, best {meta['best']} -> {out / 'adapter'}", flush=True)


if __name__ == "__main__":
    main()
