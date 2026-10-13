"""LoRA fine-tune of Qwen/Qwen3-Reranker-0.6B (Apache-2.0) on DEV_REST_180K pairs: head-to-head with ce_rec.

  python ce/train_qwen.py --pairs <dir> --base <Qwen3-Reranker-0.6B dir> --out <dir> [--budget-hours 8.5]

Same training S1 as ce_rec (Friend B's v2-m3 LoRA, DEV_REST_180K), so MATCH_VALID scores of the two are a fair
comparison: neither model trained on MATCH_VALID.

Qwen3-Reranker is a causal LM; the pair score is P(yes) = sigmoid(logit["yes"] - logit["no"]) at the last position.
Prompt (shortened from the model card's system + instruct template to save tokens; fine-tuning adapts to it):
  <|im_start|>user\n<Query>: {s1}\n<Document>: {pool}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
{s1} / {pool} = "{name} | {addr}" (the pair files' text_a / text_b), each cut to --side-tokens tokens.
Left padding, so the last position is the answer slot for every row; only that position's logits are computed.
Loss: BCE on logit["yes"] - logit["no"]. LoRA r=16 on q/k/v/o/gate/up/down. fp32 master weights + fp16 autocast
+ GradScaler on T4 (no native bf16), bf16 autocast on compute capability >= 8. Non-finite loss skips the batch.
Schedule / budget / checkpointing as ce/train_continue.py. At the end scores --score-set (default MATCH_VALID) with
the best adapter into <out>/<SET>.parquet (s1_idx, pool_idx, p_ce).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import transformers
from sklearn.metrics import log_loss, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

import common
from train import bucketed_batches, heldout_mask

PREFIX = "<|im_start|>user\n<Query>: "
MID = "\n<Document>: "
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Enc:
    def __init__(self, tok, side_tokens):
        self.tok, self.n = tok, side_tokens
        self.pre, self.mid, self.suf = (np.asarray(tok(s, add_special_tokens=False)["input_ids"], dtype=np.int32)
                                        for s in (PREFIX, MID, SUFFIX))
        self.yes, self.no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
        self.pad = tok.pad_token_id

    def __call__(self, a, b, step=100_000):
        out = []
        for s in range(0, len(a), step):
            ea = self.tok(a[s:s + step], add_special_tokens=False)["input_ids"]
            eb = self.tok(b[s:s + step], add_special_tokens=False)["input_ids"]
            out += [np.concatenate([self.pre, np.asarray(x[:self.n], dtype=np.int32), self.mid,
                                    np.asarray(z[:self.n], dtype=np.int32), self.suf]) for x, z in zip(ea, eb)]
        return out


def collate_left(rows, pad):
    n = max(len(x) for x in rows)
    ids = torch.full((len(rows), n), pad, dtype=torch.long)
    att = torch.zeros((len(rows), n), dtype=torch.long)
    for i, x in enumerate(rows):
        ids[i, n - len(x):] = torch.from_numpy(x.astype(np.int64))
        att[i, n - len(x):] = 1
    return ids, att


def yes_no_logit(model, ids, att, enc, dev):
    out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), logits_to_keep=1).logits[:, -1, :]
    return out[:, enc.yes].float() - out[:, enc.no].float()


@torch.no_grad()
def score(model, rows, enc, dev, amp_dtype, use_amp, tokens_per_batch=32_768, log_every=10**12):
    model.eval()
    order = np.argsort([len(x) for x in rows], kind="stable")
    p = np.empty(len(rows), dtype=np.float32)
    i, t0, next_log = 0, time.time(), log_every
    while i < len(order):
        n = max(1, tokens_per_batch // len(rows[order[min(i + 255, len(order) - 1)]]))
        idx = order[i:i + min(n, 256)]
        ids, att = collate_left([rows[k] for k in idx], enc.pad)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
            p[idx] = torch.sigmoid(yes_no_logit(model, ids, att, enc, dev)).cpu().numpy()
        i += len(idx)
        if i >= next_log:
            print(f"  scored {i:,}/{len(rows):,} ({i / (time.time() - t0):.0f} pairs/s)", flush=True)
            next_log += log_every
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--set", default="DEV_REST_180K")
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-id", default="Qwen/Qwen3-Reranker-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-hours", type=float, default=8.5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--side-tokens", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--heldout-cap", type=int, default=30_000)
    ap.add_argument("--max-skips", type=int, default=200)
    ap.add_argument("--score-set", default="MATCH_VALID")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = common.device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    t = pq.read_table(Path(args.pairs) / f"pairs_{args.set}.parquet", columns=["s1_idx", "text_a", "text_b", "label"])
    if args.limit:
        t = t.slice(0, args.limit)
    s1 = t.column("s1_idx").to_numpy()
    ho = heldout_mask(s1)
    tr_rows, ho_rows = np.flatnonzero(~ho), np.flatnonzero(ho)
    if ho_rows.size > args.heldout_cap:
        ho_s1, cnt = np.unique(s1[ho_rows], return_counts=True)
        order = np.random.default_rng(0).permutation(ho_s1.size)
        ho_rows = ho_rows[np.isin(s1[ho_rows], ho_s1[order[np.cumsum(cnt[order]) <= args.heldout_cap]])]
    y_all = t.column("label").to_numpy().astype(np.float32)
    print(f"{args.set}: {tr_rows.size:,} train pairs (pos {y_all[tr_rows].mean():.3f}), held-out {ho_rows.size:,}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    enc = Enc(tok, args.side_tokens)
    t0 = time.time()
    take = lambda rows: (t.column("text_a").take(rows).to_pylist(), t.column("text_b").take(rows).to_pylist())
    tr_ids, ho_ids = enc(*take(tr_rows)), enc(*take(ho_rows))
    y, ho_y = y_all[tr_rows], y_all[ho_rows]
    del t
    lens = np.array([len(x) for x in tr_ids], dtype=np.int32)
    print(f"tokenized in {time.time() - t0:.0f}s, mean length {lens.mean():.1f} (max {lens.max()})", flush=True)

    from peft import LoraConfig, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    model.to(dev)
    cuda = dev.type == "cuda"
    fp16 = cuda and torch.cuda.get_device_capability()[0] < 8
    amp_dtype = torch.float16 if fp16 else torch.bfloat16
    use_amp = cuda
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device {dev}, amp {amp_dtype if use_amp else 'off'}, trainable params {n_tr:,}", flush=True)

    steps = int(np.ceil(tr_rows.size / args.batch) * args.epochs)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    warm = max(1, int(0.03 * steps))
    budget = args.budget_hours * 3600
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and fp16)
    lossf = torch.nn.BCEWithLogitsLoss()

    meta = {"script": "ce/train_qwen.py", "base": args.base_id, "set": args.set, "heldout": "5% S1 hash, capped",
            "prompt": PREFIX + "{s1}" + MID + "{pool}" + SUFFIX, "side_tokens": args.side_tokens,
            "lora": "r16 a32 d0.05 q,k,v,o,gate,up,down", "batch": args.batch, "lr": args.lr, "warmup_steps": warm,
            "steps_planned": steps, "budget_hours": args.budget_hours, "limit": args.limit,
            "train_pairs_available": int(tr_rows.size), "heldout_pairs": int(ho_rows.size),
            "gpu": torch.cuda.get_device_name(0) if cuda else dev.type, "amp": str(amp_dtype) if use_amp else None,
            "torch": torch.__version__, "transformers": transformers.__version__, "evals": []}

    def evaluate():
        p = score(model, ho_ids, enc, dev, amp_dtype, use_amp)
        ok = np.isfinite(p)
        return {"logloss": float(log_loss(ho_y[ok], np.clip(p[ok], 1e-6, 1 - 1e-6), labels=[0, 1])),
                "auc": float(roc_auc_score(ho_y[ok], p[ok])), "non_finite": int((~ok).sum())}

    def checkpoint(step, seen, t_train, skips, final=False):
        m = evaluate()
        m.update(step=step, pairs_seen=seen, hours=round((time.time() - t_start) / 3600, 3))
        meta["evals"].append(m)
        print("held-out", m, flush=True)
        if meta.get("best") is None or m["logloss"] < meta["best"]["logloss"]:
            meta["best"] = m
            model.save_pretrained(out / "adapter")
            tok.save_pretrained(out / "adapter")
            cfg = out / "adapter" / "adapter_config.json"
            c = json.loads(cfg.read_text())
            c["base_model_name_or_path"] = args.base_id
            common.write_json(cfg, c)
        meta.update(steps_done=step, pairs_seen=seen, skipped_batches=skips, final=final,
                    train_pairs_per_s=round(seen / max(t_train, 1e-9), 1), train_seconds=round(t_train, 1),
                    peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None)
        common.write_json(out / "train_meta.json", meta)

    meta["before"] = evaluate()
    print("held-out before (zero-shot, short prompt)", meta["before"], flush=True)
    step, seen, skips, t_train, t_loop = 0, 0, 0, 0.0, time.time()
    for b in bucketed_batches(lens, args.batch, rng):
        frac = max(step / steps, (time.time() - t_loop) / budget)
        if frac >= 1 or step >= steps:
            break
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (step + 1) / warm) * (1 - frac)
        model.train()
        ids, att = collate_left([tr_ids[i] for i in b], enc.pad)
        tgt = torch.from_numpy(y[b]).to(dev)
        ts = time.time()
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
            logit = yes_no_logit(model, ids, att, enc, dev)
        loss = lossf(logit, tgt)
        opt.zero_grad(set_to_none=True)
        stop = False
        if not torch.isfinite(loss):
            skips += 1
            print(f"step {step}: non-finite loss, batch skipped ({skips})", flush=True)
            stop = skips > args.max_skips
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
        if cuda:
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
            print("too many skipped batches, stopping", flush=True)
            break
    checkpoint(step, seen, t_train, skips, final=True)
    print(f"done: {step} steps, {seen:,} pairs, best {meta['best']}", flush=True)

    if args.score_set:  # reload the best adapter (the live weights may be a later, worse step)
        from peft import PeftModel
        del model
        base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
        model = PeftModel.from_pretrained(base, out / "adapter").merge_and_unload().to(dev)
        st = pq.read_table(Path(args.pairs) / f"pairs_{args.score_set}.parquet",
                           columns=["s1_idx", "pool_idx", "text_a", "text_b"])
        if args.limit:
            st = st.slice(0, args.limit)
        t0 = time.time()
        p = score(model, enc(st.column("text_a").to_pylist(), st.column("text_b").to_pylist()), enc, dev,
                  amp_dtype, use_amp, log_every=20_000)
        pq.write_table(pa.table({"s1_idx": st.column("s1_idx"), "pool_idx": st.column("pool_idx"),
                                 "p_ce": pa.array(p)}), out / f"{args.score_set}.parquet")
        print(f"scored {args.score_set}: {st.num_rows:,} pairs in {time.time() - t0:.0f}s, "
              f"non-finite {int((~np.isfinite(p)).sum())}", flush=True)


if __name__ == "__main__":
    main()
