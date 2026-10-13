"""GPU smoke test on a synthetic pair-classification set (no contest data).

  python cloud/smoke_test.py [--model cross-encoder/ms-marco-MiniLM-L6-v2] [--steps 100] [--batch 32]
                             [--max-length 128] [--out /workspace/experiments/SMOKE]

Checks, in order: tokenizer, forward, backward, AdamW step, mixed precision (bf16 if supported, else fp16 +
GradScaler), save_pretrained, reload + identical logits, inference on held-out pairs. Records GPU, VRAM, CUDA,
torch, model, batch, seq length, non-pad tokens/s, sampled GPU utilization, peak VRAM -> <out>/metrics.json.

Synthetic data: random business-like records serialized with field labels. Positive = same record after
legal-suffix drop / word-order transposition / one-char typo; negative = a different record of the same country.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

WORDS = "alpha nova sri ganesh lakshmi royal star green city prime metro sun golden blue river park hill market " \
        "trading enterprises foods motors dental clinic bakery pharmacy textiles hardware electronics".split()
SUFFIX = ["llc", "inc", "pvt ltd", "sarl", "sas", "ltd", ""]
STREETS = ["main st", "mg road", "rue de la paix", "station road", "oak ave", "avenue victor hugo", "nehru nagar"]
CITY = {"US": ["austin tx", "denver co"], "IN": ["pune mh", "chennai tn"], "FR": ["paris", "lyon"]}


def record(rng):
    c = rng.choice(list(CITY))
    return {"name": " ".join(rng.sample(WORDS, rng.randint(2, 3))) + " " + rng.choice(SUFFIX),
            "address": f"{rng.randint(1, 999)} {rng.choice(STREETS)} {rng.choice(CITY[c])}", "country": c}


def perturb(r, rng):
    name = r["name"]
    for s in SUFFIX[:-1]:
        name = name.replace(" " + s, "")
    toks = name.split()
    if rng.random() < 0.5 and len(toks) > 1:
        toks = toks[::-1]
    name = " ".join(toks)
    if rng.random() < 0.5 and len(name) > 4:
        i = rng.randrange(len(name))
        name = name[:i] + rng.choice("aeiou") + name[i + 1:]
    return {**r, "name": name}


def ser(r):
    return f"name: {r['name']} [SEP] address: {r['address']} [SEP] country: {r['country']}"


def make_pairs(n, seed):
    rng = random.Random(seed)
    recs = [record(rng) for _ in range(n)]
    a, b, y = [], [], []
    for i, r in enumerate(recs):
        a.append(ser(r)); b.append(ser(perturb(r, rng))); y.append(1.0)
        j = rng.choice([k for k in range(max(0, i - 20), min(n, i + 20)) if recs[k]["country"] == r["country"] and k != i] or [(i + 1) % n])
        a.append(ser(r)); b.append(ser(recs[j])); y.append(0.0)
    return a, b, np.array(y, dtype=np.float32)


class GpuSampler(threading.Thread):
    """Samples nvidia-smi utilization.gpu every 0.5 s while training runs."""
    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout
                self.samples.append(float(o.split()[0]))
            except Exception:
                return
            time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/workspace/experiments/SMOKE")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    cuda = dev.type == "cuda"
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else None
    step_log = {}

    tok = AutoTokenizer.from_pretrained(args.model)
    ta, tb, y = make_pairs(2000, args.seed)
    va, vb, vy = make_pairs(300, args.seed + 1)
    enc = tok(ta[:4], tb[:4], truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
    step_log["tokenizer"] = f"ok, example len {enc['input_ids'].shape[1]}"

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1,
                                                               ignore_mismatched_sizes=True).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and amp_dtype == torch.float16)
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    sampler = GpuSampler()
    sampler.start()
    tokens, t_train, losses = 0, 0.0, []
    for step in range(args.steps):
        idx = rng.choice(len(y), args.batch, replace=False)
        b = tok([ta[i] for i in idx], [tb[i] for i in idx], truncation=True, max_length=args.max_length,
                padding=True, return_tensors="pt").to(dev)  # dynamic padding: pad to the batch max only
        tgt = torch.from_numpy(y[idx]).to(dev)
        model.train()
        t0 = time.time()
        with torch.autocast(device_type=dev.type, dtype=amp_dtype or torch.float32, enabled=cuda):
            logit = model(**b).logits[:, 0]
        loss = lossf(logit.float(), tgt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if cuda:
            torch.cuda.synchronize()
        if step >= 5:  # skip warm-up steps in the throughput number
            t_train += time.time() - t0
            tokens += int(b["attention_mask"].sum())
        losses.append(loss.item())
        if step == 0:
            step_log.update(forward="ok", backward="ok", optimizer="ok",
                            mixed_precision=str(amp_dtype).replace("torch.", "") if cuda else "off (no CUDA)")
    sampler.stop.set()
    assert all(np.isfinite(losses)), "non-finite loss"

    @torch.no_grad()
    def predict(m, a, bb):
        m.eval()
        out = []
        for s in range(0, len(a), 128):
            e = tok(a[s:s + 128], bb[s:s + 128], truncation=True, max_length=args.max_length, padding=True,
                    return_tensors="pt").to(dev)
            out.append(m(**e).logits[:, 0].float().cpu())
        return torch.cat(out)

    ck = os.path.join(args.out, "checkpoint")
    model.save_pretrained(ck, safe_serialization=True)
    tok.save_pretrained(ck)
    step_log["checkpoint_save"] = "ok"
    before = predict(model, va, vb)
    reloaded = AutoModelForSequenceClassification.from_pretrained(ck).to(dev)
    after = predict(reloaded, va, vb)
    diff = float((before - after).abs().max())
    assert diff < 1e-3, f"reloaded logits differ by {diff}"
    step_log["checkpoint_load"] = f"ok, max |logit diff| {diff:.2e}"
    auc = roc_auc_score(vy, torch.sigmoid(after).numpy())
    step_log["inference"] = f"ok, held-out AUC {auc:.3f} on {len(vy)} synthetic pairs"

    m = {"gpu": torch.cuda.get_device_name(0) if cuda else dev.type,
         "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if cuda else None,
         "cuda": torch.version.cuda, "torch": torch.__version__, "python": sys.version.split()[0],
         "model": args.model, "params_m": round(n_params / 1e6, 1), "batch": args.batch,
         "max_length": args.max_length, "steps": args.steps,
         "tokens_per_s": round(tokens / max(t_train, 1e-9)),
         "pairs_per_s": round((args.steps - 5) * args.batch / max(t_train, 1e-9), 1),
         "gpu_util_mean_pct": round(float(np.mean(sampler.samples)), 1) if sampler.samples else None,
         "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None,
         "loss_first10": round(float(np.mean(losses[:10])), 4), "loss_last10": round(float(np.mean(losses[-10:])), 4),
         "heldout_auc": round(float(auc), 4), "steps_ok": step_log}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=1)
    for k, v in m.items():
        print(f"{k:18s} {v}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
