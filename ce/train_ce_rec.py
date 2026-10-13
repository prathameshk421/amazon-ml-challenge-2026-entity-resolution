#!/usr/bin/env python3
"""ce_rec smoke/full trainer: LoRA (r=16, attention) + full classifier head, bf16 on CUDA capability >= 8, else fp32.

Thermal-safe by design for a laptop box:
  - DataLoader num_workers=0, torch threads capped (default 4)
  - tokenization is per-sample Rust fast path (~ms per batch, negligible CPU)
  - background temp monitor logs CPU(Tctl)/GPU every 30s and PAUSES training
    above thresholds until the box cools.

Usage (smoke):
  .venv/bin/python code/business_entity_resolution/src/ce/train_ce_rec.py --pairs <pairs>.parquet \
      --out work/ce/ce_rec_smoke --max-rows 30000
Full epoch:
  .venv/bin/python code/business_entity_resolution/src/ce/train_ce_rec.py --pairs <pairs>.parquet \
      --out work/ce/ce_rec_ep1

The --pairs file needs columns name_s1, addr_s1, name_pool, addr_pool, label, s1_idx, pool_idx.
"""

import argparse, csv, itertools, json, os, shutil, sys, threading, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

TEXT_COLS = ["name_s1", "addr_s1", "name_pool", "addr_pool"]


def read_temps():
    """Return (cpu_c, gpu_c) or (None, None) on failure."""
    cpu = gpu = None
    try:
        base = Path("/sys/class/hwmon")
        for h in base.iterdir():
            name = (h / "name").read_text().strip() if (h / "name").exists() else ""
            if name == "k10temp":
                for f in h.glob("temp*_input"):
                    v = int(f.read_text().strip()) / 1000.0
                    if "label" not in str(f):
                        cpu = v if cpu is None else max(cpu, v)
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu = float(out.stdout.strip().split()[0])
    except Exception:
        pass
    return cpu, gpu


class TempGuard(threading.Thread):
    """Logs temps; sets pause_event when too hot, clears when cooled."""

    def __init__(
        self,
        log_path,
        pause_event,
        stop_event,
        cpu_pause=95.0,
        gpu_pause=88.0,
        cpu_resume=88.0,
        gpu_resume=80.0,
        append=False,
    ):
        super().__init__(daemon=True)
        self.log_path = log_path
        self.pause_event = pause_event
        self.stop_event = stop_event
        self.th = (cpu_pause, gpu_pause, cpu_resume, gpu_resume)
        exists = append and os.path.exists(log_path) and os.path.getsize(log_path) > 0
        self.f = open(log_path, "a" if exists else "w", newline="")
        self.w = csv.writer(self.f)
        if not exists:
            self.w.writerow(["ts", "cpu_c", "gpu_c", "paused"])

    def run(self):
        cpu_p, gpu_p, cpu_r, gpu_r = self.th
        while not self.stop_event.is_set():
            cpu, gpu = read_temps()
            hot = (cpu is not None and cpu >= cpu_p) or (gpu is not None and gpu >= gpu_p)
            cool = (cpu is None or cpu <= cpu_r) and (gpu is None or gpu <= gpu_r)
            if hot and not self.pause_event.is_set():
                print(f"[tempguard] HOT cpu={cpu} gpu={gpu} -> pausing", flush=True)
                self.pause_event.set()
            elif self.pause_event.is_set() and cool:
                print(f"[tempguard] cooled cpu={cpu} gpu={gpu} -> resuming", flush=True)
                self.pause_event.clear()
            self.w.writerow([time.time(), cpu, gpu, int(self.pause_event.is_set())])
            self.f.flush()
            self.stop_event.wait(30)


def ditto_row(name_a, addr_a, name_b, addr_b):
    return f"{name_a} | {addr_a}", f"{name_b} | {addr_b}"


class PairDS(Dataset):
    def __init__(self, tbl, tok, max_len):
        self.t, self.tok, self.ml = tbl, tok, max_len

    def __len__(self):
        return self.t.num_rows

    def __getitem__(self, i):
        r = self.t.slice(i, 1).to_pylist()[0]
        a, b = ditto_row(r["name_s1"], r["addr_s1"], r["name_pool"], r["addr_pool"])
        enc = self.tok(a, b, truncation=True, max_length=self.ml)
        return (
            enc["input_ids"],
            enc["attention_mask"],
            float(r["label"]),
            int(r["s1_idx"]),
            int(r["pool_idx"]),
        )


def collate(batch):
    ids = [torch.tensor(b[0], dtype=torch.long) for b in batch]
    msk = [torch.tensor(b[1], dtype=torch.long) for b in batch]
    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True)
    msk = torch.nn.utils.rnn.pad_sequence(msk, batch_first=True)
    y = torch.tensor([b[2] for b in batch])
    return ids, msk, y


def auc_rank_sum(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(1, len(order) + 1)
    n1 = y_true.sum()
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    for ids, msk, y in loader:
        ids, msk, y = ids.to(dev), msk.to(dev), y.to(dev)
        logit = model(input_ids=ids, attention_mask=msk).logits.squeeze(-1)
        loss_sum += F.binary_cross_entropy_with_logits(logit, y, reduction="sum").item()
        n += len(y)
        ys += y.cpu().tolist()
        ps += logit.cpu().tolist()
    model.train()
    return loss_sum / n, auc_rank_sum(np.array(ys), np.array(ps))


def latest_ckpt(out):
    """Return (step, path) of newest checkpoints/step_* or (None, None)."""
    d = Path(out) / "checkpoints"
    if not d.is_dir():
        return None, None
    best = (None, None)
    for p in d.iterdir():
        if p.is_dir() and p.name.startswith("step_") and (p / "state.json").exists():
            try:
                s = int(p.name.split("_")[1])
            except ValueError:
                continue
            if best[0] is None or s > best[0]:
                best = (s, p)
    return best


def save_ckpt(model, opt, path, state):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model_state.pt")
    torch.save(opt.state_dict(), path / "optimizer.pt")
    json.dump(state, open(path / "state.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = full file (smoke: e.g. 30000)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-s1", type=int, default=1000, help="last N S1 groups held out for eval")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument(
        "--ckpt-every",
        type=int,
        default=2000,
        help="save mid-run checkpoint every N train steps (0 = off)",
    )
    ap.add_argument(
        "--resume", action="store_true", help="resume from latest work/ce/<out>/checkpoints/step_*"
    )
    ap.add_argument(
        "--init-from",
        default=None,
        help="warm-start: load model_state.pt (same arch) into fresh wrapper + fresh optimizer, "
        "e.g. ep1 checkpoints/step_60000/model_state.pt. Recorded in run.json args.",
    )
    ap.add_argument(
        "--init-adapter",
        default=None,
        help="warm-start: load PEFT adapter dir (adapter_model.safetensors, same LoRA recipe) "
        "into fresh wrapper + fresh optimizer, e.g. ep1 final adapter. "
        "Mutually exclusive with --init-from/--resume. Recorded in run.json args.",
    )
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_amp = dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 8
    dtype = torch.bfloat16 if is_amp else torch.float32
    print(f"[setup] device={dev} threads={a.threads} dtype={dtype}", flush=True)
    cpu0, gpu0 = read_temps()
    print(f"[setup] idle temps cpu={cpu0}C gpu={gpu0}C", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[data] loading pairs ...", flush=True)
    t = pq.read_table(a.pairs)
    if a.max_rows:
        t = t.slice(0, a.max_rows)
    df_null = sum(t.column(c).null_count for c in TEXT_COLS)
    print(f"[data] rows={t.num_rows} null_text_cells={df_null}", flush=True)
    assert df_null == 0, "null-drop rule: unexpected nulls, refusing to train"
    # group-aware split: last eval_s1 S1 groups -> eval
    s1 = t.column("s1_idx").to_pylist()
    uniq = list(dict.fromkeys(s1))
    cut = set(uniq[-a.eval_s1 :])
    is_eval = np.array([x in cut for x in s1])
    import pyarrow.compute as pc

    tr_t = t.filter(
        pc.invert(pc.is_in(t.column("s1_idx"), value_set=__import__("pyarrow").array(list(cut))))
    )
    ev_t = t.filter(pc.is_in(t.column("s1_idx"), value_set=__import__("pyarrow").array(list(cut))))
    print(
        f"[data] train={tr_t.num_rows} eval={ev_t.num_rows} eval_s1_groups={len(cut)}", flush=True
    )

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    base = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=1, torch_dtype=dtype, trust_remote_code=False
    ).to(dev)
    base.gradient_checkpointing_enable()
    base.config.use_cache = False
    cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query", "key", "value"],
        modules_to_save=["classifier"],
        task_type="SEQ_CLS",
    )
    model = get_peft_model(base, cfg).to(dev)
    model.train()
    if a.init_from is not None:
        assert not a.resume and a.init_adapter is None, (
            "--init-from is mutually exclusive with --resume/--init-adapter"
        )
        sd = torch.load(a.init_from, map_location=dev)
        res = model.load_state_dict(sd, strict=False)
        assert not res.missing_keys and not res.unexpected_keys, (
            f"--init-from key mismatch: missing={res.missing_keys} unexpected={res.unexpected_keys}"
        )
        print(f"[init] warm-started from {a.init_from}", flush=True)
    if a.init_adapter is not None:
        assert not a.resume and a.init_from is None, (
            "--init-adapter is mutually exclusive with --resume/--init-from"
        )
        from safetensors.torch import load_file

        from peft.utils.save_and_load import set_peft_model_state_dict

        ad = Path(a.init_adapter)
        assert (ad / "adapter_model.safetensors").is_file(), f"no adapter weights in {ad}"
        assert (ad / "adapter_config.json").is_file(), f"no adapter config in {ad}"
        acfg = json.load(open(ad / "adapter_config.json"))
        assert acfg.get("peft_type") == "LORA" and acfg.get("r") == 16, acfg
        assert acfg.get("lora_alpha") == 32 and acfg.get("task_type") == "SEQ_CLS", acfg
        assert set(acfg.get("target_modules", [])) == {"query", "key", "value"}, acfg
        assert "classifier" in acfg.get("modules_to_save", []), acfg
        sd = load_file(str(ad / "adapter_model.safetensors"))
        set_peft_model_state_dict(model, sd, adapter_name="default")
        print(f"[init] warm-started from adapter {a.init_adapter} ({len(sd)} tensors)", flush=True)
    model.print_trainable_parameters()
    print(
        f"[mem] {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated, "
        f"{torch.cuda.memory_reserved() / 1e9:.2f}GB reserved",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    resumed_from, start_ep, skip_batches = None, 0, 0
    step, opt_step = 0, 0
    if a.resume:
        _, ck = latest_ckpt(out)
        assert ck is not None, f"--resume but no checkpoints in {ckpt_dir}"
        st = json.load(open(ck / "state.json"))
        assert st["lr"] == a.lr, "lr changed since checkpoint, refusing resume"
        model.load_state_dict(torch.load(ck / "model_state.pt", map_location=dev))
        opt.load_state_dict(torch.load(ck / "optimizer.pt", map_location=dev))
        step, opt_step = st["step"], st["opt_step"]
        start_ep, skip_batches = st["epoch"], st["consumed"]
        resumed_from = ck.name
        print(
            f"[resume] {ck.name} step={step} opt_step={opt_step} "
            f"epoch={start_ep} skipping {skip_batches} batches",
            flush=True,
        )
    tr_ds, ev_ds = PairDS(tr_t, tok, a.max_len), PairDS(ev_t, tok, a.max_len)
    ev_ld = DataLoader(
        ev_ds, batch_size=a.batch * 2, shuffle=False, num_workers=0, collate_fn=collate
    )

    pause_ev, stop_ev = threading.Event(), threading.Event()
    guard = TempGuard(str(out / "temps.csv"), pause_ev, stop_ev, append=a.resume)
    guard.start()

    hist = []
    t0 = time.time()
    run_loss = 0.0
    model.zero_grad(set_to_none=True)
    for ep in range(start_ep, a.epochs):
        g = torch.Generator()
        g.manual_seed(a.seed + ep)
        tr_ld = DataLoader(
            tr_ds, batch_size=a.batch, shuffle=True, num_workers=0, collate_fn=collate, generator=g
        )
        it = (
            itertools.islice(tr_ld, skip_batches, None)
            if (ep == start_ep and skip_batches)
            else iter(tr_ld)
        )
        consumed = skip_batches if ep == start_ep else 0
        skip_batches = 0
        for ids, msk, y in it:
            consumed += 1
            while pause_ev.is_set():
                time.sleep(5)
            ids, msk, y = ids.to(dev), msk.to(dev), y.to(dev)
            logit = model(input_ids=ids, attention_mask=msk).logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y) / a.accum
            loss.backward()
            run_loss += loss.item() * a.accum
            step += 1
            if step % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                model.zero_grad(set_to_none=True)
                opt_step += 1
            if step % a.log_every == 0:
                el = time.time() - t0
                cur = run_loss / a.log_every
                print(
                    f"[ep{ep} step{step}] loss={cur:.4f} "
                    f"{el / step:.2f}s/step temps={read_temps()}",
                    flush=True,
                )
                if not np.isfinite(cur):
                    print("[abort] non-finite loss, stopping run", flush=True)
                    stop_ev.set()
                    sys.exit(1)
                hist.append({"ep": ep, "step": step, "loss": run_loss / a.log_every, "elapsed": el})
                run_loss = 0.0
            if step % a.eval_every == 0:
                el, ea = evaluate(model, ev_ld, dev)
                print(f"[ep{ep} step{step}] EVAL loss={el:.4f} auc={ea:.4f}", flush=True)
                hist.append({"ep": ep, "step": step, "eval_loss": el, "eval_auc": ea})
            if a.ckpt_every and step % a.ckpt_every == 0:
                new_ck = ckpt_dir / f"step_{step}"
                save_ckpt(
                    model,
                    opt,
                    new_ck,
                    {
                        "step": step,
                        "opt_step": opt_step,
                        "epoch": ep,
                        "consumed": consumed,
                        "lr": a.lr,
                    },
                )
                # rolling: keep latest only (resume reads latest_ckpt, final weights go to adapter/)
                for p in ckpt_dir.iterdir():
                    if p.is_dir() and p.name.startswith("step_") and p != new_ck:
                        shutil.rmtree(p, ignore_errors=True)
                print(f"[ep{ep} step{step}] ckpt saved", flush=True)
    el, ea = evaluate(model, ev_ld, dev)
    print(
        f"[done] EVAL loss={el:.4f} auc={ea:.4f} steps={step} time={time.time() - t0:.0f}s",
        flush=True,
    )
    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    json.dump(
        {
            "args": vars(a),
            "eval_loss": el,
            "eval_auc": ea,
            "steps": step,
            "history": hist,
            "idle_temps": [cpu0, gpu0],
            "resumed_from": resumed_from,
        },
        open(out / "run.json", "w"),
        indent=1,
    )
    stop_ev.set()
    guard.join(timeout=35)


if __name__ == "__main__":
    main()
