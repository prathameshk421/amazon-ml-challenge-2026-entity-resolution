"""Score every pair of a set with a fine-tuned cross-encoder.

  python ce/inference.py --pairs <dir> --model <out>/model --out <dir> --set TEST [--shard 0/2] [--gpu 0]
  python ce/inference.py --pairs <dir> --out <dir> --set TEST --merge

Scoring writes <out>/parts/<SET>/s<k>of<n>_c<j>.parquet per 1M-row chunk; existing chunks are skipped, so a run
killed by the 12 h session limit resumes where it stopped. Two processes (--shard 0/2 --gpu 0, --shard 1/2
--gpu 1) use both T4s. --merge concatenates the chunks in pair-file order into <out>/<SET>.parquet
(s1_idx:int32, pool_idx:uint32, p_ce:float32) and checks the row count and order against the pair file.
"""
import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK = 1_000_000


def shard_range(n_rows, k, n):
    return n_rows * k // n, n_rows * (k + 1) // n


def read_rows(pf, a, b, columns):
    """Rows [a, b) of a parquet file, reading only the row groups that overlap them."""
    groups, start, first = [], 0, None
    for g in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if start + n > a and start < b:
            groups.append(g)
            first = start if first is None else first
        start += n
    return pf.read_row_groups(groups, columns=columns).slice(a - first, b - a)


def rss_gb():
    try:
        with open("/proc/self/status") as f:
            return next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) / 2**20
    except OSError:
        return float("nan")


def run_score(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    import common

    k, n = map(int, args.shard.split("/"))
    src = Path(args.pairs) / f"pairs_{args.set}.parquet"
    lo, hi = shard_range(pq.ParquetFile(src).metadata.num_rows, k, n)
    parts = Path(args.out) / "parts" / args.set
    parts.mkdir(parents=True, exist_ok=True)
    dev = common.device()
    tok = AutoTokenizer.from_pretrained(args.model)
    if (Path(args.model) / "adapter_config.json").exists():
        # LoRA adapter (+ fully trained head in modules_to_save): load the base, merge, score as a plain model
        import json
        from peft import PeftModel
        base = args.base or json.loads((Path(args.model) / "adapter_config.json").read_text())["base_model_name_or_path"]
        model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1)
        model = PeftModel.from_pretrained(model, args.model).merge_and_unload()
        print(f"merged LoRA adapter {args.model} into {base}", flush=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
    # bf16 on CUDA GPUs with native bf16 (compute capability >= 8: A10G/L4/L40S; fp16 overflowed to NaN in one
    # model's first training run), fp16 on T4 (is_bf16_supported() is True there too, via slow emulation), fp32 off CUDA
    bf16 = dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    model = (model.to(torch.bfloat16) if bf16 else model.half() if dev.type == "cuda" else model.float()).to(dev)
    print(f"device {dev}, dtype {next(model.parameters()).dtype}", flush=True)
    # Memory: read only this chunk's rows (not the whole set), keep token ids as compact int32 arrays, free per
    # chunk. The first TEST run held the full table + Python int lists and was OOM-killed after 3M rows/process.
    pf = pq.ParquetFile(src)
    t_all, n_done = time.time(), 0
    for j, c0 in enumerate(range(lo, hi, CHUNK)):
        path = parts / f"s{k}of{n}_c{j:03d}.parquet"
        if path.exists():
            continue
        t = read_rows(pf, c0, min(c0 + CHUNK, hi), ["s1_idx", "pool_idx", "text_a", "text_b"])
        t0 = time.time()
        a, b = t.column("text_a").to_pylist(), t.column("text_b").to_pylist()
        ids = []
        for s in range(0, len(a), 100_000):
            enc = tok(a[s:s + 100_000], b[s:s + 100_000], truncation=True, max_length=args.max_length)["input_ids"]
            ids += [np.asarray(x, dtype=np.int32) for x in enc]
            del enc
        del a, b
        p = common.score_ids(model, ids, tok.pad_token_id, dev, tokens_per_batch=args.tokens_per_batch,
                             fp16=not bf16)
        tmp = path.with_name(path.name + ".tmp")
        pq.write_table(pa.table({"s1_idx": t.column("s1_idx"), "pool_idx": t.column("pool_idx"),
                                 "p_ce": pa.array(p)}), tmp)
        tmp.replace(path)
        n_done += t.num_rows
        print(f"{args.set} shard {k}/{n} chunk {j}: {t.num_rows:,} rows, "
              f"{t.num_rows / (time.time() - t0):.0f} pairs/s (incl. tokenization), rss {rss_gb():.1f} GB", flush=True)
        del t, ids, p
        gc.collect()
    print(f"{args.set} shard {k}/{n} done: {n_done:,} new rows in {time.time() - t_all:.0f}s", flush=True)


def run_merge(args):
    src = Path(args.pairs) / f"pairs_{args.set}.parquet"
    parts = sorted((Path(args.out) / "parts" / args.set).glob("s*of*_c*.parquet"),
                   key=lambda p: (int(p.name[1:p.name.index("of")]), p.name))
    t = pa.concat_tables([pq.read_table(p) for p in parts])
    ref = pq.read_table(src, columns=["s1_idx", "pool_idx"])
    if t.num_rows != ref.num_rows:
        raise RuntimeError(f"{args.set}: {t.num_rows:,} scored rows vs {ref.num_rows:,} pairs (missing chunks?)")
    for c in ("s1_idx", "pool_idx"):
        if not np.array_equal(t.column(c).to_numpy(), ref.column(c).to_numpy()):
            raise RuntimeError(f"{args.set}: {c} order differs from the pair file")
    p = t.column("p_ce").to_numpy()
    out = Path(args.out) / f"{args.set}.parquet"
    pq.write_table(pa.table({"s1_idx": pa.array(t.column("s1_idx").to_numpy().astype(np.int32)),
                             "pool_idx": pa.array(t.column("pool_idx").to_numpy().astype(np.uint32)),
                             "p_ce": pa.array(p.astype(np.float32))}), out, compression="zstd")
    print(f"{out}: {t.num_rows:,} rows, non-finite {int((~np.isfinite(p)).sum())}, mean p {np.nanmean(p):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--model")
    ap.add_argument("--base", help="base checkpoint for a LoRA adapter (default: adapter_config.json's path)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--tokens-per-batch", type=int, default=65_536)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        run_merge(args)
    else:
        if not args.model:
            ap.error("--model is required for scoring")
        run_score(args)


if __name__ == "__main__":
    main()
