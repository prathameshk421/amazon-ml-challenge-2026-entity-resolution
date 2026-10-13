"""Kaggle kernel (2x T4), two trainings at once, one per GPU:
  GPU 0: Qwen/Qwen3-Reranker-0.6B LoRA on DEV_REST_180K (ce/train_qwen.py), same S1 as ce_rec -> head-to-head
  GPU 1: ce_rec v2-m3 LoRA continued on DEV_REST_REST fold 0 (ce/train_continue.py)
Each then scores MATCH_VALID (neither trained on it) with its best adapter.

Inputs (by glob under /kaggle/input): pair files (dataset ber-ce-pairs); scripts + ce_rec adapter files, flat
(dataset ber-ce-continue, staged by ce/kaggle_continue/stage.py).
Network: pip peft if missing; the two base checkpoints at pinned revisions. Training is offline.
Outputs: /kaggle/working/{qwen,v2m3_f0}/{adapter/, train_meta.json, MATCH_VALID.parquet}, {qwen,v2m3_f0}.log.
"""
import glob
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

BUDGET_HOURS = os.environ.get("BUDGET_HOURS", "8.5")
V2M3 = ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        ["config.json", "model.safetensors", "sentencepiece.bpe.model", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json"])
QWEN = ("Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
         "generation_config.json", "chat_template.jinja"])
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json"]
W = Path("/kaggle/working")


def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise SystemExit(f"input not found: {pattern}")
    return Path(hits[0]).parent


def run(name, gpu, cmds, scripts):
    """Run commands in order on one GPU, prefixing output; a failed command stops that GPU's chain only."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    with open(W / f"{name}.log", "w") as f:
        for cmd in cmds:
            print(f"[{name}] >>> {' '.join(cmd)}", flush=True)
            p = subprocess.Popen([sys.executable, "-u", *cmd], cwd=scripts, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                f.write(line)
                f.flush()
                if "it/s]" not in line:
                    print(f"[{name}] {line}", end="", flush=True)
            if p.wait():
                print(f"[{name}] exit code {p.returncode}, stopping this GPU", flush=True)
                return


def main():
    t0 = time.time()
    # Kaggle image ships torchao 0.10; peft's LoRA dispatch raises ImportError for torchao < 0.16. Unused here.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    try:
        import peft  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft==0.21.0"], check=True)
    import torch
    from huggingface_hub import snapshot_download
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print("gpus", gpus, flush=True)
    pairs = find("pairs_DEV_REST_REST.parquet")
    scripts = find("train_continue.py")
    adapter = Path("/tmp/ce_rec_adapter")
    adapter.mkdir(exist_ok=True)
    for f in ADAPTER_FILES:
        shutil.copy2(scripts / f, adapter / f)
    v2m3 = snapshot_download(V2M3[0], revision=V2M3[1], local_dir="/tmp/v2m3", allow_patterns=V2M3[2])
    qwen = snapshot_download(QWEN[0], revision=QWEN[1], local_dir="/tmp/qwen", allow_patterns=QWEN[2])
    print(f"inputs ready in {time.time() - t0:.0f}s", flush=True)
    budget = BUDGET_HOURS if len(gpus) > 1 else str(round(float(BUDGET_HOURS) / 2, 2))  # 12 h session limit

    jobs = {
        "qwen": (0, [["train_qwen.py", "--pairs", str(pairs), "--base", qwen, "--out", str(W / "qwen"),
                      "--budget-hours", budget, "--score-set", "MATCH_VALID"]]),
        "v2m3_f0": (1 if len(gpus) > 1 else 0, [
            ["train_continue.py", "--pairs", str(pairs), "--base", v2m3, "--init-adapter", str(adapter),
             "--fold", "0", "--out", str(W / "v2m3_f0"), "--budget-hours", budget],
            ["inference.py", "--pairs", str(pairs), "--model", str(W / "v2m3_f0" / "adapter"), "--base", v2m3,
             "--out", str(W / "v2m3_f0"), "--set", "MATCH_VALID"],
            ["inference.py", "--pairs", str(pairs), "--out", str(W / "v2m3_f0"), "--set", "MATCH_VALID", "--merge"]]),
    }
    if len(gpus) < 2:  # one GPU: run them one after the other, halving each budget
        print(f"only one GPU: running sequentially, {budget} h each", flush=True)
    threads = [threading.Thread(target=run, args=(n, g, c, scripts)) for n, (g, c) in jobs.items()]
    for th in threads:
        th.start()
        if len(gpus) < 2:
            th.join()
    for th in threads:
        th.join()
    print(f"total {(time.time() - t0) / 3600:.2f} h", flush=True)
    for n in jobs:
        print(n, "adapter", (W / n / "adapter" / "adapter_model.safetensors").exists(),
              "MATCH_VALID", (W / n / "MATCH_VALID.parquet").exists(), flush=True)


if __name__ == "__main__":
    main()
