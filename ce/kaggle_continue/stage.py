"""Stage the Kaggle uploads for ce/kaggle_continue/kaggle_run.py under work/ce/kaggle_continue/.

  .venv/bin/python ce/kaggle_continue/stage.py --user prathameshkinagi
  .venv/bin/kaggle datasets create -p work/ce/kaggle_continue/data      (later: datasets version -m ...)
  .venv/bin/kaggle kernels push -p work/ce/kaggle_continue/kernel

The pair files come from the existing private dataset <user>/ber-ce-pairs; this one only carries code + the adapter.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "cross_encoder_lora" / "adapter"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--budget-hours", default="8.5")
    args = ap.parse_args()
    out = ROOT / "work" / "ce" / "kaggle_continue"
    data, kern = out / "data", out / "kernel"
    for d in (data, kern):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for f in ("train_continue.py", "train_qwen.py", "train.py", "common.py", "inference.py"):
        shutil.copy2(ROOT / "ce" / f, data / f)
    for f in ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(ADAPTER / f, data / f)
    (data / "dataset-metadata.json").write_text(json.dumps(
        {"title": "ber-ce-continue", "id": f"{args.user}/ber-ce-continue", "licenses": [{"name": "other"}]}, indent=1))
    src = (ROOT / "ce" / "kaggle_continue" / "kaggle_run.py").read_text()
    (kern / "kaggle_run.py").write_text(src.replace('os.environ.get("BUDGET_HOURS", "8.5")',
                                                    f'os.environ.get("BUDGET_HOURS", "{args.budget_hours}")'))
    (kern / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.user}/ber-ce-qwen-v2m3", "title": "ber-ce-qwen-v2m3", "code_file": "kaggle_run.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{args.user}/ber-ce-pairs", f"{args.user}/ber-ce-continue"],
        "kernel_sources": [], "competition_sources": []}, indent=1))
    print("staged", data, kern)


if __name__ == "__main__":
    main()
