"""GPU environment checks. Exit code 1 if any check fails; writes the results as JSON.

  python cloud/verify_env.py [--out /workspace/logs/verify_env.json] [--allow-no-cuda]

--allow-no-cuda: local dry run (Mac/CPU); CUDA checks are reported as skipped instead of failed.
"""
import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

CHECKS = {}


def check(name):
    def wrap(fn):
        CHECKS[name] = fn
        return fn
    return wrap


@check("cuda_available")
def _cuda(t):
    assert t.cuda.is_available(), "torch.cuda.is_available() is False"
    return {"device": t.cuda.get_device_name(0), "count": t.cuda.device_count(), "cuda": t.version.cuda,
            "cudnn": t.backends.cudnn.version(), "capability": ".".join(map(str, t.cuda.get_device_capability(0)))}


@check("gpu_memory_visible")
def _mem(t):
    p = t.cuda.get_device_properties(0)
    free, total = t.cuda.mem_get_info(0)
    assert total > 8 * 2**30, f"only {total / 2**30:.1f} GiB"
    return {"total_gib": round(p.total_memory / 2**30, 2), "free_gib": round(free / 2**30, 2)}


@check("allocate_tensors")
def _alloc(t):
    x = t.empty(int(2 * 2**30 / 4), dtype=t.float32, device="cuda")  # 2 GiB
    x.fill_(1.0)
    s = float(x[:1000].sum())
    del x
    t.cuda.empty_cache()
    assert s == 1000.0
    return {"allocated_gib": 2}


def _matmul(t, dtype):
    a = t.randn(4096, 4096, device="cuda", dtype=dtype)
    b = t.randn(4096, 4096, device="cuda", dtype=dtype)
    (a @ b).sum().item()
    t.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        c = a @ b
    t.cuda.synchronize()
    dt = time.time() - t0
    assert t.isfinite(c).all()
    return round(20 * 2 * 4096 ** 3 / dt / 1e12, 1)


@check("bf16")
def _bf16(t):
    assert t.cuda.is_bf16_supported(), "no native bf16 (use fp16 + GradScaler)"
    return {"matmul_tflops": _matmul(t, t.bfloat16)}


@check("fp16")
def _fp16(t):
    return {"matmul_tflops": _matmul(t, t.float16)}


@check("nccl")
def _nccl(t):
    import torch.distributed as dist
    return {"available": dist.is_nccl_available(), "version": ".".join(map(str, t.cuda.nccl.version()))}


def imports():
    out = {}
    for m in ["transformers", "sentence_transformers", "lightgbm", "tokenizers", "datasets", "accelerate",
              "evaluate", "sklearn", "pandas", "numpy", "scipy", "rapidfuzz", "pyarrow", "tqdm", "safetensors",
              "huggingface_hub"]:
        try:
            out[m] = {"ok": True, "version": getattr(importlib.import_module(m), "__version__", "?")}
        except Exception as e:  # report every failure, not just the first
            out[m] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-no-cuda", action="store_true")
    args = ap.parse_args()

    import torch
    res = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
           "hf_home": os.environ.get("HF_HOME"), "checks": {}}
    try:
        res["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                            "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        res["nvidia_smi"] = None
    if os.path.ismount("/workspace"):
        du = shutil.disk_usage("/workspace")
        res["workspace"] = {"mounted": True, "free_gib": round(du.free / 2**30, 1)}
    else:
        res["workspace"] = {"mounted": False}

    for name, fn in CHECKS.items():
        if not torch.cuda.is_available() and args.allow_no_cuda:
            res["checks"][name] = {"ok": None, "skipped": "no CUDA (--allow-no-cuda)"}
            continue
        try:
            res["checks"][name] = {"ok": True, **fn(torch)}
        except Exception as e:
            res["checks"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    res["imports"] = imports()
    if not args.allow_no_cuda:
        res["checks"]["workspace_mounted"] = {"ok": res["workspace"]["mounted"]}
        res["checks"]["hf_home_on_workspace"] = {"ok": (res["hf_home"] or "").startswith("/workspace")}

    failed = [k for k, v in res["checks"].items() if v.get("ok") is False]
    failed += [f"import {k}" for k, v in res["imports"].items() if not v["ok"]]
    res["failed"] = failed
    for k, v in res["checks"].items():
        print(f"{'PASS' if v.get('ok') else 'SKIP' if v.get('ok') is None else 'FAIL'}  {k:22s} "
              f"{ {a: b for a, b in v.items() if a != 'ok'} }")
    for k, v in res["imports"].items():
        print(f"{'PASS' if v['ok'] else 'FAIL'}  import {k:15s} {v.get('version', v.get('error'))}")
    print("ALL CHECKS PASSED" if not failed else f"FAILED: {failed}")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
