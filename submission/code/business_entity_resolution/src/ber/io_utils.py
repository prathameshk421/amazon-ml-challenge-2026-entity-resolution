"""Shared I/O helpers: config, paths, atomic writes, logging, metrics JSON, worker pools (plan §3.3)."""

import datetime
import hashlib
import json
import logging
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

try:
    import resource  # POSIX only; peak RSS is reported as NaN on Windows
except ImportError:
    resource = None

CODE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = CODE_DIR / "configs"
BLOCKING_JSON = CONFIG_DIR / "blocking.json"
PATHS_JSON = CONFIG_DIR / "paths.json"


# ---------------------------------------------------------------- config and paths

def load_config():
    with open(BLOCKING_JSON, encoding="utf-8") as f:
        return json.load(f)


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(cfg):
    """sha1 of the canonical JSON of blocking.json (plan §3.3)."""
    return hashlib.sha1(canonical_json(cfg).encode("utf-8")).hexdigest()


def apply_tuned(updates, experiment, metrics_file):
    """Write experiment decisions into blocking.json (plan §8: every chosen value gets `tuned_by`).

    updates: {dotted.key.path: value}, e.g. {"cap_frac.NT": 0.005, "abbrev_mining.enabled": False}.
    """
    cfg = load_config()
    for dotted, value in updates.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
        cfg["tuned_by"][dotted] = {"experiment": experiment, "metrics": os.path.relpath(metrics_file),
                                   "value": value}
    atomic_write_text(BLOCKING_JSON, json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    return cfg


def paths():
    """Data/work/metrics roots from paths.json, resolved against the current directory (repo root)."""
    with open(PATHS_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v).resolve() for k, v in raw.items()}


def work_path(*parts):
    p = paths()["work_root"].joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def data_file(split, source):
    """source: 1, 2, 3 or 'gt'."""
    name = f"{split}_ground_truth.tsv" if source == "gt" else f"{split}_source{source}.tsv"
    return paths()["data_root"] / split / name


# ---------------------------------------------------------------- atomic writes

def _tmp(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, path.with_name(path.name + ".tmp")


def atomic_write_text(path, text):
    path, tmp = _tmp(path)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def atomic_save_npy(path, arr):
    import numpy as np
    path, tmp = _tmp(path)
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def atomic_write_parquet(df, path):
    path, tmp = _tmp(path)
    df.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- logging and metrics

def get_logger(command):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = work_path("logs", f"{command}_{ts}.log")
    logger = logging.getLogger(f"ber.{command}")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stderr)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=CODE_DIR, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_metrics(name, payload, n_queries, n_true_pairs, cfg=None):
    """Write artifacts/metrics/<name>.json with the mandatory header fields (plan §3.3)."""
    cfg = load_config() if cfg is None else cfg
    doc = {"n_queries": int(n_queries), "n_true_pairs": int(n_true_pairs), "config": cfg,
           "config_hash": config_hash(cfg), "norm_version": cfg["norm_version"],
           "git_commit": git_commit(),
           "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    doc.update(payload)
    path = paths()["metrics_root"] / f"{name}.json"
    atomic_write_json(path, doc)
    return path


def log_test_usage(what, why, file):
    """Append one line to reports/test_usage_log.md (plan R4)."""
    path = Path("reports/test_usage_log.md").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"- {ts} | what: {what} | why: {why} | file: {file}\n")


def peak_rss_mb():
    if resource is None:
        return float("nan")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KiB on Linux


# ---------------------------------------------------------------- parallelism

def n_workers(cfg):
    """W = min(max_workers, cpu_count - 2), at least 1 (plan §3.3)."""
    return max(1, min(cfg["max_workers"], (os.cpu_count() or 1) - 2))


def spawn_pool(n, initializer=None, initargs=()):
    """Spawn-context pool; workers must open big arrays with np.load(..., mmap_mode='r')."""
    os.environ["OMP_NUM_THREADS"] = "1"
    return multiprocessing.get_context("spawn").Pool(n, initializer=initializer, initargs=initargs)
