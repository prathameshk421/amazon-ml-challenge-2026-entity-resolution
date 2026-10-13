"""CLI `candidates --set {MATCH_TRAIN,MATCH_VALID,HOLDOUT_TEST,val-S1,TEST} [--limit N]` (plan §7):
stage-1+2 inference (M6+M7) for the named query set, writing `work/candidates/{SET}/part-*.parquet`.
Unscored -- `rerank-eval` is the scored, DEV_B/HOLDOUT_TEST-only command; this one only produces the
candidate file for a matcher (or, for TEST, `write-outputs`) to consume, reusing `rerank.py`'s stage-1+2
+ cut pipeline (`rerank.generate_candidates`) and the same staleness guard as `rerank-eval`.

TEST is the official test split (all test S1, no ground truth) run against the test index; the other
sets are the plan §5.4 train-derived query sets, run against the train index. `--limit N` caps the
number of queries processed (first N in query-set order), for a quick smoke run.
"""

import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, rerank, retrieve, splits

TRAIN_SETS = ("MATCH_TRAIN", "MATCH_VALID", "HOLDOUT_TEST", "val-S1")


def _stage2_model(cfg):
    """Loaded stage-1 model + stage-2 feature order/cut, after the plan §5.8 staleness guard."""
    model = retrieve.load_stage1_model(cfg)
    if model is None:
        raise RuntimeError("candidates requires channels.LEARN == stage1_model")
    meta = io_utils.read_json(io_utils.work_path("rerank", "model_v1_meta.json"))
    cur_stage1_hash = io_utils.config_hash(retrieve.stage1_config(cfg))
    cur_model_hash = io_utils.config_hash(model)
    if meta["stage1_config_hash"] != cur_stage1_hash or meta["stage1_model_hash"] != cur_model_hash:
        raise RuntimeError(
            "work/rerank/model_v1_meta.json is stale: the stage-1 config or model changed since "
            "rerank-train (plan §5.8 staleness guard) -- re-run rerank-build/rerank-train/rerank-cut")
    return model, meta["features"]


def run_candidates(args):
    t0 = time.time()
    set_name = args.set
    lg = io_utils.get_logger(f"candidates_{set_name}")
    cfg = io_utils.load_config()
    model, feat_names = _stage2_model(cfg)
    cut = cfg["rerank"]["cut"]

    if set_name == "TEST":
        split, out_name, collect_stats = "test", "TEST", True
        n = pq.ParquetFile(io_utils.work_path("records", "test_s1.parquet")).metadata.num_rows
        s1_idx = np.arange(n, dtype=np.int32)
        country_raw = pd.read_parquet(io_utils.work_path("records", "test_s1.parquet"),
                                      columns=["country_raw"])["country_raw"].to_numpy()
        io_utils.log_test_usage("stage-1+2 inference over all test S1",
                                "R4: official candidate generation (candidates --set TEST)",
                                str(io_utils.work_path("index", "test", "meta.json")))
    elif set_name in TRAIN_SETS:
        split, collect_stats = "train", False
        out_name = "HOLDOUT_TEST" if set_name == "val-S1" else set_name
        s1_idx = splits.load_query_sets()[out_name]
        country_raw = pd.read_parquet(io_utils.work_path("records", "train_s1.parquet"),
                                      columns=["country_raw"])["country_raw"].to_numpy()[s1_idx]
    else:
        raise ValueError(f"candidates: unknown --set {set_name!r}")

    if args.limit is not None:
        s1_idx, country_raw = s1_idx[:args.limit], country_raw[:args.limit]

    out_dir = io_utils.work_path("candidates", out_name)
    pool_lists, _, _, _ = rerank.generate_candidates(
        split, s1_idx, country_raw, cfg, model, feat_names, cut, out_dir,
        collect_query_stats=collect_stats)

    sizes = np.array([len(p) for p in pool_lists], dtype=np.int64)
    payload = {"set": set_name, "out_dir": out_name, "split": split, "limited": args.limit is not None,
               "n_queries": int(s1_idx.size), "n_empty": int((sizes == 0).sum()),
               "size_mean": float(sizes.mean()) if sizes.size else 0.0,
               "wall_s": round(time.time() - t0, 1), "peak_rss_mb": round(io_utils.peak_rss_mb(), 1)}
    io_utils.write_metrics(f"candidates_{set_name}", payload, int(s1_idx.size), 0, cfg)
    lg.info("candidates %s done in %.1fs: %d queries, mean size %.2f, %d empty", set_name,
            time.time() - t0, s1_idx.size, payload["size_mean"], payload["n_empty"])
