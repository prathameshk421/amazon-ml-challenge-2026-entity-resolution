"""Train S1 -> train-S1 / val-S1 (80/20, country x singleton), then work slices inside train-S1.

Canonical TSVs: data/student_resource/dataset/train/train_source1_{train,validation}.tsv
S2 and S3 are never subsetted. val-S1 ground truth is for scoring only.
"""

import json
import random
from collections import Counter

import numpy as np
import pandas as pd

from ber import io_utils

SPLIT_SEED = 42
PERM_SEED = 20260925
VAL_S1_SIZE = 441_363  # round(0.2) per country x singleton stratum
# (name, start, stop) slices of the train-S1 permutation; stop None = to the end
PERM_SLICES = [("DEV_A", 0, 10_000), ("DEV_B", 10_000, 20_000), ("S1R_TRAIN", 20_000, 40_000),
               ("RR_TRAIN", 40_000, 90_000), ("RR_CAL", 90_000, 110_000),
               ("MATCH_TRAIN", 110_000, 290_000), ("MATCH_VALID", 290_000, 310_000),
               ("DEV_REST", 310_000, None)]
DISJOINT_SETS = [name for name, _, _ in PERM_SLICES] + ["HOLDOUT_TEST"]
VAL_SET_NAMES = ("HOLDOUT_TEST", "val-S1")


def s1_train_val_ids(entity_id, country, nmatch, seed=SPLIT_SEED, val_frac=0.20):
    """Same assignment as data/write_s1_split.py (file order, Random seed 42)."""
    strata = {}
    for eid, ctry, n in zip(entity_id, country, nmatch):
        strata.setdefault((str(ctry), int(n) == 0), []).append(eid)
    rng = random.Random(seed)
    val_ids, train_ids = [], []
    for key in sorted(strata, key=str):
        ids = strata[key]
        rng.shuffle(ids)
        k = int(round(len(ids) * val_frac))
        val_ids.extend(ids[:k])
        train_ids.extend(ids[k:])
    rng.shuffle(val_ids)
    rng.shuffle(train_ids)
    return train_ids, val_ids


def _query_sets_path():
    return io_utils.work_path("splits", "query_sets.json")


def build_sets(entity_id, country, nmatch):
    """entity_id / country (str) / nmatch aligned by s1_idx -> {SET: sorted s1_idx int32 array}."""
    s1_idx = pd.Series(np.arange(len(entity_id), dtype=np.int32), index=entity_id)
    train_ids, val_ids = s1_train_val_ids(entity_id, country, nmatch)
    if len(val_ids) != VAL_S1_SIZE:
        raise ValueError(f"val-S1 has {len(val_ids)} entities, expected {VAL_S1_SIZE}")
    perm = np.random.default_rng(PERM_SEED).permutation(sorted(train_ids))
    named = {name: perm[a:b] for name, a, b in PERM_SLICES}
    named["HOLDOUT_TEST"] = val_ids
    ab = set(named["DEV_A"]) | set(named["DEV_B"])
    named["DICT_SRC"] = [e for e in train_ids if e not in ab]
    sets = {name: np.sort(s1_idx.loc[list(v)].to_numpy(np.int32)) for name, v in named.items()}
    allidx = np.concatenate([sets[n] for n in DISJOINT_SETS])
    if allidx.size != len(entity_id) or np.unique(allidx).size != len(entity_id):
        raise ValueError("query sets are not pairwise disjoint or do not cover all train S1")
    sets["val-S1"] = sets["HOLDOUT_TEST"]
    return sets


def _distribution(values):
    return {str(k): int(v) for k, v in sorted(Counter(values.tolist()).items())}


def _singleton_dist(nmatch, idx):
    n = nmatch[idx]
    n_sing = int((n == 0).sum())
    return {"matched": int(n.size - n_sing), "singleton": n_sing}


def run_splits(args):
    """CLI `splits`: train-S1 / val-S1 then work slices inside train-S1."""
    log = io_utils.get_logger("splits")
    s1 = pd.read_parquet(io_utils.work_path("records", "train_s1.parquet"),
                         columns=["s1_idx", "entity_id", "country_raw"])
    if not np.array_equal(s1["s1_idx"].to_numpy(), np.arange(len(s1))):
        raise ValueError("train_s1.parquet s1_idx is not 0..n-1 in order")
    nmatch = np.load(io_utils.work_path("records", "train_s1_nmatch.npy"))
    sets = build_sets(s1["entity_id"].to_numpy(), s1["country_raw"].to_numpy(), nmatch)
    sizes = {name: int(v.size) for name, v in sets.items()}
    seeds = {"split_seed": SPLIT_SEED, "permutation_seed": PERM_SEED,
             "stratify": "country x singleton vs matched"}
    doc = {"sets": {name: v.tolist() for name, v in sets.items()}, "sizes": sizes, "seeds": seeds}
    path = _query_sets_path()
    io_utils.atomic_write_text(path, json.dumps(doc, separators=(",", ":")) + "\n")
    country = s1["country_raw"].to_numpy()
    per_set = {}
    for name, v in sets.items():
        per_set[name] = {"n": sizes[name], "country": _distribution(country[v]),
                         "singleton": _singleton_dist(nmatch, v),
                         "nmatch": _distribution(nmatch[v])}
    io_utils.write_metrics("splits", {"sizes": sizes, "seeds": seeds, "per_set": per_set},
                           n_queries=len(s1), n_true_pairs=int(nmatch.sum()))
    log.info("splits written: %s %s", path, sizes)


def load_query_sets():
    """{SET: sorted s1_idx int32 array} from splits/query_sets.json."""
    doc = io_utils.read_json(_query_sets_path())
    sets = {name: np.asarray(v, dtype=np.int32) for name, v in doc["sets"].items()}
    if "val-S1" not in sets and "HOLDOUT_TEST" in sets:
        sets["val-S1"] = sets["HOLDOUT_TEST"]
    if "HOLDOUT_TEST" not in sets and "val-S1" in sets:
        sets["HOLDOUT_TEST"] = sets["val-S1"]
    return sets
