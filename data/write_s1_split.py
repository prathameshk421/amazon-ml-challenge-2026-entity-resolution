#!/usr/bin/env python3
"""Split train S1 (and its ground truth) 80/20 next to the original TSVs.

S2 and S3 files are not copied or filtered. Blocking still uses the full pool.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter

TRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "student_resource", "dataset", "train")
SEED = 42
VAL_FRAC = 0.20
N_S1 = 2_206_821
N_TRAIN = 1_765_458
N_VAL = 441_363

S1_IN = "train_source1.tsv"
GT_IN = "train_ground_truth.tsv"
S1_TRAIN_OUT = "train_source1_train.tsv"
S1_VAL_OUT = "train_source1_validation.tsv"
GT_TRAIN_OUT = "train_ground_truth_train.tsv"
GT_VAL_OUT = "train_ground_truth_validation.tsv"


def _rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline()
        if not header.endswith("\n"):
            header += "\n"
        for line in f:
            if not line.endswith("\n"):
                line += "\n"
            yield line


def assign_ids():
    singleton = {}
    for line in _rows(os.path.join(TRAIN_DIR, GT_IN)):
        eid, _, rest = line.partition("\t")
        singleton[eid] = rest.strip("\n") == ""
    strata = {}
    n = 0
    for line in _rows(os.path.join(TRAIN_DIR, S1_IN)):
        fields = line.rstrip("\n").split("\t")
        eid, country = fields[0], fields[3]
        strata.setdefault((country, singleton[eid]), []).append(eid)
        n += 1
    if n != N_S1 or len(singleton) != N_S1:
        raise SystemExit("S1/GT size mismatch: s1=%d gt=%d" % (n, len(singleton)))
    rng = random.Random(SEED)
    val_ids, train_ids = [], []
    counts = {}
    for key in sorted(strata, key=str):
        ids = strata[key]
        rng.shuffle(ids)
        k = int(round(len(ids) * VAL_FRAC))
        val_ids.extend(ids[:k])
        train_ids.extend(ids[k:])
        counts["%s|%s" % (key[0], "singleton" if key[1] else "matched")] = {
            "n": len(ids), "validation": k, "train": len(ids) - k,
        }
    rng.shuffle(val_ids)
    rng.shuffle(train_ids)
    if len(train_ids) != N_TRAIN or len(val_ids) != N_VAL:
        raise SystemExit("unexpected sizes train=%d val=%d" % (len(train_ids), len(val_ids)))
    return set(train_ids), set(val_ids), counts


def write_subset(src_name, train_name, val_name, train_set, val_set):
    src = os.path.join(TRAIN_DIR, src_name)
    paths = {
        "train": open(os.path.join(TRAIN_DIR, train_name), "w", encoding="utf-8", newline=""),
        "validation": open(os.path.join(TRAIN_DIR, val_name), "w", encoding="utf-8", newline=""),
    }
    n_train = n_val = 0
    try:
        with open(src, encoding="utf-8", newline="") as f:
            header = f.readline()
            if not header.endswith("\n"):
                header += "\n"
            paths["train"].write(header)
            paths["validation"].write(header)
            for line in f:
                if not line.endswith("\n"):
                    line += "\n"
                eid = line.split("\t", 1)[0]
                if eid in train_set:
                    paths["train"].write(line)
                    n_train += 1
                elif eid in val_set:
                    paths["validation"].write(line)
                    n_val += 1
                else:
                    raise SystemExit("id not in train or validation: %s" % eid)
    finally:
        for fh in paths.values():
            fh.close()
    return n_train, n_val


def main():
    train_set, val_set, counts = assign_ids()
    overlap = train_set & val_set
    if overlap:
        raise SystemExit("train/validation overlap: %d" % len(overlap))
    s1_tr, s1_va = write_subset(S1_IN, S1_TRAIN_OUT, S1_VAL_OUT, train_set, val_set)
    gt_tr, gt_va = write_subset(GT_IN, GT_TRAIN_OUT, GT_VAL_OUT, train_set, val_set)
    if (s1_tr, s1_va, gt_tr, gt_va) != (N_TRAIN, N_VAL, N_TRAIN, N_VAL):
        raise SystemExit("row counts %s" % ((s1_tr, s1_va, gt_tr, gt_va),))

    # singleton rates from the GT files we just wrote
    def singleton_rate(path):
        n = n_sing = 0
        for line in _rows(path):
            n += 1
            n_sing += line.partition("\t")[2].strip("\n") == ""
        return n, n_sing, n_sing / n

    def country_counts(path):
        c = Counter()
        for line in _rows(path):
            c[line.rstrip("\n").split("\t")[3]] += 1
        return dict(c)

    tr_n, tr_sing, tr_rate = singleton_rate(os.path.join(TRAIN_DIR, GT_TRAIN_OUT))
    va_n, va_sing, va_rate = singleton_rate(os.path.join(TRAIN_DIR, GT_VAL_OUT))
    manifest = {
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "stratify": "country x singleton vs matched",
        "n_s1": N_S1,
        "n_train_s1": N_TRAIN,
        "n_validation_s1": N_VAL,
        "strata": counts,
        "singleton": {
            "train": {"n": tr_n, "n_singleton": tr_sing, "rate": tr_rate},
            "validation": {"n": va_n, "n_singleton": va_sing, "rate": va_rate},
        },
        "country": {
            "train": country_counts(os.path.join(TRAIN_DIR, S1_TRAIN_OUT)),
            "validation": country_counts(os.path.join(TRAIN_DIR, S1_VAL_OUT)),
        },
        "files": {
            "s1_train": S1_TRAIN_OUT,
            "s1_validation": S1_VAL_OUT,
            "ground_truth_train": GT_TRAIN_OUT,
            "ground_truth_validation": GT_VAL_OUT,
            "s2_unchanged": "train_source2.tsv",
            "s3_unchanged": "train_source3.tsv",
        },
        "rule": "S2 and S3 stay whole. Validation S1 ground truth is for scoring only.",
    }
    man_path = os.path.join(TRAIN_DIR, "s1_split_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps({k: manifest[k] for k in (
        "n_train_s1", "n_validation_s1", "strata", "singleton", "country", "files")}, indent=2))
    print("wrote", man_path)


if __name__ == "__main__":
    main()
