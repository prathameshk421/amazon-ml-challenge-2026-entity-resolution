"""Seeded entity lists for candidate gen: fit-train-sample (60K stratified) + val (full).
Writes artifacts/fit_train_sample.json + artifacts/val_list.json (id lists only).
Run: .venv/bin/python legacy/src/make_gen_lists.py
"""

import csv
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")

rng = random.Random(2026)
split = json.load(open(os.path.join(ROOT, "artifacts/split.json")))
fit, val = split["fit_ids"], split["val_ids"]
fitset = set(fit)
sing = {}
with open(os.path.join(TR, "train_ground_truth.tsv"), newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["source1_entity_id"] in fitset and row["matched_entity_ids"] == "":
            sing[row["source1_entity_id"]] = True
co = {}
with open(os.path.join(TR, "train_source1.tsv"), newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["entity_id"] in fitset:
            co[row["entity_id"]] = row["country"]
strata = {}
for e in fit:
    strata.setdefault((co[e], sing.get(e, False)), []).append(e)
sample = []
for k, ids in strata.items():
    k_n = max(1, int(round(len(ids) * 60000 / len(fit))))
    sample.extend(rng.sample(sorted(ids), k_n))
rng.shuffle(sample)
json.dump({"ids": sample, "seed": 2026, "note": "60K stratified fit sample for matcher training"},
          open(os.path.join(ROOT, "artifacts/fit_train_sample.json"), "w"))
json.dump({"ids": val, "note": "full val (441363)"},
          open(os.path.join(ROOT, "artifacts/val_list.json"), "w"))
print("fit_sample=%d val=%d" % (len(sample), len(val)))
