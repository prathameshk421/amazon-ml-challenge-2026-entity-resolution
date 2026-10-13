"""Critique job F: single-thread throughput of the stage-2 hot path (plan §5.8) on this Mac:
rapidfuzz cdist (4 name scorers + 2 address scorers) for 1 query x N candidates, and LightGBM predict
(33 features, num_leaves 63) per row. Projects CPU-hours for the planned query volumes.
Output: plan/critique/out/f_throughput.json
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "plan", "critique", "out")
os.environ["OMP_NUM_THREADS"] = "1"
names, addrs = [], []
with open(os.path.join(ROOT, "student_resource", "dataset", "train", "train_source3.tsv"), encoding="utf-8", newline="") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 200000:
            break
        _, n, a, _ = line.rstrip("\n").split("\t")
        names.append(n.lower())
        addrs.append(a.lower())
R = {}
rng = np.random.default_rng(0)
for N in (100, 700):
    t = time.time()
    Q = 200
    for q in range(Q):
        idx = rng.integers(0, len(names), N)
        cn = [names[j] for j in idx]
        ca = [addrs[j] for j in idx]
        qn, qa = names[q], addrs[q]
        for sc in (fuzz.token_set_ratio, fuzz.ratio, fuzz.partial_ratio):
            process.cdist([qn], cn, scorer=sc, workers=1)
        process.cdist([qn.replace(" ", "")], [c.replace(" ", "") for c in cn], scorer=JaroWinkler.normalized_similarity, workers=1)
        for sc in (fuzz.token_set_ratio, fuzz.token_sort_ratio):
            process.cdist([qa], ca, scorer=sc, workers=1)
    R[f"rapidfuzz_ms_per_query_N{N}"] = (time.time() - t) / Q * 1000

X = rng.random((200000, 33), dtype=np.float32)
y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(0, 0.3, 200000) > 1.0).astype(int)
bst = lgb.train({"objective": "binary", "num_leaves": 63, "learning_rate": 0.05, "min_data_in_leaf": 200,
                 "verbose": -1, "num_threads": 1, "seed": 1}, lgb.Dataset(X, y), num_boost_round=500)
Xp = rng.random((200000, 33), dtype=np.float32)
t = time.time()
bst.predict(Xp, num_threads=1)
R["lgb_500trees_us_per_row"] = (time.time() - t) / 200000 * 1e6
t = time.time()
for k in range(200):
    bst.predict(Xp[k * 700:(k + 1) * 700], num_threads=1)
R["lgb_500trees_ms_per_700row_call"] = (time.time() - t) / 200 * 1000
# projections: planned query volumes (plan §5.4/§7): TEST 1,732,544 + MATCH_POOL (fit - 90,000) + VAL
fit = 2206821 - round(2206821 * 0.2)
vol = {"TEST": 1732544, "MATCH_POOL": fit - 90000, "VAL": round(2206821 * 0.2), "RR+DEV": 100000}
R["query_volumes"] = vol
per_q_ms = R["rapidfuzz_ms_per_query_N700"] + R["lgb_500trees_ms_per_700row_call"]
R["stage2_ms_per_query_at_700_cands_excl_stage1_and_feature_glue"] = per_q_ms
R["stage2_cpu_hours_total"] = sum(vol.values()) * per_q_ms / 3.6e6
R["candidate_pairs_tsv_bytes_at_mean"] = {str(m): 1732544 * (m * 13 + 14) for m in (50, 80, 100)}
json.dump(R, open(os.path.join(OUT, "f_throughput.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
