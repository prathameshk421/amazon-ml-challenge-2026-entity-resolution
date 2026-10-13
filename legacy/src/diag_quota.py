"""Bucketing experiment: per-channel QUOTA union (no single additive score, no training).
Each channel ranks the union by its own score and contributes its top-q; the candidate
set is the union. Channels:
  NAME : 10*EX + NT + TR + 0.2*PH   (current Latin name channels)
  ADDR : AD
  BOTH : F1_exact10 (name + addr sum, current best single formula)
  PAIR : token-pair idf (aux.npz, df<=20000)
  ROM  : romanized-name token + trigram idf (df rescaled to full pool)
Same 10K fit queries (seed 123). Reports pair/entity recall and mean/p99 unique set size
per quota config. Run with PYTHONHASHSEED=0. Writes artifacts/metrics/diag_quota.json.
"""

import csv
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name, normalize_address
from block_query2 import Q
from diag_blocking import union_scores
from diag_blocking2 import Aux, aux_scores

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
IDX = os.path.join(ROOT, "artifacts/index")
MET = os.path.join(ROOT, "artifacts/metrics")
assert os.environ.get("PYTHONHASHSEED") == "0"

CONFIGS = {
    # name: {channel: quota}
    "BOTH150 (ref)": {"BOTH": 150},
    "BOTH500 (ref)": {"BOTH": 500},
    "BOTH2000 (ref)": {"BOTH": 2000},
    "q50x5": {"BOTH": 50, "NAME": 50, "ADDR": 50, "PAIR": 50, "ROM": 50},
    "q100x5": {"BOTH": 100, "NAME": 100, "ADDR": 100, "PAIR": 100, "ROM": 100},
    "q200x5": {"BOTH": 200, "NAME": 200, "ADDR": 200, "PAIR": 200, "ROM": 200},
    "q300x5": {"BOTH": 300, "NAME": 300, "ADDR": 300, "PAIR": 300, "ROM": 300},
    "q500x5": {"BOTH": 500, "NAME": 500, "ADDR": 500, "PAIR": 500, "ROM": 500},
    "B300+N100+A100+P100+R50": {"BOTH": 300, "NAME": 100, "ADDR": 100, "PAIR": 100, "ROM": 50},
    "B500+N200+A200+P200+R100": {"BOTH": 500, "NAME": 200, "ADDR": 200, "PAIR": 200, "ROM": 100},
}

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


def topq(score, q):
    nz = np.flatnonzero(score > 0)
    if len(nz) <= q:
        return nz
    return nz[np.argpartition(-score[nz], q - 1)[:q]]


def main():
    rng = random.Random(123)
    split = json.load(open(os.path.join(ROOT, "artifacts/split.json")))
    qids = rng.sample(split["fit_ids"], 10000)
    want = set(qids)
    recs = {}
    with open(os.path.join(TR, "train_source1.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in want:
                recs[row["entity_id"]] = (normalize_name(row["business_name"]),
                                          normalize_address(row["business_address"]))
    truth = {}
    with open(os.path.join(TR, "train_ground_truth.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["source1_entity_id"] in want:
                m = row["matched_entity_ids"]
                truth[row["source1_entity_id"]] = m.split(",") if m else []
    src = np.load(os.path.join(IDX, "pool.src.npy"))
    num = np.load(os.path.join(IDX, "pool.num.npy"))
    key = (src.astype(np.int64) << 32) | num
    order = np.argsort(key)
    skey = key[order]
    tro = {}
    for e, ids in truth.items():
        rows = []
        for x in ids:
            s = 0 if x.startswith("S2-") else 1
            k = (np.int64(s) << np.int64(32)) | np.int64(int(x.split("-", 1)[1]))
            j = int(np.searchsorted(skey, k))
            if j < len(skey) and skey[j] == k:
                rows.append(int(order[j]))
        tro[e] = rows
    del key, order, skey
    q = Q()
    aux = Aux()
    log("indexes loaded")
    hit = {c: 0 for c in CONFIGS}
    ehit = {c: 0 for c in CONFIGS}
    sizes = {c: [] for c in CONFIGS}
    tot = etot = 0
    for qi, e in enumerate(qids):
        nn, na = recs[e]
        res = union_scores(q, nn, na)
        extra = aux_scores(aux, nn)
        if res is not None:
            u0, S0, _ = res
        else:
            u0, S0 = np.zeros(0, dtype=np.uint32), np.zeros((5, 0))
        P = [u0] + [a for a, _, _ in extra]
        Pall = np.concatenate(P).astype(np.uint32) if P else np.zeros(0, dtype=np.uint32)
        ts = set(tro[e])
        if len(Pall) == 0:
            for c in CONFIGS:
                sizes[c].append(0)
            if ts:
                tot += len(ts)
                etot += 1
            continue
        u, inv = np.unique(Pall, return_inverse=True)
        j0 = np.searchsorted(u, u0)
        name = np.zeros(len(u))
        addr = np.zeros(len(u))
        name[j0] = 10 * S0[0] + S0[1] + S0[2] + 0.2 * S0[4]
        addr[j0] = S0[3]
        W = np.concatenate([np.zeros(len(u0))] + [np.full(len(a), w) for a, w, _ in extra])
        C = np.concatenate([np.zeros(len(u0), dtype=np.uint8)] +
                           [np.full(len(a), c, dtype=np.uint8) for a, _, c in extra])
        def chan(cs):
            m = np.isin(C, cs)
            return np.bincount(inv[m], weights=W[m], minlength=len(u)) if m.any() else np.zeros(len(u))
        sc = {"NAME": name, "ADDR": addr, "BOTH": name + addr, "PAIR": chan([5]), "ROM": chan([6, 7])}
        cache = {}
        for c, quotas in CONFIGS.items():
            sel = set()
            for ch, qq in quotas.items():
                k = (ch, qq)
                if k not in cache:
                    cache[k] = set(u[topq(sc[ch], qq)].tolist())
                sel |= cache[k]
            sizes[c].append(len(sel))
            if ts:
                h = len(sel & ts)
                hit[c] += h
                ehit[c] += int(h > 0)
        if ts:
            tot += len(ts)
            etot += 1
        if (qi + 1) % 2000 == 0:
            log("q %d" % (qi + 1))
    out = {"n_true_pairs": tot, "n_entities": etot, "configs": {}}
    for c in CONFIGS:
        out["configs"][c] = {"quotas": CONFIGS[c], "pair_recall": round(hit[c] / tot, 4),
                             "entity_recall": round(ehit[c] / etot, 4),
                             "mean_size": round(float(np.mean(sizes[c])), 1),
                             "p99_size": float(np.percentile(sizes[c], 99))}
    json.dump(out, open(os.path.join(MET, "diag_quota.json"), "w"), indent=1)
    for c, v in out["configs"].items():
        log("%-28s pair %.4f ent %.4f size %.0f p99 %.0f" % (c, v["pair_recall"], v["entity_recall"],
                                                            v["mean_size"], v["p99_size"]))
    log("WROTE diag_quota.json")


if __name__ == "__main__":
    main()
