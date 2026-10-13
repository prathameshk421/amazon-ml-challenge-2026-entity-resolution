"""Bucketing diagnostic C: does a cheap similarity RE-RANK of a wide shortlist fix the
ranking loss (reachable but ranked > K)? No model is trained: fixed hand formulas only.

Queries: first NQ of the same 10K fit-part queries (seed 123). Shortlist = top-N1 of the
F1_exact10 union score (diag_blocking.py). For each shortlisted candidate compute, with
rapidfuzz (vectorized cdist):
  n_ts : token_set_ratio(normalized S1 name, normalized pool name) / 100, where a
         non-Latin pool name is first romanized (legacy/src/translit.py) -> 'n_ts_rom'
  n_r  : ratio on the same strings (order-sensitive char similarity)
  a_ts : token_set_ratio on normalized addresses / 100
  a_pr : partial_token_set? no -> token_sort_ratio / 100
Rerank formulas (no fitting):
  R0: stage-1 score order (reference)
  R1: n_ts + a_ts
  R2: max(n_ts, a_ts) + 0.5*min(n_ts, a_ts)
  R3: n_ts_rom + a_ts            (romanized names)
  R4: 0.5*(n_ts_rom + n_r_rom) + a_ts + 0.1*stage1_rank_score
Writes artifacts/metrics/diag_rerank.json.
"""

import csv
import json
import os
import random
import sys
import time

import numpy as np
from rapidfuzz import fuzz, process

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name, normalize_address
from block_query2 import Q
from diag_blocking import union_scores, formulas
from translit import romanize_words_schwa, has_indic

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
IDX = os.path.join(ROOT, "artifacts/index")
MET = os.path.join(ROOT, "artifacts/metrics")
NQ = int(os.environ.get("NQ", "2000"))
N1 = int(os.environ.get("N1", "2000"))
KS = [10, 25, 50, 100, 150, 250, 500, 1000, 2000]

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


def main():
    rng = random.Random(123)
    split = json.load(open(os.path.join(ROOT, "artifacts/split.json")))
    qids = rng.sample(split["fit_ids"], 10000)[:NQ]
    want = set(qids)
    recs = {}
    with open(os.path.join(TR, "train_source1.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in want:
                recs[row["entity_id"]] = (row["business_name"], row["business_address"])
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
    nblob = np.memmap(os.path.join(IDX, "pool.names.bin"), mode="r", dtype=np.uint8)
    noff = np.load(os.path.join(IDX, "pool.names.off.npy"), mmap_mode="r")
    ablob = np.memmap(os.path.join(IDX, "pool.addrs.bin"), mode="r", dtype=np.uint8)
    aoff = np.load(os.path.join(IDX, "pool.addrs.off.npy"), mmap_mode="r")
    q = Q()
    log("index loaded")
    names = ["R0_stage1", "R1_nts_ats", "R2_maxmin", "R3_rom_nts_ats", "R4_rom_blend"]
    hits = {r: {K: 0 for K in KS} for r in names}
    ehits = {r: {K: 0 for K in KS} for r in names}
    tot = etot = 0
    t_sim = 0.0
    for qi, e in enumerate(qids):
        ts = set(tro[e])
        if not ts:
            continue
        name, addr = recs[e]
        nn = normalize_name(name)
        na = normalize_address(addr)
        res = union_scores(q, nn, na)
        tot += len(ts)
        etot += 1
        if res is None:
            continue
        u, S, qs = res
        s1 = formulas(S, qs)["F1_exact10"]
        if len(u) > N1:
            head = np.argpartition(-s1, N1 - 1)[:N1]
        else:
            head = np.arange(len(u))
        head = head[np.argsort(-s1[head], kind="stable")]
        rows = u[head].tolist()
        t1 = time.time()
        pn_raw = [bytes(nblob[noff[x]:noff[x + 1]]).decode("utf-8", "replace") for x in rows]
        pa_raw = [bytes(ablob[aoff[x]:aoff[x + 1]]).decode("utf-8", "replace") for x in rows]
        pn = [normalize_name(s) for s in pn_raw]
        pnr = [normalize_name(romanize_words_schwa(s)) if has_indic(s) else p
               for s, p in zip(pn_raw, pn)]
        pa = [normalize_address(s) for s in pa_raw]
        n_ts = process.cdist([nn], pn, scorer=fuzz.token_set_ratio, workers=1)[0] / 100.0
        n_tsr = process.cdist([nn], pnr, scorer=fuzz.token_set_ratio, workers=1)[0] / 100.0
        n_rr = process.cdist([nn], pnr, scorer=fuzz.ratio, workers=1)[0] / 100.0
        a_ts = process.cdist([na], pa, scorer=fuzz.token_set_ratio, workers=1)[0] / 100.0
        t_sim += time.time() - t1
        rank_score = 1.0 - np.arange(len(rows)) / max(len(rows), 1)
        sc = {"R0_stage1": rank_score,
              "R1_nts_ats": n_ts + a_ts + 1e-6 * rank_score,
              "R2_maxmin": np.maximum(n_ts, a_ts) + 0.5 * np.minimum(n_ts, a_ts) + 1e-6 * rank_score,
              "R3_rom_nts_ats": n_tsr + a_ts + 1e-6 * rank_score,
              "R4_rom_blend": 0.5 * (n_tsr + n_rr) + a_ts + 0.1 * rank_score}
        pos = {x: i for i, x in enumerate(rows)}
        for r, v in sc.items():
            o = np.argsort(-v, kind="stable")
            rk = np.empty(len(o), dtype=np.int64)
            rk[o] = np.arange(1, len(o) + 1)
            tr = [rk[pos[x]] for x in ts if x in pos]
            for K in KS:
                h = sum(1 for t in tr if t <= K)
                hits[r][K] += h
                ehits[r][K] += int(h > 0)
        if (qi + 1) % 250 == 0:
            log("q %d/%d sim %.1f ms/q" % (qi + 1, NQ, t_sim / (qi + 1) * 1000))
    out = {"n_queries": NQ, "N1": N1, "n_true_pairs": tot, "n_entities": etot,
           "sim_ms_per_query": t_sim / NQ * 1000,
           "pair_recall": {r: {str(K): round(hits[r][K] / tot, 4) for K in KS} for r in names},
           "entity_recall": {r: {str(K): round(ehits[r][K] / etot, 4) for K in KS} for r in names}}
    json.dump(out, open(os.path.join(MET, "diag_rerank.json"), "w"), indent=1)
    log("WROTE diag_rerank.json")


if __name__ == "__main__":
    main()
