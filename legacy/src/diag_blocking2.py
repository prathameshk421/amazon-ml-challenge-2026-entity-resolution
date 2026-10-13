"""Bucketing experiment: current channels + NEW PAIR and ROM channels (aux.npz).
Same 10K fit queries (seed 123) as diag_blocking.py. No training: fixed formulas.
  F1      : current channels, exact weight 10 (best hand formula in diag_blocking)
  F5      : F1 + PAIR idf (df<=PCAP) + ROM token idf (df<=20000) + ROM trigram idf (df<=8000)
  F6      : F5 with PAIR weight 2
Reports reachability, pair/entity recall@K overall and by segment, union size.
Run with PYTHONHASHSEED=0 (hash keys must match build_aux_index.py).
Writes artifacts/metrics/diag_blocking2.json.
"""

import csv
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name, normalize_address
from block_query2 import Q
from diag_blocking import union_scores, formulas
from diag_scripts import classify
from build_aux_index import pair_keys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
IDX = os.path.join(ROOT, "artifacts/index")
MET = os.path.join(ROOT, "artifacts/metrics")
POOL_N = 10320219
PCAP = int(os.environ.get("PCAP", "20000"))
N_ROM = 752869  # romanized rows reported by build_aux_index.py
KS = [25, 50, 100, 150, 250, 500, 1000, 2000]
assert os.environ.get("PYTHONHASHSEED") == "0"

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


class Aux:
    def __init__(self):
        z = np.load(os.path.join(IDX, "aux.npz"))
        self.keys = z["keys"]
        self.indptr = z["indptr"]
        self.data = z["data"]

    def get(self, h):
        h = np.uint64(h & 0xFFFFFFFFFFFFFFFF)
        j = int(np.searchsorted(self.keys, h))
        if j < len(self.keys) and self.keys[j] == h:
            return self.data[self.indptr[j]:self.indptr[j + 1]]
        return None


def aux_scores(aux, nn):
    """-> list of (postings, weight, channel) for PAIR(5), ROMT(6), ROMG(7)."""
    out = []
    for h in set(pair_keys(nn)):
        a = aux.get(h)
        if a is not None and len(a) <= PCAP:
            out.append((a, np.log((POOL_N + 1) / (len(a) + 1)), 5))
    # ROM df is measured on the romanized sub-pool only; rescale to full-pool
    # frequency so caps and idf mean the same as for the Latin channels
    sc = POOL_N / N_ROM
    for t in set(nn.split()):
        a = aux.get(hash("RT:" + t))
        if a is not None and len(a) * sc <= 20000:
            out.append((a, np.log((POOL_N + 1) / (len(a) * sc + 1)), 6))
    s = " " + nn + " "
    for g in set(s[i:i + 3] for i in range(len(s) - 2)):
        a = aux.get(hash("RG:" + g))
        if a is not None and len(a) * sc <= 8000:
            out.append((a, np.log((POOL_N + 1) / (len(a) * sc + 1)), 7))
    return out


def main():
    rng = random.Random(123)
    split = json.load(open(os.path.join(ROOT, "artifacts/split.json")))
    qids = rng.sample(split["fit_ids"], 10000)
    want = set(qids)
    recs, chain = {}, Counter()
    with open(os.path.join(TR, "train_source1.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            nn = normalize_name(row["business_name"])
            chain[nn] += 1
            if row["entity_id"] in want:
                recs[row["entity_id"]] = (nn, normalize_address(row["business_address"]),
                                          row["country"])
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
    aux = Aux()
    log("indexes loaded")
    rows_out = []
    usize = {"F1": [], "F5": []}
    qt = 0.0
    for qi, e in enumerate(qids):
        nn, na, country = recs[e]
        ts = tro[e]
        t1 = time.time()
        res = union_scores(q, nn, na)
        extra = aux_scores(aux, nn)
        # merge: build unified candidate set
        if res is not None:
            u0, S0, qs = res
            base = formulas(S0, qs)["F1_exact10"]
        else:
            u0 = np.zeros(0, dtype=np.uint32)
            base = np.zeros(0)
        P = [u0] + [a for a, _, _ in extra]
        W = [np.zeros(len(u0))] + [np.full(len(a), w) for a, w, _ in extra]
        C = [np.zeros(len(u0), dtype=np.uint8)] + [np.full(len(a), c, dtype=np.uint8) for a, _, c in extra]
        Pall = np.concatenate(P).astype(np.uint32)
        if len(Pall) == 0:
            for x in ts:
                rows_out.append({"s1": e, "reach0": 0, "reach5": 0})
            continue
        Wall = np.concatenate(W)
        Call = np.concatenate(C)
        u, inv = np.unique(Pall, return_inverse=True)
        f1 = np.zeros(len(u))
        f1[np.searchsorted(u, u0)] = base
        ch = {}
        for c in (5, 6, 7):
            m = Call == c
            ch[c] = np.bincount(inv[m], weights=Wall[m], minlength=len(u)) if m.any() else np.zeros(len(u))
        f5 = f1 + ch[5] + ch[6] + ch[7]
        f6 = f1 + ch[5]                      # PAIR only
        f7 = f1 + ch[6] + ch[7]              # ROM only
        f8 = f1 + 0.3 * ch[5] + ch[6] + ch[7]  # both, pair down-weighted
        in0 = np.zeros(len(u), dtype=bool)
        in0[np.searchsorted(u, u0)] = True
        qt += time.time() - t1
        usize["F1"].append(len(u0))
        usize["F5"].append(len(u))
        if not ts:
            continue
        pos = np.minimum(np.searchsorted(u, ts), len(u) - 1)
        hit = u[pos] == np.asarray(ts)
        for k, x in enumerate(ts):
            pn = bytes(nblob[noff[x]:noff[x + 1]]).decode("utf-8", "replace")
            pa = bytes(ablob[aoff[x]:aoff[x + 1]]).decode("utf-8", "replace")
            d = {"s1": e, "country": country, "chain": chain[nn],
                 "name_cls": classify(pn)[0], "addr_cls": classify(pa)[0]}
            if hit[k]:
                j = int(pos[k])
                d["reach0"] = int(in0[j])
                d["reach5"] = 1
                d["v_pair"] = float(ch[5][j])
                d["v_rom"] = float(ch[6][j] + ch[7][j])
                for nm, sc in (("F1", f1), ("F5", f5), ("F6", f6), ("F7", f7), ("F8", f8)):
                    if nm == "F1" and not in0[j]:
                        d["rank_F1"] = np.nan
                    else:
                        d["rank_" + nm] = int((sc >= sc[j]).sum())
            else:
                d["reach0"] = 0
                d["reach5"] = 0
            rows_out.append(d)
        if (qi + 1) % 2000 == 0:
            log("q %d  union F1 %.0f  F5 %.0f  %.1f ms/q" % (qi + 1, np.mean(usize["F1"]),
                np.mean(usize["F5"]), qt / (qi + 1) * 1000))
    df = pd.DataFrame(rows_out)
    df.to_parquet(os.path.join(ROOT, "artifacts/diag_pairs2.parquet"))
    def rec(sub, col):
        rk = sub[col].fillna(10 ** 12)
        ent = sub.assign(h=rk).groupby("s1")["h"].min()
        return {"pair": {str(K): round(float((rk <= K).mean()), 4) for K in KS},
                "entity": {str(K): round(float((ent <= K).mean()), 4) for K in KS}}
    out = {"PCAP": PCAP, "n_true_pairs": int(len(df)),
           "union_mean": {k: float(np.mean(v)) for k, v in usize.items()},
           "union_p99": {k: float(np.percentile(v, 99)) for k, v in usize.items()},
           "ms_per_query": qt / len(qids) * 1000,
           "reach_before": round(float(df["reach0"].mean()), 4),
           "reach_after": round(float(df["reach5"].mean()), 4),
           "all": {c: rec(df, "rank_" + c) for c in ("F1", "F5", "F6", "F7", "F8")},
           "segments": {}}
    df["chain_b"] = pd.cut(df["chain"], [0, 1, 9, 10 ** 9], labels=["1", "2-9", "10+"]).astype(str)
    for col in ["chain_b", "country", "name_cls", "addr_cls"]:
        for v, sub in df.groupby(col):
            if len(sub) >= 50:
                out["segments"]["%s=%s" % (col, v)] = {
                    "n": int(len(sub)), "reach_before": round(float(sub["reach0"].mean()), 4),
                    "reach_after": round(float(sub["reach5"].mean()), 4),
                    "F1_pair": rec(sub, "rank_F1")["pair"], "F5_pair": rec(sub, "rank_F5")["pair"],
                    "F7_pair": rec(sub, "rank_F7")["pair"], "F8_pair": rec(sub, "rank_F8")["pair"]}
    json.dump(out, open(os.path.join(MET, "diag_blocking2.json"), "w"), indent=1)
    log("WROTE diag_blocking2.json")


if __name__ == "__main__":
    main()
