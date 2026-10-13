"""Phase 2: blocking evaluation + candidate generation.
Loads artifacts/index/index.npz (mmap) + pool blobs.
EVAL: 10K fit queries (seed 123) vs FULL pool: exact / token(df<=20K) / trigram top-N
  (overlap + idf-weighted, N=10,25,50) / phonetic top-20 / unions. BK-tree k=1..4 on a
  200K pool sample x 2K queries (own BK-tree + rapidfuzz distance).
GEN (after choosing): candidates for fit-train-sample (60K) + full val -> edge-list parquet.
Writes artifacts/metrics/blocking.json + artifacts/candidates/*.parquet.
Run: ./.venv/bin/python legacy/src/block_query.py [--eval-only | --gen-only]
"""

import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name
from bktree import BKTree
import jellyfish

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
IDX = os.path.join(ROOT, "artifacts/index")
MET = os.path.join(ROOT, "artifacts/metrics")
CAND = os.path.join(ROOT, "artifacts/candidates")
os.makedirs(CAND, exist_ok=True)
POOL_N = 10320219
SEED = 123

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


class PoolIndex:
    def __init__(self):
        import pickle
        z = np.load(os.path.join(IDX, "index.npz"))
        self.z = z
        with open(os.path.join(IDX, "exact.pkl"), "rb") as f:
            self.exact = pickle.load(f)
        self.nblob = np.memmap(os.path.join(IDX, "pool.names.bin"), mode="r", dtype=np.uint8)
        self.noff = np.load(os.path.join(IDX, "pool.names.off.npy"), mmap_mode="r")
        self.ablob = np.memmap(os.path.join(IDX, "pool.addrs.bin"), mode="r", dtype=np.uint8)
        self.aoff = np.load(os.path.join(IDX, "pool.addrs.off.npy"), mmap_mode="r")
        self.src = np.load(os.path.join(IDX, "pool.src.npy"), mmap_mode="r")
        self.num = np.load(os.path.join(IDX, "pool.num.npy"), mmap_mode="r")
        self.co = np.load(os.path.join(IDX, "pool.co.npy"), mmap_mode="r")
        self.cmap = json.load(open(os.path.join(IDX, "pool.meta.json")))["cmap"]
        self.tri_df = np.diff(z["tri_indptr"]).astype(np.float64)
        self.tri_idf = np.log((POOL_N + 1) / (self.tri_df + 1))

    def postings(self, prefix, key):
        keys = self.z[prefix + "_keys"]
        j = np.searchsorted(keys, key)
        if j < len(keys) and keys[j] == key:
            s, e = self.z[prefix + "_indptr"][j], self.z[prefix + "_indptr"][j + 1]
            return self.z[prefix + "_data"][s:e]
        return np.zeros(0, dtype=np.uint32)

    def name(self, i):
        return bytes(self.nblob[self.noff[i]:self.noff[i + 1]]).decode("utf-8", "replace")

    def extid(self, i):
        return ("S2-" if self.src[i] == 0 else "S3-") + str(int(self.num[i]))

    @staticmethod
    def trigrams(s):
        s = " " + s + " "
        return set(s[i:i + 3] for i in range(len(s) - 2))


def gather_union(post_list):
    if not post_list:
        return np.zeros(0, dtype=np.uint32)
    return np.unique(np.concatenate(post_list))


def topn_overlap(counts_ids, N):
    # counts_ids: dict rowid->score; deterministic tie-break by rowid
    if not counts_ids:
        return []
    items = sorted(counts_ids.items(), key=lambda kv: (-kv[1], kv[0]))
    return [r for r, _ in items[:N]]


def query_all(pi, nn, tri_df_cap=8000, met_df_cap=50000, tri_top=50, met_top=20):
    """Return dict method -> candidate rowid list."""
    out = {}
    out["exact"] = [int(i) for i in pi.exact.get(nn, np.zeros(0, dtype=np.uint32))]
    toks = nn.split()
    tp = [pi.postings("tok", t) for t in toks]
    tp = [p for p in tp if len(p) > 0]
    out["token"] = [int(i) for i in gather_union(tp)] if tp else []
    # trigram overlap (plain + idf), df-capped
    tgs = PoolIndex.trigrams(nn)
    cnt, wcnt = {}, {}
    for tg in tgs:
        keys = pi.z["tri_keys"]
        j = int(np.searchsorted(keys, tg))
        if j < len(keys) and keys[j] == tg:
            df = float(pi.tri_df[j])
            if df > tri_df_cap:
                continue
            s, e = pi.z["tri_indptr"][j], pi.z["tri_indptr"][j + 1]
            w = float(pi.tri_idf[j])
            for r in pi.z["tri_data"][s:e]:
                r = int(r)
                cnt[r] = cnt.get(r, 0) + 1
                wcnt[r] = wcnt.get(r, 0.0) + w
    out["tri50"] = topn_overlap(cnt, tri_top)
    out["tri_idf50"] = topn_overlap(wcnt, tri_top)
    # phonetic
    mcnt = {}
    for t in set(toks):
        if not t:
            continue
        code = jellyfish.metaphone(t)
        p = pi.postings("met", code)
        if 0 < len(p) <= met_df_cap:
            for r in p:
                r = int(r)
                mcnt[r] = mcnt.get(r, 0) + 1
    out["phon20"] = topn_overlap(mcnt, met_top)
    return out


def load_s1_records(want):
    recs = {}
    with open(os.path.join(TR, "train_source1.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in want:
                recs[row["entity_id"]] = (row["business_name"], row["business_address"])
    return recs


def load_truth(want):
    tr = {}
    with open(os.path.join(TR, "train_ground_truth.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["source1_entity_id"] in want:
                m = row["matched_entity_ids"]
                tr[row["source1_entity_id"]] = m.split(",") if m else []
    return tr


def main():
    import random
    mode = sys.argv[1] if len(sys.argv) > 1 else "--eval-only"
    rng = random.Random(SEED)
    split = json.load(open(os.path.join(ROOT, "artifacts/split.json")))
    fit_ids, val_ids = split["fit_ids"], split["val_ids"]
    # extid -> rowid map (10.3M entries; ~1GB... build from pool arrays in numpy instead)
    log("loading pool id arrays for truth mapping")
    src = np.load(os.path.join(IDX, "pool.src.npy"))
    num = np.load(os.path.join(IDX, "pool.num.npy"))

    def truth_rowids(truth_ext):
        out = {}
        # vectorized per entity via searchsorted on (src,num) int64 key
        key = (src.astype(np.int64) << 32) | num
        order = np.argsort(key)
        skey = key[order]
        for e, ids in truth_ext.items():
            rows = []
            for x in ids:
                s = 0 if x.startswith("S2-") else 1
                k = (np.int64(s) << np.int64(32)) | np.int64(int(x.split("-", 1)[1]))
                j = int(np.searchsorted(skey, k))
                if j < len(skey) and skey[j] == k:
                    rows.append(int(order[j]))
            out[e] = rows
        return out

    if mode in ("--eval-only", "--eval-gen"):
        log("loading index")
        pi = PoolIndex()
        # 10K eval queries from fit
        qids = rng.sample(fit_ids, 10000)
        recs = load_s1_records(set(qids))
        truth_ext = load_truth(set(qids))
        tro = truth_rowids(truth_ext)
        nns = {e: normalize_name(recs[e][0]) for e in qids}
        res = {}
        cfgs = ["exact", "token", "tri10", "tri25", "tri50", "tri_idf10", "tri_idf25",
                "tri_idf50", "phon20", "union_tri", "union_all"]
        agg = {c: {"hit": 0, "tot": 0, "ehit": 0, "etot": 0, "ncand": 0, "t": 0.0} for c in cfgs}
        missed = {}  # true rowid -> (e, extid)
        t_query = 0.0
        for qi, e in enumerate(qids):
            t = tro[e]
            t1 = time.time()
            r = query_all(pi, nns[e])
            t_query += time.time() - t1
            r["tri10"] = r["tri50"][:10]
            r["tri25"] = r["tri50"][:25]
            r["tri_idf10"] = r["tri_idf50"][:10]
            r["tri_idf25"] = r["tri_idf50"][:25]
            r["union_tri"] = sorted(set(r["exact"]) | set(r["token"]) | set(r["tri50"]))
            r["union_all"] = sorted(set(r["union_tri"]) | set(r["tri_idf50"]) | set(r["phon20"]))
            ts = set(t)
            for c in cfgs:
                got = set(r[c]) & ts
                agg[c]["hit"] += len(got)
                agg[c]["tot"] += len(t)
                agg[c]["ncand"] += len(r[c])
                if t:
                    agg[c]["etot"] += 1
                    if got:
                        agg[c]["ehit"] += 1
            if t:
                wall = set(r["union_all"])
                for x in t:
                    if x not in wall:
                        missed[x] = e
            if (qi + 1) % 2000 == 0:
                log("eval %d/10000" % (qi + 1))
        for c in cfgs:
            a = agg[c]
            res[c] = {"pair_recall": a["hit"] / a["tot"], "entity_recall": a["ehit"] / a["etot"],
                      "avg_cands": a["ncand"] / len(qids),
                      "reduction": 1 - (a["ncand"] / len(qids)) / POOL_N,
                      "ms_per_query": t_query / len(qids) * 1000,
                      "timing_note": "shared query_all wall time; tri10/25 are free slices of tri50"}
        res["n_queries"] = len(qids)
        res["n_pos_pairs"] = sum(len(v) for v in tro.values())
        # 10 unrecovered examples with texts
        ex = []
        for x, e in list(missed.items())[:10]:
            ex.append({"s1": e, "s1_name": recs[e][0], "s1_addr": recs[e][1],
                       "missed": pi.extid(x), "missed_name": pi.name(x)})
        res["missed_count"] = len(missed)
        res["missed_examples"] = ex
        json.dump(res, open(os.path.join(MET, "blocking.json"), "w"), indent=1)
        for c in cfgs:
            log("%-10s %s" % (c, json.dumps(res[c] if isinstance(res[c], dict) else res[c])))
        log("missed by all: %d true pairs" % len(missed))

    if mode in ("--eval-only", "--eval-gen"):
        log("BK-tree sample experiment")
        import pickle
        pi_small = None
        # 200K pool sample rowids
        rng2 = random.Random(7)
        pool_sample = sorted(rng2.sample(range(POOL_N), 200000))
        nblob = np.memmap(os.path.join(IDX, "pool.names.bin"), mode="r", dtype=np.uint8)
        noff = np.load(os.path.join(IDX, "pool.names.off.npy"), mmap_mode="r")
        pnames = [bytes(nblob[noff[i]:noff[i + 1]]).decode("utf-8", "replace") for i in pool_sample]
        pnn = [normalize_name(x) for x in pnames]
        log("building BK-tree over 200K sample")
        bk = BKTree()
        t1 = time.time()
        for nm, ri in zip(pnn, pool_sample):
            bk.add(nm, ri)
        log("BK build %.1fs" % (time.time() - t1))
        q2 = rng.sample(fit_ids, 2000)
        recs2 = load_s1_records(set(q2))
        truth2 = load_truth(set(q2))
        # map truth ext->rowid restricted to sample
        in_sample = set(pool_sample)
        src = np.load(os.path.join(IDX, "pool.src.npy"))
        num = np.load(os.path.join(IDX, "pool.num.npy"))
        key = (src.astype(np.int64) << 32) | num
        order = np.argsort(key)
        skey = key[order]
        bkres = {}
        for k in [1, 2, 3, 4]:
            hit = tot = ehit = etot = nc = 0
            t1 = time.time()
            for e in q2:
                nn = normalize_name(recs2[e][0])
                got = set()
                for d, p in bk.query(nn, k):
                    if isinstance(p, list):
                        got.update(p)
                    else:
                        got.add(p)
                tset = set()
                for x in truth2[e]:
                    s = 0 if x.startswith("S2-") else 1
                    kk = (np.int64(s) << np.int64(32)) | np.int64(int(x.split("-", 1)[1]))
                    j = int(np.searchsorted(skey, kk))
                    if j < len(skey) and skey[j] == kk and int(order[j]) in in_sample:
                        tset.add(int(order[j]))
                hit += len(got & tset)
                tot += len(tset)
                nc += len(got)
                if tset:
                    etot += 1
                    if got & tset:
                        ehit += 1
            el = (time.time() - t1) / len(q2) * 1000
            bkres["k%d" % k] = {"pair_recall": hit / tot if tot else 0,
                                "entity_recall": ehit / etot if etot else 0,
                                "avg_cands": nc / len(q2),
                                "reduction": 1 - (nc / len(q2)) / 200000,
                                "ms_per_query": el,
                                "note": "200K pool sample, not full pool"}
            log("bk k=%d %s" % (k, json.dumps(bkres["k%d" % k])))
        b = json.load(open(os.path.join(MET, "blocking.json")))
        b["bktree_sample"] = bkres
        json.dump(b, open(os.path.join(MET, "blocking.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
