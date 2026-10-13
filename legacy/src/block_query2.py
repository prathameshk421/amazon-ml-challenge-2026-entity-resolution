"""Phase 2b: numpy-optimized ranked blocking eval.
Candidates(e) = top-K by combined idf score over: exact(+100 boost) + name-token idf
  + trigram idf (df<=8000) + addr-token idf + 0.2*shared-metaphone.
Eval: same 10K fit queries (seed 123); recall@K K=50/100/150/250/500; source marginals;
  10 missed-by-all examples. Writes artifacts/metrics/blocking2.json.
"""

import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name, normalize_address
import jellyfish

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
IDX = os.path.join(ROOT, "artifacts/index")
MET = os.path.join(ROOT, "artifacts/metrics")
POOL_N = 10320219
SEED = 123
TRI_DF_CAP = 8000
PHON_W = float(os.environ.get("PHON_W", "0.2"))

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


class Q:
    def __init__(self):
        import pickle
        # materialize: lazy NpzFile.__getitem__ re-reads from zip on EVERY access
        _z = np.load(os.path.join(IDX, "index.npz"))
        self.z = {k: _z[k] for k in _z.files}
        _a = np.load(os.path.join(IDX, "addr.npz"))
        self.a = {k: _a[k] for k in _a.files}
        with open(os.path.join(IDX, "exact.pkl"), "rb") as f:
            self.exact = pickle.load(f)
        self.tri_df = np.diff(self.z["tri_indptr"]).astype(np.float64)
        self.tri_idf = np.log((POOL_N + 1) / (self.tri_df + 1))
        self.met_df = np.diff(self.z["met_indptr"]).astype(np.float64)
        tok_df = np.diff(self.z["tok_indptr"]).astype(np.float64)
        self.tok_idf = np.log((POOL_N + 1) / (tok_df + 1))
        adf = np.diff(self.a["a_indptr"]).astype(np.float64)
        self.a_idf = np.log((POOL_N + 1) / (adf + 1))
        # O(1) key lookup (searchsorted on 1.4M unicode keys costs ~0.1s each)
        self.pos = {}
        for prefix, zz in (("tok", self.z), ("tri", self.z), ("met", self.z), ("a", self.a)):
            self.pos[prefix] = {k: i for i, k in enumerate(zz[prefix + "_keys"].tolist())}

    def range_of(self, z, prefix, key):
        j = self.pos[prefix].get(key)
        if j is None:
            return None
        return z[prefix + "_indptr"][j], z[prefix + "_indptr"][j + 1], j

    def ranked(self, nn, na, K=500):
        EX, NT, TR, AD, PH = 1, 2, 4, 8, 16
        P = [np.zeros(0, dtype=np.uint32)]
        W = [np.zeros(0, dtype=np.float32)]
        F = [np.zeros(0, dtype=np.uint8)]
        z = self.z
        if nn in self.exact:
            a = self.exact[nn]
            P.append(a)
            W.append(np.full(len(a), 100.0, dtype=np.float32))
            F.append(np.full(len(a), EX, dtype=np.uint8))
        for t in set(nn.split()):
            r = self.range_of(z, "tok", t)
            if r:
                s, e, j = r
                a = z["tok_data"][s:e]
                P.append(a)
                W.append(np.full(len(a), float(self.tok_idf[j]), dtype=np.float32))
                F.append(np.full(len(a), NT, dtype=np.uint8))
        tgs = set((" " + nn + " ")[i:i + 3] for i in range(len(nn))) if nn else set()
        for tg in tgs:
            r = self.range_of(z, "tri", tg)
            if r:
                s, e, j = r
                if self.tri_df[j] > TRI_DF_CAP:
                    continue
                a = z["tri_data"][s:e]
                P.append(a)
                W.append(np.full(len(a), float(self.tri_idf[j]), dtype=np.float32))
                F.append(np.full(len(a), TR, dtype=np.uint8))
        for t in set(na.split()):
            r = self.range_of(self.a, "a", t)
            if r:
                s, e, j = r
                a = self.a["a_data"][s:e]
                P.append(a)
                W.append(np.full(len(a), float(self.a_idf[j]), dtype=np.float32))
                F.append(np.full(len(a), AD, dtype=np.uint8))
        for t in set(nn.split()):
            if t:
                r = self.range_of(z, "met", jellyfish.metaphone(t))
                if r:
                    s, e, j = r
                    if self.met_df[j] > 20000:
                        continue
                    a = z["met_data"][s:e]
                    P.append(a)
                    W.append(np.full(len(a), PHON_W, dtype=np.float32))
                    F.append(np.full(len(a), PH, dtype=np.uint8))
        Pall = np.concatenate(P)
        if len(Pall) == 0:
            return [], {}
        Wall = np.concatenate(W)
        Fall = np.concatenate(F)
        u, inv = np.unique(Pall, return_inverse=True)
        scores = np.bincount(inv, weights=Wall)
        # top-K by (-score, rowid): argpartition then sort the head
        if len(u) > K:
            head = np.argpartition(-scores, K - 1)[:K]
            order = head[np.lexsort((u[head], -scores[head]))]
        else:
            order = np.lexsort((u, -scores))
        # source flags for the head only (vectorized OR per tag)
        Fb = np.zeros(len(u), dtype=np.uint8)
        for tag in (EX, NT, TR, AD, PH):
            m = Fall == tag
            if m.any():
                present = np.bincount(inv[m], minlength=len(u)) > 0
                Fb[present] |= tag
        flags = {int(u[i]): int(Fb[i]) for i in order.tolist()}
        return [int(u[i]) for i in order.tolist()], flags


def main():
    import random
    rng = random.Random(SEED)
    split = json.load(open(os.path.join(ROOT, "artifacts/split.json")))
    fit_ids = split["fit_ids"]
    qids = rng.sample(fit_ids, 10000)
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
    log("mapping truth to rowids")
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
    log("loading index")
    q = Q()
    KS = [50, 100, 150, 250, 500]
    agg = {K: {"hit": 0, "tot": 0, "ehit": 0, "etot": 0} for K in KS}
    srcbones = {s: 0 for s in ["EX", "NT", "TR", "AD", "PH"]}
    srctot = 0
    t_q = 0.0
    allmiss = {}
    nblob = np.memmap(os.path.join(IDX, "pool.names.bin"), mode="r", dtype=np.uint8)
    noff = np.load(os.path.join(IDX, "pool.names.off.npy"), mmap_mode="r")
    for qi, e in enumerate(qids):
        nn = normalize_name(recs[e][0])
        na = normalize_address(recs[e][1])
        t1 = time.time()
        top500, flags = q.ranked(nn, na, K=500)
        t_q += time.time() - t1
        ts = set(tro[e])
        tops = {}
        for K in KS:
            tops[K] = set(top500[:K])
        for K in KS:
            got = tops[K] & ts
            agg[K]["hit"] += len(got)
            agg[K]["tot"] += len(ts)
            if ts:
                agg[K]["etot"] += 1
                if got:
                    agg[K]["ehit"] += 1
        for x in ts:
            if x in flags:
                f = flags[x]
                srctot += 1
                if f & 1:
                    srcbones["EX"] += 1
                if f & 2:
                    srcbones["NT"] += 1
                if f & 4:
                    srcbones["TR"] += 1
                if f & 8:
                    srcbones["AD"] += 1
                if f & 16:
                    srcbones["PH"] += 1
            else:
                allmiss[x] = e
        if (qi + 1) % 2000 == 0:
            log("eval %d/10000" % (qi + 1))
    res = {"K": {}}
    for K in KS:
        a = agg[K]
        res["K"][str(K)] = {"pair_recall": a["hit"] / a["tot"],
                            "entity_recall": a["ehit"] / a["etot"],
                            "avg_cands": K,
                            "reduction": 1 - K / POOL_N}
    res["ms_per_query"] = t_q / len(qids) * 1000
    res["source_attribution_top500"] = {k: (v / srctot if srctot else 0) for k, v in srcbones.items()}
    res["n_pos_pairs"] = sum(len(v) for v in tro.values())
    res["missed_count"] = len(allmiss)
    ex = []
    for x, e in list(allmiss.items())[:10]:
        nm = bytes(nblob[noff[x]:noff[x + 1]]).decode("utf-8", "replace")
        sxx = "S2-" if src[x] == 0 else "S3-"
        ex.append({"s1": e, "s1_name": recs[e][0], "s1_addr": recs[e][1],
                   "missed": sxx + str(int(num[x])), "missed_name": nm})
    res["missed_examples"] = ex
    json.dump(res, open(os.path.join(MET, "blocking2.json"), "w"), indent=1)
    for K in KS:
        log("top%-4d %s" % (K, json.dumps(res["K"][str(K)])))
    log("ms/query=%.1f src_attr=%s missed=%d" % (res["ms_per_query"],
        json.dumps(res["source_attribution_top500"]), len(allmiss)))


if __name__ == "__main__":
    main()
