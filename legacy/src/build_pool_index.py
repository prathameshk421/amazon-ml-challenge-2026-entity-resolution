"""Phase 2 step B: build full-pool blocking indexes -> artifacts/index/index.npz.
CSR postings (uint32 row-ids) for: exact normalized name, rare normalized tokens
(df<=20000, pruned at assembly), char trigrams, metaphone token codes.
Loads blobs via mmap. Run in background after step A.
"""

import json
import os
import time
from array import array
from collections import defaultdict

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name
import jellyfish

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IDX = os.path.join(ROOT, "artifacts/index")
TOK_DF_CAP = 20000

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


def csr_from_dict(d, keys_sorted, prune_max=None):
    # preallocated single-copy assembly (peak-RAM safe)
    keep = []
    total = 0
    for k in keys_sorted:
        ln = len(d[k])
        if prune_max is not None and ln > prune_max:
            continue
        keep.append(k)
        total += ln
    data = np.empty(total, dtype=np.uint32)
    indptr = np.empty(len(keep) + 1, dtype=np.int64)
    indptr[0] = 0
    off = 0
    for i, k in enumerate(keep):
        arr = d[k]
        ln = len(arr)
        data[off:off + ln] = np.asarray(arr, dtype=np.uint32)
        del d[k]
        off += ln
        indptr[i + 1] = off
    return np.array(keep), indptr, data


def trigrams(s):
    s = " " + s + " "
    return set(s[i:i + 3] for i in range(len(s) - 2))


def main():
    meta = json.load(open(os.path.join(IDX, "pool.meta.json")))
    n = meta["n"]
    log("n=%d" % n)
    nblob = np.memmap(os.path.join(IDX, "pool.names.bin"), mode="r", dtype=np.uint8)
    noff = np.load(os.path.join(IDX, "pool.names.off.npy"), mmap_mode="r")
    exact = defaultdict(lambda: array("I"))
    tok = defaultdict(lambda: array("I"))
    tri = defaultdict(lambda: array("I"))
    met = defaultdict(lambda: array("I"))
    for i in range(n):
        nm = bytes(nblob[noff[i]:noff[i + 1]]).decode("utf-8", "replace")
        nn = normalize_name(nm)
        exact[nn].append(i)
        for t in set(nn.split()):
            tok[t].append(i)
        for tg in trigrams(nn):
            tri[tg].append(i)
        for t in set(nn.split()):
            if t:
                met[jellyfish.metaphone(t)].append(i)
        if (i + 1) % 2000000 == 0:
            log("indexed %dM" % ((i + 1) // 1000000))
    log("accumulated: exact=%d tok=%d tri=%d met=%d" % (len(exact), len(tok), len(tri), len(met)))
    out = {}
    exact_dict = None
    for prefix, d, prune in [("ex", exact, None), ("tok", tok, TOK_DF_CAP),
                             ("tri", tri, None), ("met", met, None)]:
        if prefix == "ex":
            # 7M unicode keys blow up npz (fixed-width U dtype); pickle instead
            exact_dict = {k: np.asarray(v, dtype=np.uint32) for k, v in d.items()}
            log("ex: keys=%d postings=%d (pickled)" % (len(exact_dict),
                sum(len(v) for v in exact_dict.values())))
            del d
            continue
        keys = sorted(d.keys())
        k, ip, data = csr_from_dict(d, keys, prune)
        out[prefix + "_keys"] = k
        out[prefix + "_indptr"] = ip
        out[prefix + "_data"] = data
        log("%s: keys=%d postings=%d" % (prefix, len(k), len(data)))
        del d
    np.savez(os.path.join(IDX, "index.npz"), **out)
    import pickle
    with open(os.path.join(IDX, "exact.pkl"), "wb") as f:
        pickle.dump(exact_dict, f, protocol=4)
    stats = {"tok_df_cap": TOK_DF_CAP,
             "n_exact_keys": int(len(exact_dict)),
             "n_tok_keys": int(len(out["tok_keys"])),
             "n_tri_keys": int(len(out["tri_keys"])),
             "n_met_keys": int(len(out["met_keys"])),
             "tok_postings": int(len(out["tok_data"])),
             "tri_postings": int(len(out["tri_data"])),
             "met_postings": int(len(out["met_data"]))}
    json.dump(stats, open(os.path.join(IDX, "index.stats.json"), "w"), indent=1)
    log("WROTE index.npz " + json.dumps(stats))


if __name__ == "__main__":
    main()
