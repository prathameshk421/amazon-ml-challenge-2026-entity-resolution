"""Phase 2 supplement: address-token CSR index (df<=20000 at assembly).
Writes artifacts/index/addr.npz. Run in background.
"""

import os
import time
from array import array
from collections import defaultdict

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_address

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IDX = os.path.join(ROOT, "artifacts/index")
CAP = 20000

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


def main():
    nblob = np.memmap(os.path.join(IDX, "pool.addrs.bin"), mode="r", dtype=np.uint8)
    aoff = np.load(os.path.join(IDX, "pool.addrs.off.npy"), mmap_mode="r")
    n = len(aoff) - 1
    d = defaultdict(lambda: array("I"))
    for i in range(n):
        ad = bytes(nblob[aoff[i]:aoff[i + 1]]).decode("utf-8", "replace")
        for t in set(normalize_address(ad).split()):
            d[t].append(i)
        if (i + 1) % 2000000 == 0:
            log("indexed %dM" % ((i + 1) // 1000000))
    keep = sorted(k for k, v in d.items() if len(v) <= CAP)
    total = sum(len(d[k]) for k in keep)
    data = np.empty(total, dtype=np.uint32)
    indptr = np.empty(len(keep) + 1, dtype=np.int64)
    indptr[0] = 0
    off = 0
    for i, k in enumerate(keep):
        v = d[k]
        data[off:off + len(v)] = np.asarray(v, dtype=np.uint32)
        del d[k]
        off += len(v)
        indptr[i + 1] = off
    np.savez(os.path.join(IDX, "addr.npz"),
             a_keys=np.array(keep), a_indptr=indptr, a_data=data)
    log("WROTE addr.npz keys=%d postings=%d" % (len(keep), total))


if __name__ == "__main__":
    main()
