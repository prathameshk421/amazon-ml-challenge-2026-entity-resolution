"""Phase 2 step A: stream full train pool -> mmap blobs + token-df table.
Writes artifacts/index/pool.{names,addrs}.bin/.off.npy, meta (src/num/country),
tokdf.json (tokens with df, pruned to df>=2 to bound size).
Run in background: nohup ... > logs/idx_a.log 2>&1 &
"""

import csv
import json
import os
import re
import time
from collections import Counter

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
IDX = os.path.join(ROOT, "artifacts/index")
os.makedirs(IDX, exist_ok=True)
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


def main():
    df = Counter()
    srcs, nums, cos = [], [], []
    cmap = {}
    nbf = open(os.path.join(IDX, "pool.names.bin"), "wb")
    abf = open(os.path.join(IDX, "pool.addrs.bin"), "wb")
    noff, aoff = [0], [0]
    n = 0
    for fn, src in [("train_source2.tsv", 0), ("train_source3.tsv", 1)]:
        with open(os.path.join(TR, fn), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                eid = row["entity_id"]
                nm = row["business_name"] or ""
                ad = row["business_address"] or ""
                co = row["country"] or ""
                nb, ab = nm.encode("utf-8"), ad.encode("utf-8")
                nbf.write(nb)
                abf.write(ab)
                noff.append(noff[-1] + len(nb))
                aoff.append(aoff[-1] + len(ab))
                srcs.append(src)
                nums.append(int(eid.split("-", 1)[1]))
                if co not in cmap:
                    cmap[co] = len(cmap)
                cos.append(cmap[co])
                for t in set(TOKEN_RE.findall(nm.lower())):
                    if len(t) <= 25:
                        df[t] += 1
                n += 1
                if n % 2000000 == 0:
                    log("streamed %dM" % (n // 1000000))
    nbf.close()
    abf.close()
    np.save(os.path.join(IDX, "pool.names.off.npy"), np.array(noff, dtype=np.int64))
    np.save(os.path.join(IDX, "pool.addrs.off.npy"), np.array(aoff, dtype=np.int64))
    np.save(os.path.join(IDX, "pool.src.npy"), np.array(srcs, dtype=np.uint8))
    np.save(os.path.join(IDX, "pool.num.npy"), np.array(nums, dtype=np.int64))
    np.save(os.path.join(IDX, "pool.co.npy"), np.array(cos, dtype=np.uint8))
    meta = {"n": n, "cmap": cmap,
            "vocab_size": len(df),
            "n_singleton_tokens": sum(1 for v in df.values() if v == 1)}
    json.dump(meta, open(os.path.join(IDX, "pool.meta.json"), "w"))
    log("vocab=%d singleton=%d; writing tokdf" % (len(df), meta["n_singleton_tokens"]))
    json.dump(df, open(os.path.join(IDX, "pool.tokdf.json"), "w"))
    log("DONE n=%d" % n)


if __name__ == "__main__":
    main()
