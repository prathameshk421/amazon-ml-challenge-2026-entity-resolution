"""Bucketing experiment: auxiliary pool index with two NEW key families.
  PAIR: unordered pairs of name tokens (both len>=2), from the first 8 distinct tokens of
        the name key text. Rescues names made only of common words (every single token
        is over the df cap) and chain names with a typo in one token.
  ROM : for pool names containing Indic-script letters, tokens and char-trigrams of the
        romanized name (legacy/src/translit.py, stdlib unicodedata only). Today those names
        normalize to "" and are invisible to every name channel.
Name key text = romanized name if it contains Indic letters, else normalize_name(raw).
Keys are 64-bit Python hashes (run with PYTHONHASHSEED=0; queries must use the same).
Writes artifacts/index/aux.npz (keys uint64 sorted, indptr, data uint32; no df cap
applied here, caps are applied at query time).
"""

import os
import sys
import time
from array import array

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name
from translit import romanize_words_schwa, has_indic

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IDX = os.path.join(ROOT, "artifacts/index")
assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


def name_key_text(raw):
    if raw and has_indic(raw):
        return normalize_name(romanize_words_schwa(raw)), True
    return normalize_name(raw), False


def pair_keys(text):
    toks = []
    for t in text.split():
        if len(t) >= 2 and t not in toks:
            toks.append(t)
        if len(toks) == 8:
            break
    out = []
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            a, b = (toks[i], toks[j]) if toks[i] < toks[j] else (toks[j], toks[i])
            out.append(hash("P:" + a + "|" + b))
    return out


def rom_keys(text):
    out = [hash("RT:" + t) for t in set(text.split())]
    s = " " + text + " "
    out += [hash("RG:" + s[i:i + 3]) for i in set(range(len(s) - 2))]
    return out


def main():
    nblob = np.memmap(os.path.join(IDX, "pool.names.bin"), mode="r", dtype=np.uint8)
    noff = np.load(os.path.join(IDX, "pool.names.off.npy"))
    nb = bytes(nblob)
    n = len(noff) - 1
    H = array("q")
    R = array("I")
    n_rom = 0
    for i in range(n):
        raw = nb[noff[i]:noff[i + 1]].decode("utf-8", "replace")
        txt, rom = name_key_text(raw)
        ks = pair_keys(txt)
        if rom:
            n_rom += 1
            ks += rom_keys(txt)
        ks = set(ks)
        H.extend(ks)
        R.extend([i] * len(ks))
        if (i + 1) % 2000000 == 0:
            log("%dM rows, %d postings, %d romanized" % ((i + 1) // 1000000, len(H), n_rom))
    del nb
    h = np.frombuffer(H, dtype=np.int64).view(np.uint64)
    r = np.frombuffer(R, dtype=np.uint32)
    log("sorting %d postings" % len(h))
    o = np.argsort(h, kind="stable")
    h = h[o]
    r = r[o]
    del o
    keys, start = np.unique(h, return_index=True)
    indptr = np.append(start, len(h)).astype(np.int64)
    np.savez(os.path.join(IDX, "aux.npz"), keys=keys, indptr=indptr, data=r)
    df = np.diff(indptr)
    log("WROTE aux.npz keys=%d postings=%d romanized_rows=%d df>5000=%d df>20000=%d"
        % (len(keys), len(h), n_rom, int((df > 5000).sum()), int((df > 20000).sum())))


if __name__ == "__main__":
    main()
