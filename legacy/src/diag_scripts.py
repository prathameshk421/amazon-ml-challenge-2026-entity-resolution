"""Bucketing diagnostic A: writing-system census + normalization-loss check.

Train pool (all 10.3M S2+S3 records, from artifacts/index/pool.*) and test sources
(C7-allowed: character-set / non-ASCII statistics per country only, no matching).
Per record and field (name, address) the class is one of:
  empty | ascii | latin_ext (non-ASCII, but every letter is Latin) | mixed (Latin +
  another script) | nonlatin (letters present, none Latin).
Also counts records whose raw text is non-empty but normalize_*() returns "" (the
text is erased before blocking), and the letters that normalization deletes.
Writes artifacts/metrics/diag_scripts.json.
"""

import csv
import json
import os
import sys
import time
import unicodedata
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name, normalize_address, _base

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IDX = os.path.join(ROOT, "artifacts/index")
TE = os.path.join(ROOT, "student_resource/dataset/test")
MET = os.path.join(ROOT, "artifacts/metrics")
csv.field_size_limit(10 ** 8)

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


_script_cache = {}
def script_of(c):
    s = _script_cache.get(c)
    if s is None:
        if not unicodedata.category(c).startswith("L"):
            s = ""
        else:
            nm = unicodedata.name(c, "UNKNOWN")
            s = nm.split(" ")[0]
            if s in ("CJK", "HANGUL", "HIRAGANA", "KATAKANA"):
                s = "CJK"
        _script_cache[c] = s
    return s


def classify(text):
    """-> (class, dominant_nonlatin_script or '')"""
    if not text:
        return "empty", ""
    if text.isascii():
        return "ascii", ""
    scripts = Counter()
    for c in text:
        if ord(c) < 128:
            if c.isalpha():
                scripts["LATIN"] += 1
            continue
        s = script_of(c)
        if s:
            scripts[s] += 1
    other = {k: v for k, v in scripts.items() if k != "LATIN"}
    if not other:
        return "latin_ext", ""
    dom = max(other, key=other.get)
    return ("mixed" if scripts.get("LATIN") else "nonlatin"), dom


def lost_letters(text):
    """Letters present in the raw text that do not survive _base()."""
    out = []
    for c in set(text):
        if ord(c) < 128 or not unicodedata.category(c).startswith("L"):
            continue
        if not _base(c):
            out.append(c)
    return out


class Acc:
    def __init__(self):
        self.cls = defaultdict(Counter)       # (country, field) -> class counts
        self.scr = defaultdict(Counter)       # (country, field) -> script counts
        self.erased = defaultdict(int)        # (country, field) -> normalized to ""
        self.lost = defaultdict(Counter)      # country -> lost letter counts
        self.both_nonlatin = Counter()        # country -> name AND addr nonlatin/empty
        self.n = Counter()

    def add(self, co, name, addr):
        self.n[co] += 1
        cn, sn = classify(name)
        ca, sa = classify(addr)
        self.cls[(co, "name")][cn] += 1
        self.cls[(co, "addr")][ca] += 1
        if sn:
            self.scr[(co, "name")][sn] += 1
        if sa:
            self.scr[(co, "addr")][sa] += 1
        if name and not normalize_name(name):
            self.erased[(co, "name")] += 1
        if addr and not normalize_address(addr):
            self.erased[(co, "addr")] += 1
        if cn == "nonlatin" and ca in ("nonlatin", "empty"):
            self.both_nonlatin[co] += 1
        if cn != "ascii" or ca != "ascii":
            for c in lost_letters(name + " " + addr):
                self.lost[co][c] += 1

    def dump(self):
        out = {"n": dict(self.n), "class": {}, "scripts": {}, "erased_by_normalize": {},
               "name_nonlatin_and_addr_nonlatin_or_empty": dict(self.both_nonlatin),
               "lost_letters_top30": {}}
        for (co, f), c in self.cls.items():
            out["class"]["%s|%s" % (co, f)] = dict(c)
        for (co, f), c in self.scr.items():
            out["scripts"]["%s|%s" % (co, f)] = dict(c.most_common(12))
        for (co, f), v in self.erased.items():
            out["erased_by_normalize"]["%s|%s" % (co, f)] = v
        for co, c in self.lost.items():
            out["lost_letters_top30"][co] = [[k, unicodedata.name(k, "?"), v]
                                             for k, v in c.most_common(30)]
        return out


def main():
    res = {}
    meta = json.load(open(os.path.join(IDX, "pool.meta.json")))
    inv = {v: k for k, v in meta["cmap"].items()}
    nblob = np.memmap(os.path.join(IDX, "pool.names.bin"), mode="r", dtype=np.uint8)
    noff = np.load(os.path.join(IDX, "pool.names.off.npy"))
    ablob = np.memmap(os.path.join(IDX, "pool.addrs.bin"), mode="r", dtype=np.uint8)
    aoff = np.load(os.path.join(IDX, "pool.addrs.off.npy"))
    co = np.load(os.path.join(IDX, "pool.co.npy"))
    src = np.load(os.path.join(IDX, "pool.src.npy"))
    n = len(noff) - 1
    accs = {"train_S2": Acc(), "train_S3": Acc()}
    nb = bytes(nblob)
    ab = bytes(ablob)
    for i in range(n):
        name = nb[noff[i]:noff[i + 1]].decode("utf-8", "replace")
        addr = ab[aoff[i]:aoff[i + 1]].decode("utf-8", "replace")
        accs["train_S2" if src[i] == 0 else "train_S3"].add(inv[int(co[i])], name, addr)
        if (i + 1) % 2000000 == 0:
            log("train pool %dM" % ((i + 1) // 1000000))
    del nb, ab
    for k, a in accs.items():
        res[k] = a.dump()
    for s in (1, 2, 3):
        a = Acc()
        with open(os.path.join(TE, "test_source%d.tsv" % s), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                a.add(row["country"], row["business_name"], row["business_address"])
        res["test_S%d" % s] = a.dump()
        log("test S%d done" % s)
    a = Acc()
    with open(os.path.join(ROOT, "student_resource/dataset/train/train_source1.tsv"),
              newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            a.add(row["country"], row["business_name"], row["business_address"])
    res["train_S1"] = a.dump()
    json.dump(res, open(os.path.join(MET, "diag_scripts.json"), "w"), indent=1,
              ensure_ascii=False)
    log("WROTE diag_scripts.json")


if __name__ == "__main__":
    main()
