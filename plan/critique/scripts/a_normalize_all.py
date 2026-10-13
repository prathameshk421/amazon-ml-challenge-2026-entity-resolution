"""Critique job A: normalize names/addresses of train S1, train pool, test S1 and test-pool France rows
with the plan's reference normalizer (plan/reference/translit_ref.py, no dictionary).
Writes plan/critique/out/norm_<part>.parquet. Read-only on everything else.
"""
import os
import sys
import time
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "plan", "reference"))
from translit_ref import normalize_field  # noqa: E402

D = os.path.join(ROOT, "student_resource", "dataset")
OUT = os.path.join(ROOT, "plan", "critique", "out")


def script_of(text):
    for c in text:
        o = ord(c)
        if 0x0900 <= o <= 0x0D7F:
            return unicodedata.name(c, "X ").split(" ")[0]
    return ""


def addr_state(a):
    if a == "":
        return 1
    rest = a.replace("<NULL>", "")
    if not any(ch.isalnum() for ch in rest):
        return 2
    return 0


def run(part, files, only_country=None):
    cols = {k: [] for k in ("src", "num", "name_raw", "name_norm", "addr_norm", "had_indic", "script",
                            "country", "addr_state")}
    n = 0
    t = time.time()
    for fn in files:
        with open(os.path.join(D, fn), encoding="utf-8", newline="") as f:
            next(f)
            for line in f:
                eid, name, addr, country = line.rstrip("\n").split("\t")
                if only_country and country != only_country:
                    continue
                nn, had = normalize_field(name, "name")
                an, _ = normalize_field(addr, "addr")
                cols["src"].append(int(eid[1]))
                cols["num"].append(int(eid[3:]))
                cols["name_raw"].append(name)
                cols["name_norm"].append(nn)
                cols["addr_norm"].append(an)
                cols["had_indic"].append(had)
                cols["script"].append(script_of(name) if had else "")
                cols["country"].append(country)
                cols["addr_state"].append(addr_state(addr))
                n += 1
    tab = pa.table({
        "src": pa.array(cols["src"], pa.int8()), "num": pa.array(cols["num"], pa.int64()),
        "name_raw": cols["name_raw"], "name_norm": cols["name_norm"], "addr_norm": cols["addr_norm"],
        "had_indic": cols["had_indic"], "script": cols["script"], "country": cols["country"],
        "addr_state": pa.array(cols["addr_state"], pa.int8())})
    pq.write_table(tab, os.path.join(OUT, f"norm_{part}.parquet"), compression="zstd")
    print(part, n, "rows", round(time.time() - t, 1), "s", flush=True)


if __name__ == "__main__":
    run("train_s1", ["train/train_source1.tsv"])
    run("test_s1", ["test/test_source1.tsv"])
    run("test_pool_france", ["test/test_source2.tsv", "test/test_source3.tsv"], only_country="France")
    run("train_pool_s2", ["train/train_source2.tsv"])
    run("train_pool_s3", ["train/train_source3.tsv"])
