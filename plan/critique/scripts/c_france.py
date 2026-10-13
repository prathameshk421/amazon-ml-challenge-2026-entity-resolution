"""Critique job C: structural statistics on France test rows (allowed: character/token statistics only,
no S1<->pool matching on test). Compares token vocabularies of France S1 vs France pool (test) and
checks how the reference normalizer's fixed maps (NAME_ABBREV / ADDR_ABBREV) collide with French usage.
Output: plan/critique/out/c_france.json
"""
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "plan", "reference"))
from translit_ref import ADDR_ABBREV, NAME_ABBREV  # noqa: E402

OUT = os.path.join(ROOT, "plan", "critique", "out")
D = os.path.join(ROOT, "student_resource", "dataset", "test")
R = {}

s1 = pq.read_table(os.path.join(OUT, "norm_test_s1.parquet"), columns=["name_norm", "addr_norm", "country"]).to_pydict()
fr = [(n, a) for n, a, c in zip(s1["name_norm"], s1["addr_norm"], s1["country"]) if c == "France"]
pool = pq.read_table(os.path.join(OUT, "norm_test_pool_france.parquet"), columns=["name_norm", "addr_norm"]).to_pydict()
pl = list(zip(pool["name_norm"], pool["addr_norm"]))
R["n_france_s1"], R["n_france_pool"] = len(fr), len(pl)


def docfreq(rows, j):
    c = Counter()
    for r in rows:
        c.update(set(r[j].split()))
    return c


s1n, s1a, pn, pa_ = docfreq(fr, 0), docfreq(fr, 1), docfreq(pl, 0), docfreq(pl, 1)
N1, NP = len(fr), len(pl)
legal = ["sarl", "sas", "sasu", "eurl", "sa", "snc", "sci", "scop", "selarl", "scm", "gie", "ei", "eirl",
         "societe", "ste", "ets", "etablissements", "cie", "compagnie", "and", "et", "company", "co", "freres",
         "fils", "groupe", "group", "france", "llc", "inc", "limited"]
R["name_token_share_france"] = {t: {"s1": round(s1n[t] / N1, 4), "pool": round(pn[t] / NP, 4)} for t in legal}
addr = ["rue", "r", "avenue", "av", "ave", "boulevard", "bd", "blvd", "bld", "chemin", "ch", "che", "route", "rte",
        "place", "pl", "impasse", "imp", "allee", "all", "quai", "cours", "faubourg", "fbg", "cedex", "bp", "cs",
        "st", "street", "saint", "ste", "suite", "sainte", "dr", "drive", "ct", "court", "ln", "lane", "no", "bis", "ter",
        "zi", "za", "zac", "lieu", "dit", "residence", "res", "batiment", "bat", "etage"]
R["addr_token_share_france"] = {t: {"s1": round(s1a[t] / N1, 4), "pool": round(pa_[t] / NP, 4)} for t in addr}
# tokens produced by the fixed maps (the map outputs; the inputs no longer exist after normalization)
R["note"] = ("shares are per-record document frequencies after the reference normalizer; "
             "'street' / 'suite' / 'drive' / 'court' in France rows can only come from ADDR_ABBREV "
             "(st->street, ste->suite, dr->drive, ct->court) or from genuine English words.")
# raw-side check of the collision inputs, on raw France rows
raw_hits = Counter()
raw_n = Counter()
ex = {}
pat = {k: re.compile(r"(?<![A-Za-z])" + k + r"(?![A-Za-z])", re.I) for k in ("st", "ste", "dr", "ct", "ave", "ln", "co")}
for fn, key in (("test_source1.tsv", "s1"), ("test_source2.tsv", "pool"), ("test_source3.tsv", "pool")):
    with open(os.path.join(D, fn), encoding="utf-8", newline="") as f:
        next(f)
        for line in f:
            _, name, a, c = line.rstrip("\n").split("\t")
            if c != "France":
                continue
            raw_n[key] += 1
            for k, rx in pat.items():
                if rx.search(a):
                    raw_hits[key + "|addr|" + k] += 1
                    if (key + k) not in ex:
                        ex[key + k] = a
            if pat["co"].search(name):
                raw_hits[key + "|name|co"] += 1
R["raw_collision_hits_share"] = {k: round(v / raw_n[k.split("|")[0]], 4) for k, v in sorted(raw_hits.items())}
R["raw_collision_examples"] = ex
# accents / non-ascii on raw S1 France (NFKD handles combining marks; list chars that are NOT decomposable)
nd = Counter()
with open(os.path.join(D, "test_source1.tsv"), encoding="utf-8", newline="") as f:
    next(f)
    for line in f:
        _, name, a, c = line.rstrip("\n").split("\t")
        if c != "France":
            continue
        for ch in name + a:
            if ord(ch) > 127 and not unicodedata.normalize("NFKD", ch).isascii():
                nd[ch] += 1
R["france_s1_nonascii_not_nfkd_ascii"] = dict(nd.most_common(20))
# vocabulary asymmetry: frequent France pool name tokens absent from France S1 vocabulary and vice versa
R["top_pool_name_tokens_rare_in_s1"] = [[t, v, s1n[t]] for t, v in pn.most_common(400) if s1n[t] * NP < 0.2 * v * N1][:25]
R["top_s1_name_tokens_rare_in_pool"] = [[t, v, pn[t]] for t, v in s1n.most_common(400) if pn[t] * N1 < 0.2 * v * NP][:25]
R["top_pool_addr_tokens_rare_in_s1"] = [[t, v, s1a[t]] for t, v in pa_.most_common(400) if s1a[t] * NP < 0.2 * v * N1][:25]
R["top_s1_addr_tokens_rare_in_pool"] = [[t, v, pa_[t]] for t, v in s1a.most_common(400) if pa_[t] * N1 < 0.2 * v * NP][:25]
R["top_france_s1_name_tokens"] = s1n.most_common(30)
R["top_france_pool_name_tokens"] = pn.most_common(30)
# skeleton check for French phonetics (the skeleton was designed for Indic-English)
sys.path.insert(0, os.path.join(ROOT, "plan", "reference"))
from translit_ref import skeleton  # noqa: E402
R["skeleton_examples"] = {w: skeleton(w) for w in ["boulangerie", "boulanger", "chez", "cheese", "champs", "shams",
                                                   "societe", "saint", "sainte", "street", "suite", "rue", "roux",
                                                   "renault", "reno", "gauthier", "gautier", "jacques", "jack", "zak"]}
json.dump(R, open(os.path.join(OUT, "c_france.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False)[:5000])
