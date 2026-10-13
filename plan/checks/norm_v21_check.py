"""Compare reference normalizer v1 (plan/reference/translit_ref_v1.py) vs v2.1
(plan/reference/translit_ref.py: Gurmukhi TIPPI/BINDI/ADDAK/OM fixes, single-letter collapse in
names, st/ste canonical classes in addresses) on TRAIN true links.
Sample: all links of 150,000 random S1 entities (seed 5) + all links whose pool name contains
Gurmukhi letters. Metrics per version: exact name equality, >=1 shared name token (len>=2),
mean address-token Jaccard; per-script shared-token rate for Indic pool names.
Writes plan/checks/norm_v21_check.json.
"""

import importlib.util
import json
import os
import random
import unicodedata
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "student_resource/dataset/train")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


V1 = load_mod("v1", os.path.join(ROOT, "plan/reference/translit_ref_v1.py"))
V2 = load_mod("v2", os.path.join(ROOT, "plan/reference/translit_ref.py"))


def script(s):
    for c in s:
        if 0x0900 <= ord(c) <= 0x0D7F:
            return unicodedata.name(c).split()[0]
    return ""


def main():
    s1 = {}
    with open(os.path.join(D, "train_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            e, n, a, co = line.rstrip("\n").split("\t")
            s1[e] = (n, a)
    pool = {}
    for s in ("2", "3"):
        with open(os.path.join(D, "train_source%s.tsv" % s), encoding="utf-8") as f:
            f.readline()
            for line in f:
                e, n, a, co = line.rstrip("\n").split("\t")
                pool[e] = (n, a)
    samp = set(random.Random(5).sample(sorted(s1), 150000))
    pairs = []
    with open(os.path.join(D, "train_ground_truth.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            if not b:
                continue
            for x in b.split(","):
                if a in samp or script(pool[x][0]) == "GURMUKHI":
                    pairs.append((a, x))
    res = {}
    for tag, M in (("v1", V1), ("v2.1", V2)):
        c = Counter()
        per = Counter()
        jac = 0.0
        nj = 0
        for a, x in pairs:
            sc = script(pool[x][0])
            q = M.normalize_field(s1[a][0], "name")[0]
            p = M.normalize_field(pool[x][0], "name")[0]
            qa = set(M.normalize_field(s1[a][1], "addr")[0].split())
            pa = set(M.normalize_field(pool[x][1], "addr")[0].split())
            shared = bool({t for t in q.split() if len(t) >= 2} & {t for t in p.split() if len(t) >= 2})
            if a in samp:
                c["n"] += 1
                c["exact"] += q == p
                c["tok"] += shared
                if qa and pa:
                    jac += len(qa & pa) / len(qa | pa)
                    nj += 1
            if sc:
                per[sc + "_n"] += 1
                per[sc + "_tok"] += shared
        res[tag] = {"sample_links": c["n"], "exact_name_equal": round(c["exact"] / c["n"], 4),
                    "shared_token": round(c["tok"] / c["n"], 4),
                    "addr_jaccard_mean": round(jac / nj, 4),
                    "per_script_shared_token": {k[:-2]: round(per[k[:-2] + "_tok"] / v, 4)
                                                for k, v in per.items() if k.endswith("_n")},
                    "per_script_n": {k[:-2]: v for k, v in per.items() if k.endswith("_n")}}
    json.dump(res, open(os.path.join(ROOT, "plan/checks/norm_v21_check.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
