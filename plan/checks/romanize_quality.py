"""How close is stdlib romanization (legacy/src/translit.py) to the Latin S1 name, on fit-part
true pairs whose pool name contains Indic letters? Informs whether the rebuild needs a
train-derived token dictionary. Writes plan/checks/romanize_quality.json.
"""

import json
import os
import random
import sys
from collections import Counter

import jellyfish

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "legacy", "src"))
from normalize import normalize_name
from translit import romanize_words_schwa, has_indic

D = os.path.join(ROOT, "student_resource/dataset/train")


def tri(s):
    s = " " + s + " "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def main():
    split = json.load(open(os.path.join(ROOT, "artifacts/split.json")))
    fit = set(split["fit_ids"])
    indic = {}
    per_script = Counter()
    for s in ("2", "3"):
        with open(os.path.join(D, "train_source%s.tsv" % s), encoding="utf-8") as f:
            f.readline()
            for line in f:
                eid, name, addr, co = line.rstrip("\n").split("\t")
                if has_indic(name):
                    indic[eid] = name
    s1 = {}
    with open(os.path.join(D, "train_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            eid, name, addr, co = line.rstrip("\n").split("\t")
            if eid in fit:
                s1[eid] = name
    pairs = []
    with open(os.path.join(D, "train_ground_truth.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            if a in s1 and b:
                for x in b.split(","):
                    if x in indic:
                        pairs.append((a, x))
    rng = random.Random(7)
    rng.shuffle(pairs)
    pairs = pairs[:30000]
    st = Counter()
    jac = []
    tok_align = Counter()
    examples = []
    for a, x in pairs:
        L = normalize_name(s1[a])
        R = normalize_name(romanize_words_schwa(indic[x]))
        lt, rt = L.split(), R.split()
        tj = len(tri(L) & tri(R)) / max(1, len(tri(L) | tri(R)))
        jac.append(tj)
        st["pairs"] += 1
        st["any_token_equal"] += int(bool(set(lt) & set(rt)))
        lm = {jellyfish.metaphone(t) for t in lt if t}
        rm = {jellyfish.metaphone(t) for t in rt if t}
        st["any_metaphone_equal"] += int(bool(lm & rm))
        st["same_token_count"] += int(len(lt) == len(rt))
        st["tri_jac_ge_0.3"] += int(tj >= 0.3)
        st["tri_jac_ge_0.5"] += int(tj >= 0.5)
        if len(lt) == len(rt):
            for p, q in zip(rt, lt):
                tok_align[(p, q)] += 1
        if len(examples) < 12:
            examples.append({"s1": s1[a][:50], "pool": indic[x][:50], "romanized": R[:50]})
    n = st["pairs"]
    jac.sort()
    out = {"n_indic_pool_names": len(indic), "n_pairs_evaluated": n,
           "rates": {k: round(v / n, 4) for k, v in st.items() if k != "pairs"},
           "tri_jaccard_quantiles": {q: round(jac[int(q * (n - 1))], 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
           "top_positional_token_pairs": [[p, q, c] for (p, q), c in tok_align.most_common(40)],
           "examples": examples}
    json.dump(out, open(os.path.join(ROOT, "plan/checks/romanize_quality.json"), "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("n_indic_pool_names", "n_pairs_evaluated", "rates",
                                         "tri_jaccard_quantiles")}, indent=1))


if __name__ == "__main__":
    main()
