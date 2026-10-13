"""Validate the reference romanizer: goldens + quality on 30K fit true pairs (same sample as
plan/checks/romanize_quality.py). Writes plan/reference/ref_quality.json."""
import json, os, random, sys
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "plan/reference"))
from translit_ref import normalize_field, skeleton
from rapidfuzz.distance import JaroWinkler
D = os.path.join(ROOT, "student_resource/dataset/train")
split = json.load(open(os.path.join(ROOT, "artifacts/split.json")))
fit = set(split["fit_ids"])
indic, s1 = {}, {}
for s in ("2", "3"):
    with open(os.path.join(D, "train_source%s.tsv" % s), encoding="utf-8") as f:
        f.readline()
        for line in f:
            eid, name, addr, co = line.rstrip("\n").split("\t")
            if any(0x0900 <= ord(c) <= 0x0D7F for c in name):
                indic[eid] = name
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
            pairs += [(a, x) for x in b.split(",") if x in indic]
rng = random.Random(7); rng.shuffle(pairs); pairs = pairs[:30000]
st = Counter(); tokc = Counter(); gold = {}
for a, x in pairs:
    L = normalize_field(s1[a])[0].split()
    R, had = normalize_field(indic[x])
    R = R.split()
    assert all(t.isascii() for t in R), indic[x]
    st["n"] += 1
    st["any_token_equal"] += bool(set(L) & set(R))
    sl = {skeleton(t) for t in L if len(skeleton(t)) >= 2}
    sr = {skeleton(t) for t in R if len(skeleton(t)) >= 2}
    st["any_skeleton_equal"] += bool(sl & sr)
    # share of S1 tokens (len>=2) matched by an identical romanized token / by skeleton
    for t in L:
        if len(t) < 2: continue
        st["s1_tokens"] += 1
        st["s1_tok_exact"] += t in R
        st["s1_tok_skel"] += skeleton(t) in sr
        st["s1_tok_jw85"] += any(JaroWinkler.normalized_similarity(t, r) >= 0.85 for r in R)
    if len(L) == len(R):
        for r, l in zip(R, L):
            if r != l: tokc[(r, l)] += 1
out = {"n_pairs": st["n"],
       "any_token_equal": round(st["any_token_equal"] / st["n"], 4),
       "any_skeleton_equal": round(st["any_skeleton_equal"] / st["n"], 4),
       "s1_token_exact_rate": round(st["s1_tok_exact"] / st["s1_tokens"], 4),
       "s1_token_skeleton_rate": round(st["s1_tok_skel"] / st["s1_tokens"], 4),
       "s1_token_jw085_rate": round(st["s1_tok_jw85"] / st["s1_tokens"], 4),
       "top_mismatch_positional": [[r, l, c] for (r, l), c in tokc.most_common(40)]}
json.dump(out, open(os.path.join(ROOT, "plan/reference/ref_quality.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != "top_mismatch_positional"}, indent=1))
print(out["top_mismatch_positional"][:40])
