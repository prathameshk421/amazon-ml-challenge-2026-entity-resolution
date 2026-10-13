"""Phase 1b: signal separation on fit-part sample.
Positives = all true pairs of 12K sampled fit entities (6K US + 6K India, seed 42).
Random negs = 20/positive from disjoint 400K pool reservoir.
Hard negs = up to 20/entity from 400K pool sample (norm-equal | token_set>=85 | top-10).
Writes artifacts/metrics/eda_signals.json + artifacts/eda_pairs_sample.parquet.
Run in background: nohup ./.venv/bin/python legacy/src/eda_signals.py > logs/eda_signals.log 2>&1 &
"""

import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name, normalize_address, postal_tokens
from editex import editex_sim
import jellyfish

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
MET = os.path.join(ROOT, "artifacts/metrics")
ART = os.path.join(ROOT, "artifacts")
SEED = 42
N_ENT = 6000  # per country

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)

rng = random.Random(SEED)
split = json.load(open(os.path.join(ROOT, "artifacts/split.json")))
fit_ids = set(split["fit_ids"])

# ---- 1. sample entities: need country + singleton -> stream S1 + GT singleton flags
log("loading S1 country map (fit only)")
s1country = {}
with open(os.path.join(TR, "train_source1.tsv"), newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["entity_id"] in fit_ids:
            s1country[row["entity_id"]] = row["country"]
log("fit S1 with country: %d" % len(s1country))
sing = {}
with open(os.path.join(TR, "train_ground_truth.tsv"), newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        e = row["source1_entity_id"]
        if e in fit_ids:
            sing[e] = (row["matched_entity_ids"] == "")
by_country = defaultdict(list)
for e, c in s1country.items():
    if not sing.get(e, True):
        by_country[c].append(e)
sample_ents = []
for c in ["US", "India"]:
    ids = sorted(by_country[c])
    rng.shuffle(ids)
    sample_ents.extend(ids[:N_ENT])
sample_set = set(sample_ents)
log("sampled entities: %d" % len(sample_ents))

# ---- 2. S1 records + truth for sample
s1rec = {}
with open(os.path.join(TR, "train_source1.tsv"), newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["entity_id"] in sample_set:
            s1rec[row["entity_id"]] = (row["business_name"], row["business_address"], row["country"])
truth = {}
with open(os.path.join(TR, "train_ground_truth.tsv"), newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["source1_entity_id"] in sample_set:
            m = row["matched_entity_ids"]
            truth[row["source1_entity_id"]] = m.split(",") if m else []
n_pos = sum(len(v) for v in truth.values())
log("positives: %d" % n_pos)

# ---- 3. pool pass: truth records + two disjoint 400K reservoirs
need = set()
for v in truth.values():
    need.update(v)
pool_truth = {}
mine, rand = [], []
swapped = sorted(need)
log("streaming pool (need %d truth recs)" % len(need))
r1, r2 = random.Random(SEED), random.Random(SEED + 1)
n_pool = 0
for fn, src in [("train_source2.tsv", "S2"), ("train_source3.tsv", "S3")]:
    with open(os.path.join(TR, fn), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            eid = row["entity_id"]
            rec = (eid, row["business_name"] or "", row["business_address"] or "",
                   row["country"] or "", src)
            if eid in need:
                pool_truth[eid] = rec
                continue
            n_pool += 1
            h = (hash(eid) & 0xFFFFFFF) % 100
            if h < 4:
                mine.append(rec)
            elif h < 8:
                rand.append(rec)
log("pool streamed; truth found %d/%d; mine=%d rand=%d" % (len(pool_truth), len(need), len(mine), len(rand)))
assert len(pool_truth) == len(need), "missing truth records!"

for r in mine:
    assert r[0] not in need
POOL = {r[0]: r for r in mine + rand}

# ---- 4. precompute normalized forms + phonetic code sets for universe
log("precomputing norms/codes")
strings = {}  # eid -> dict
def add_rec(eid, nm, ad, co):
    nn, na = normalize_name(nm), normalize_address(ad)
    strings[eid] = {"nm": nm, "ad": ad, "co": co, "nn": nn, "na": na,
                    "nnt": nn.split(), "nat": na.split(),
                    "meta": set(jellyfish.metaphone(t) for t in nn.split() if t),
                    "snd": set(jellyfish.soundex(t) for t in nn.split() if t and t[0].isalpha()),
                    "ameta": set(jellyfish.metaphone(t) for t in na.split() if t),
                    "asnd": set(jellyfish.soundex(t) for t in na.split() if t and t[0].isalpha()),
                    "post": set(postal_tokens(na)),
                    "digit": set(t for t in na.split() if t[:1].isdigit())}
for e, (nm, ad, co) in s1rec.items():
    add_rec(e, nm, ad, co)
for eid, rec in pool_truth.items():
    _, nm, ad, co, _ = rec
    add_rec(eid, nm, ad, co)
for r in mine + rand:
    eid, nm, ad, co, _ = r
    if eid not in strings:
        add_rec(eid, nm, ad, co)
log("universe strings: %d" % len(strings))

# ---- 5. hard negatives via trigram index on mine-pool names
def trigrams(s):
    s = " " + s + " "
    return [s[i:i + 3] for i in range(len(s) - 2)]
log("building trigram index (skip trigrams with df>8000)")
index = defaultdict(list)
df_tri = Counter()
mine_ids = [r[0] for r in mine]
tri_of = {}
for eid in mine_ids:
    tgs = set(trigrams(strings[eid]["nn"]))
    tri_of[eid] = tgs
    for tg in tgs:
        df_tri[tg] += 1
keep = set(t for t, d in df_tri.items() if d <= 8000)
for eid in mine_ids:
    for tg in tri_of[eid]:
        if tg in keep:
            index[tg].append(eid)
log("index trigrams: %d kept of %d" % (len(keep), len(df_tri)))
hard = defaultdict(list)
for _ie, e in enumerate(sample_ents):
    tset = set(truth[e])
    q = strings[e]
    cnt = Counter()
    for tg in set(trigrams(q["nn"])):
        if tg in keep:
            for c in index.get(tg, ()):
                cnt[c] += 1
    cands = [c for c, k in cnt.most_common(300) if k >= 2 and c not in tset and c != e]
    scored = []
    for c in cands:
        s = fuzz.token_set_ratio(q["nn"], strings[c]["nn"]) / 100.0
        scored.append((s, c))
    scored.sort(reverse=True)
    picked = []
    for s, c in scored:
        if strings[c]["nn"] == q["nn"] or s >= 0.85:
            picked.append((s, c))
            if len(picked) >= 20:
                break
    if len(picked) < 20:  # top-10 by similarity regardless of threshold
        have = set(c for _, c in picked)
        for s, c in scored[:10]:
            if c not in have:
                picked.append((s, c))
                if len(picked) >= 20:
                    break
    hard[e] = [c for _, c in picked[:20]]
    if (_ie + 1) % 2000 == 0:
        log("mined %d/%d entities" % (_ie + 1, len(sample_ents)))
n_hard = sum(len(v) for v in hard.values())
log("hard negatives: %d" % n_hard)

# ---- 6. random negatives 20/positive
rand_ids = [r[0] for r in rand]
rnd = []
for e in sample_ents:
    tset = set(truth[e])
    for _ in range(20 * len(tset)):
        for _try in range(10):
            c = rand_ids[rng.randrange(len(rand_ids))]
            if c not in tset:
                break
        rnd.append((e, c))
log("random negatives: %d" % len(rnd))

pairs = ([(e, c, 1) for e in sample_ents for c in truth[e]] +
         [(e, c, 0) for e, c in rnd] +
         [(e, c, 2) for e in sample_ents for c in hard[e]])
log("total pairs: %d" % len(pairs))

# ---- 7. signals
def jacc(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

SIGS = ["n_lev_raw", "n_lev_norm", "n_jacc", "n_tfidf", "n_meta", "n_snd", "n_editex",
        "a_lev_raw", "a_lev_norm", "a_jacc", "a_tfidf", "a_meta", "a_snd", "a_editex",
        "a_postal", "a_digit", "country"]
X = {s: np.zeros(len(pairs), dtype=np.float32) for s in SIGS}
for i, (e, c, y) in enumerate(pairs):
    q, r = strings[e], strings[c]
    X["n_lev_raw"][i] = fuzz.ratio(q["nm"], r["nm"]) / 100.0
    X["n_lev_norm"][i] = fuzz.ratio(q["nn"], r["nn"]) / 100.0
    X["n_jacc"][i] = jacc(q["nnt"], r["nnt"])
    X["n_meta"][i] = jacc(q["meta"], r["meta"])
    X["n_snd"][i] = jacc(q["snd"], r["snd"])
    X["a_lev_raw"][i] = fuzz.ratio(q["ad"], r["ad"]) / 100.0
    X["a_lev_norm"][i] = fuzz.ratio(q["na"], r["na"]) / 100.0
    X["a_jacc"][i] = jacc(q["nat"], r["nat"])
    X["a_meta"][i] = jacc(q["ameta"], r["ameta"])
    X["a_snd"][i] = jacc(q["asnd"], r["asnd"])
    X["a_postal"][i] = 1.0 if (q["post"] & r["post"]) else (1.0 if not q["post"] and not r["post"] else 0.0)
    X["a_digit"][i] = jacc(q["digit"], r["digit"])
    X["country"][i] = 1.0 if q["co"] == r["co"] else 0.0
log("base signals done")
Y = np.array([y for _, _, y in pairs])
E = np.array([e for e, _, _ in pairs])
C = np.array([c for _, c, _ in pairs])

# TF-IDF char n-gram cosine (fit on mine pool, transform in chunks)
def tfidf_cos(texts_q, texts_r, fit_texts, label):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=100000)
    vec.fit(fit_texts)
    out = np.zeros(len(texts_q), dtype=np.float32)
    BS = 100000
    for s in range(0, len(texts_q), BS):
        Q = vec.transform(texts_q[s:s + BS])
        R = vec.transform(texts_r[s:s + BS])
        out[s:s + BS] = np.asarray(Q.multiply(R).sum(axis=1)).ravel()
        log("%s tfidf chunk %d/%d" % (label, s // BS + 1, (len(texts_q) + BS - 1) // BS))
    return np.clip(out, 0, 1)
Qn = [strings[e]["nn"] for e, _, _ in pairs]
Rn = [strings[c]["nn"] for _, c, _ in pairs]
X["n_tfidf"] = tfidf_cos(Qn, Rn, [strings[i]["nn"] for i in mine_ids], "name")
Qa = [strings[e]["na"] for e, _, _ in pairs]
Ra = [strings[c]["na"] for _, c, _ in pairs]
X["a_tfidf"] = tfidf_cos(Qa, Ra, [strings[i]["na"] for i in mine_ids], "addr")

# editex on seeded subsample (15K per class for speed)
sub_idx = {"pos": [], "rnd": [], "hard": []}
for i, y in enumerate(Y):
    k = "pos" if y == 1 else ("rnd" if y == 0 else "hard")
    sub_idx[k].append(i)
editex_sub = {}
for k, idx in sub_idx.items():
    sel = sorted(rng.sample(idx, min(15000, len(idx))))
    editex_sub[k] = sel
    nvals = np.zeros(len(sel), dtype=np.float32)
    avals = np.zeros(len(sel), dtype=np.float32)
    for j, i in enumerate(sel):
        e, c, _ = pairs[i]
        nvals[j] = editex_sim(strings[e]["nn"], strings[c]["nn"])
        avals[j] = editex_sim(strings[e]["na"], strings[c]["na"])
    X["n_editex"][np.array(sel)] = nvals
    X["a_editex"][np.array(sel)] = avals
log("editex subsample done")

# ---- 8. metrics: AUC + histogram overlap, matches vs random / vs hard
def overlap(a, b):
    ha, _ = np.histogram(np.clip(a, 0, 1), bins=50, range=(0, 1))
    hb, _ = np.histogram(np.clip(b, 0, 1), bins=50, range=(0, 1))
    ha = ha / ha.sum() if ha.sum() else ha
    hb = hb / hb.sum() if hb.sum() else hb
    return float(np.minimum(ha, hb).sum())

m_pos = Y == 1
m_rnd = Y == 0
m_hrd = Y == 2
res = {}
for s in SIGS:
    v = X[s]
    if s in ("n_editex", "a_editex"):
        idx_p = np.array(editex_sub["pos"])
        idx_r = np.array(editex_sub["rnd"])
        idx_h = np.array(editex_sub["hard"])
        vp, vr, vh = v[idx_p], v[idx_r], v[idx_h]
    else:
        vp, vr, vh = v[m_pos], v[m_rnd], v[m_hrd]
    res[s] = {
        "mean_pos": float(vp.mean()), "mean_rand": float(vr.mean()), "mean_hard": float(vh.mean()),
        "auc_rand": float(roc_auc_score([1] * len(vp) + [0] * len(vr), list(vp) + list(vr))),
        "overlap_rand": overlap(vp, vr),
        "auc_hard": float(roc_auc_score([1] * len(vp) + [0] * len(vh), list(vp) + list(vh))),
        "overlap_hard": overlap(vp, vh),
    }
# phonetic by country
eco = np.array([s1rec[e][2] for e in E])
for s in ["n_meta", "n_snd", "n_editex"]:
    for co in ["US", "India"]:
        m = eco == co
        vp = X[s][m_pos & m] if s != "n_editex" else X[s][np.array([i for i in editex_sub["pos"] if eco[i] == co])]
        vh = X[s][m_hrd & m] if s != "n_editex" else X[s][np.array([i for i in editex_sub["hard"] if eco[i] == co])]
        res["%s|%s" % (s, co)] = {
            "mean_pos": float(vp.mean()), "mean_hard": float(vh.mean()),
            "auc_hard": float(roc_auc_score([1] * len(vp) + [0] * len(vh), list(vp) + list(vh))),
            "overlap_hard": overlap(vp, vh)}
# country agreement on positives
agree = X["country"][m_pos]
ca = {"agree_frac": float(agree.mean()),
      "differ_frac": float((agree == 0).mean()),
      "n_pos": int(m_pos.sum())}
# transposition analysis: positives with identical token multiset but different strings
trans = 0
tj, tl = [], []
for i in np.where(m_pos)[0]:
    e, c, _ = pairs[i]
    q, r = strings[e], strings[c]
    if sorted(q["nnt"]) == sorted(r["nnt"]) and q["nn"] != r["nn"]:
        trans += 1
        tj.append(X["n_jacc"][i])
        tl.append(X["n_lev_norm"][i])
ca["transposelike_pos"] = trans
ca["transposelike_frac"] = trans / int(m_pos.sum())
if tj:
    ca["trans_jacc_mean"] = float(np.mean(tj))
    ca["trans_lev_mean"] = float(np.mean(tl))
ranked = sorted([s for s in SIGS], key=lambda s: (-res[s]["auc_hard"], res[s]["overlap_hard"]))
out = {"signals": res, "ranked_by_hard_auc": ranked, "country_agree": ca,
       "n_pos": int(m_pos.sum()), "n_rand": int(m_rnd.sum()), "n_hard": int(m_hrd.sum()),
       "editex_note": "editex computed on seeded 15K/class subsample only",
       "mine_pool": len(mine), "rand_pool": len(rand)}
with open(os.path.join(MET, "eda_signals.json"), "w") as f:
    json.dump(out, f, indent=1)
log("WROTE eda_signals.json")

# save scored sample pairs (for Phase 1c catalog examples)
df = pd.DataFrame({
    "s1": E, "cand": C, "label": Y,
    "s1_country": [strings[e]["co"] for e in E],
    "s1_name": [strings[e]["nm"] for e in E],
    "s1_addr": [strings[e]["ad"] for e in E],
    "c_name": [strings[c]["nm"] for c in C],
    "c_addr": [strings[c]["ad"] for c in C],
    "c_country": [strings[c]["co"] for c in C],
    **{s: X[s] for s in SIGS if s not in ("n_editex", "a_editex")}})
df.to_parquet(os.path.join(ART, "eda_pairs_sample.parquet"))
log("WROTE eda_pairs_sample.parquet (%d rows)" % len(df))
for s in ranked:
    r = res[s]
    log("%-12s auc_hard=%.4f ov_hard=%.3f auc_rand=%.4f pos=%.3f hard=%.3f rand=%.3f"
        % (s, r["auc_hard"], r["overlap_hard"], r["auc_rand"], r["mean_pos"], r["mean_hard"], r["mean_rand"]))
log("country agree=%.4f trans_frac=%.4f" % (ca["agree_frac"], ca["transposelike_frac"]))
