"""Critique job D: which blocking evidence actually exists for true pairs, once df caps are applied.
 1. Full-train-pool document frequencies: NT (tokens len>=2), SK (skeletons len>=2), AT (address tokens).
 2. Romanization quality per Indic script over ALL true pairs whose pool name has Indic letters
    (raw vs 'uncapped' = shared key with df <= cap_frac*n_pool, i.e. usable by the planned index).
 3. The §5.3 learned dictionary re-implemented on a random 80% of S1 entities, evaluated on the other 20%;
    count entries supported by a single S1 entity.
 4. Evidence per true pair on a 20K-entity S1 sample (all scripts): EN, NT/SK/NP/AT uncapped, and a
    conjunctive cross key (name skeleton x uncapped address token) with its df.
Output: plan/critique/out/d_evidence.json (+ d_examples.json)
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq
from rapidfuzz.distance import JaroWinkler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "plan", "reference"))
from translit_ref import skeleton  # noqa: E402

OUT = os.path.join(ROOT, "plan", "critique", "out")
GT = os.path.join(ROOT, "student_resource", "dataset", "train", "train_ground_truth.tsv")
R, EX = {}, {}
T0 = time.time()


def log(*a):
    print(round(time.time() - T0, 1), *a, flush=True)


def load(part, cols):
    return pq.read_table(os.path.join(OUT, f"norm_{part}.parquet"), columns=cols)


cols = ["src", "num", "name_norm", "addr_norm", "had_indic", "script", "country", "addr_state", "name_raw"]
t2, t3 = load("train_pool_s2", cols), load("train_pool_s3", cols)
pkey = np.concatenate([(t.column("src").to_numpy().astype(np.int64) << 32) | t.column("num").to_numpy().astype(np.int64) for t in (t2, t3)])
p_name = t2["name_norm"].to_pylist() + t3["name_norm"].to_pylist()
p_addr = t2["addr_norm"].to_pylist() + t3["addr_norm"].to_pylist()
import pyarrow as pa
p_raw = pa.chunked_array(t2["name_raw"].chunks + t3["name_raw"].chunks)
p_indic = np.concatenate([t["had_indic"].to_numpy(zero_copy_only=False) for t in (t2, t3)])
p_script = pa.chunked_array(t2["script"].chunks + t3["script"].chunks)
p_astate = np.concatenate([t["addr_state"].to_numpy() for t in (t2, t3)])
del t2, t3
n_pool = len(pkey)
CAP = int(0.002 * n_pool)
R["n_pool"], R["cap_0.002"] = n_pool, CAP
log("pool loaded", n_pool)

SKC = {}


def sk(t):
    v = SKC.get(t)
    if v is None:
        v = skeleton(t)
        SKC[t] = v
    return v


# ---- 1. df ----
df_nt, df_sk, df_at = Counter(), Counter(), Counter()
for nm, ad in zip(p_name, p_addr):
    toks = {t for t in nm.split() if len(t) >= 2}
    df_nt.update(toks)
    df_sk.update({s for s in (sk(t) for t in toks) if len(s) >= 2})
    df_at.update(set(ad.split()))
log("df done", len(df_nt), len(df_sk), len(df_at))
R["n_distinct"] = {"NT": len(df_nt), "SK": len(df_sk), "AT": len(df_at)}
R["n_capped_keys"] = {"NT": sum(1 for v in df_nt.values() if v > CAP), "SK": sum(1 for v in df_sk.values() if v > CAP),
                      "AT": sum(1 for v in df_at.values() if v > CAP)}
R["top_SK_df"] = [[k, v] for k, v in df_sk.most_common(25)]
R["top_AT_df"] = [[k, v] for k, v in df_at.most_common(15)]

# ---- truth ----
order = np.argsort(pkey)
pks = pkey[order]
s1 = load("train_s1", ["num", "name_norm", "addr_norm", "country"])
s1_num = s1["num"].to_numpy().astype(np.int64)
s1_name = s1["name_norm"].to_pylist()
s1_addr = s1["addr_norm"].to_pylist()
s1_country = s1["country"].to_pylist()
n_s1 = len(s1_num)
s1_pos = {int(n): i for i, n in enumerate(s1_num.tolist())}
truth = defaultdict(list)
with open(GT, encoding="utf-8", newline="") as f:
    next(f)
    for line in f:
        a, b = line.rstrip("\n").split("\t")
        if b:
            i = s1_pos[int(a[3:])]
            ks = np.array([(int(x[1]) << 32) | int(x[3:]) for x in b.split(",")], np.int64)
            truth[i] = order[np.searchsorted(pks, ks)].tolist()
log("truth loaded", len(truth))
s1_cnt = Counter(s1_name)
rng = np.random.default_rng(7)
hold = rng.random(n_s1) < 0.2  # same draw as job B


def usable(toks, df):
    return {t for t in toks if df.get(t, 0) <= CAP}


# ---- 2. romanization per script (all Indic true pairs) ----
per = defaultdict(lambda: Counter())
fail_examples = defaultdict(list)
dict_pairs_fit, dict_pairs_hold = [], []
for i, plist in truth.items():
    stoks = [t for t in s1_name[i].split() if len(t) >= 2]
    sset = set(stoks)
    sks = {sk(t) for t in sset if len(sk(t)) >= 2}
    atq = set(s1_addr[i].split())
    for p in plist:
        if not p_indic[p]:
            continue
        sc = p_script[p].as_py() or "?"
        ptoks = [t for t in p_name[p].split() if len(t) >= 2]
        pset = set(ptoks)
        psk = {sk(t) for t in pset if len(sk(t)) >= 2}
        c = per[sc]
        c["n"] += 1
        te = sset & pset
        se = sks & psk
        c["tok_eq"] += bool(te)
        c["sk_eq"] += bool(se)
        c["tok_eq_uncapped"] += bool(usable(te, df_nt))
        c["sk_eq_uncapped"] += bool(usable(se, df_sk))
        ae = usable(atq & set(p_addr[p].split()), df_at)
        c["addr_uncapped"] += bool(ae)
        name_ev = bool(usable(te, df_nt)) or bool(usable(se, df_sk)) or s1_name[i] == p_name[p]
        c["name_uncapped_any"] += name_ev
        c["silent_name_and_addr"] += (not name_ev) and (not ae)
        c["addr_empty_or_null"] += p_astate[p] > 0
        if (not name_ev) and len(fail_examples[sc]) < 6 and rng.random() < 0.05:
            fail_examples[sc].append({"s1": s1_name[i], "pool_raw": p_raw[p].as_py(), "pool_rom": p_name[p],
                                      "shared_generic_skeletons": sorted(se)})
        (dict_pairs_hold if hold[i] else dict_pairs_fit).append((i, ptoks, stoks))
R["romanization_by_script"] = {sc: {k: (v if k == "n" else round(v / c["n"], 4)) for k, v in c.items()}
                               for sc, c in sorted(per.items(), key=lambda x: -x[1]["n"])}
tot = Counter()
for c in per.values():
    tot.update(c)
R["romanization_all"] = {k: (v if k == "n" else round(v / tot["n"], 4)) for k, v in tot.items()}
EX["romanization_fail_examples"] = fail_examples
log("romanization done", tot["n"])

# ---- 3. dictionary per §5.3 on fit80, evaluate on hold20 ----
c_r, c_rl, ents = Counter(), Counter(), defaultdict(set)
for i, ptoks, stoks in dict_pairs_fit:
    sset = set(stoks)
    for r in dict.fromkeys(ptoks):
        if r in sset:
            continue
        c_r[r] += 1
        if not stoks:
            continue
        best, bs = None, -1.0
        for l in stoks:
            s = JaroWinkler.normalized_similarity(r, l)
            if s > bs:
                best, bs = l, s
        if bs >= 0.60:
            c_rl[(r, best)] += 1
            if len(ents[(r, best)]) < 2:
                ents[(r, best)].add(i)
tdict = {}
for (r, l), v in c_rl.items():
    if v >= 5 and v / c_r[r] >= 0.80:
        tdict[r] = l
single_ent = sum(1 for (r, l) in c_rl if r in tdict and tdict[r] == l and len(ents[(r, l)]) == 1)
gen = sum(1 for r, l in tdict.items() if df_nt.get(l, 0) > CAP)
R["dictionary_fit80"] = {"n_pairs_fit": len(dict_pairs_fit), "n_entries": len(tdict),
                         "entries_supported_by_one_s1_entity": single_ent,
                         "entries_whose_target_token_is_capped(generic)": gen}
EX["dict_single_entity_entries"] = [[r, l] for (r, l) in list(c_rl) if r in tdict and tdict[r] == l and len(ents[(r, l)]) == 1][:30]
EX["dict_entries_sample"] = sorted(tdict.items(), key=lambda kv: -c_rl[kv])[:40]
cnt = Counter()
for i, ptoks, stoks in dict_pairs_hold:
    sset = set(stoks)
    p0 = set(ptoks)
    p1 = {tdict.get(t, t) for t in ptoks}
    cnt["n"] += 1
    cnt["tok_eq_nodict"] += bool(sset & p0)
    cnt["tok_eq_dict"] += bool(sset & p1)
    cnt["tok_eq_uncapped_nodict"] += bool(usable(sset & p0, df_nt))
    cnt["tok_eq_uncapped_dict"] += bool(usable(sset & p1, df_nt))
R["dictionary_eval_hold20"] = {k: (v if k == "n" else round(v / cnt["n"], 4)) for k, v in cnt.items()}
del dict_pairs_fit, dict_pairs_hold
log("dictionary done", len(tdict))

# ---- 4. evidence on a 20K S1 sample (all pairs) + cross keys ----
samp = np.random.default_rng(11).choice(n_s1, 20000, replace=False)
samp = [int(i) for i in samp]


def first8(toks):
    out = []
    for t in toks:
        if len(t) >= 2 and t not in out:
            out.append(t)
            if len(out) == 8:
                break
    return out


def pairs(items):
    return {(a + "|" + b if a <= b else b + "|" + a) for k, a in enumerate(items) for b in items[k + 1:]}


def xkeys(name, addr):
    sks = {sk(t) for t in name.split() if len(t) >= 2}
    sks = {s for s in sks if len(s) >= 2}
    ats = {a for a in addr.split() if df_at.get(a, 0) <= CAP}
    return {s + "#" + a for s in sks for a in ats}


need_np, need_x = set(), set()
qinfo = {}
for i in samp:
    f8 = first8(s1_name[i].split())
    q_np = pairs(f8)
    q_x = xkeys(s1_name[i], s1_addr[i])
    need_np |= q_np
    need_x |= q_x
    qinfo[i] = (q_np, q_x)
df_np, df_x = Counter(), Counter()
for nm, ad in zip(p_name, p_addr):
    f8 = first8(nm.split())
    if len(f8) >= 2:
        for k in pairs(f8):
            if k in need_np:
                df_np[k] += 1
    for k in xkeys(nm, ad):
        if k in need_x:
            df_x[k] += 1
log("np/x df done", len(df_np), len(df_x))

segc = defaultdict(Counter)
xsize_all, xsize_chain = [], []
for i in samp:
    q_np, q_x = qinfo[i]
    q_np_u = {k for k in q_np if df_np.get(k, 0) <= CAP}
    xs = sum(df_x.get(k, 0) for k in q_x)
    xsize_all.append(xs)
    ch = s1_cnt[s1_name[i]]
    chs = "1" if ch == 1 else ("2-9" if ch < 10 else ">=10")
    if ch >= 10:
        xsize_chain.append(xs)
    qt = {t for t in s1_name[i].split() if len(t) >= 2}
    qsk = {sk(t) for t in qt if len(sk(t)) >= 2}
    qat = set(s1_addr[i].split())
    for p in truth.get(i, []):
        pt = {t for t in p_name[p].split() if len(t) >= 2}
        psk = {sk(t) for t in pt if len(sk(t)) >= 2}
        p_np = pairs(first8(p_name[p].split()))
        en = s1_name[i] == p_name[p] and s1_name[i] != ""
        nt = bool(usable(qt & pt, df_nt))
        skk = bool(usable(qsk & psk, df_sk))
        npk = bool(q_np_u & p_np)
        at = bool(usable(qat & set(p_addr[p].split()), df_at))
        at_rare = any(df_at.get(a, 0) <= 1000 for a in qat & set(p_addr[p].split()))
        xk = bool(q_x & xkeys(p_name[p], p_addr[p]))
        name_any = en or nt or skk or npk
        for s in ("all", "chain=" + chs, "country=" + s1_country[i],
                  "pool_indic" if p_indic[p] else "pool_latin",
                  "pool_addr_empty_or_null" if p_astate[p] > 0 else "pool_addr_present"):
            c = segc[s]
            c["n"] += 1
            c["EN"] += en
            c["NT_uncapped"] += nt
            c["SK_uncapped"] += skk
            c["NP_uncapped"] += npk
            c["AT_uncapped"] += at
            c["AT_df_le_1000"] += at_rare
            c["name_any_uncapped(EN|NT|SK|NP)"] += name_any
            c["name_silent_and_AT_uncapped"] += (not name_any) and at
            c["no_evidence_at_all(excl NG,PH,NC,AB)"] += (not name_any) and (not at)
            c["cross_key_SKxAT"] += xk
            c["EN_only_name_evidence"] += en and not (nt or skk or npk)
R["evidence_20k_sample"] = {s: {k: (v if k == "n" else round(v / c["n"], 4)) for k, v in c.items()}
                            for s, c in segc.items()}


def q(a):
    a = np.array(a, np.float64)
    return {"mean": float(a.mean()), "p50": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99))} if len(a) else {}


R["cross_key_union_upper_bound_sum_df"] = {"all_queries": q(xsize_all), "chain_ge10_queries": q(xsize_chain)}
log("evidence done")
json.dump(R, open(os.path.join(OUT, "d_evidence.json"), "w"), indent=1, default=str)
json.dump(EX, open(os.path.join(OUT, "d_examples.json"), "w"), indent=1, ensure_ascii=False, default=str)
log("written")
