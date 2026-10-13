"""Bucketing diagnostic B: where do true matches sit in the FULL (untruncated) ranked
union of the current blocker, why are the rest unreachable, and would cheap
re-weightings or new keys fix it.

Queries: the same 10K fit-part S1 entities (seed 123) used by block_query2.py
(fit part only; val and test untouched). For every true (S1, pool) pair it records:
  - rank under several scoring formulas over the full union (1 = best; ties
    counted pessimistically: rank = 1 + #candidates with score >= true score - 1)
  - which channels voted (EX exact, NT name token, TR trigram, AD addr token, PH)
  - truth-record attributes: name/addr writing-system class, normalized-empty flags,
    S1 chain size (# S1 entities with the same normalized name), country
For pairs with NO channel vote (unreachable) it tests candidate new keys:
  romanized name (legacy/src/translit.py) token/trigram overlap, space-free concatenated
  name (domain-style), postal token equality.
Writes artifacts/metrics/diag_blocking.json and artifacts/diag_pairs.parquet.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name, normalize_address, postal_tokens, strip_suffix_tokens
from block_query2 import Q, TRI_DF_CAP
from diag_scripts import classify
from translit import romanize_words_schwa
import jellyfish

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
IDX = os.path.join(ROOT, "artifacts/index")
MET = os.path.join(ROOT, "artifacts/metrics")
SEED = 123
NQ = int(os.environ.get("NQ", "10000"))
KS = [25, 50, 100, 150, 250, 500, 1000, 2000, 5000]

t0 = time.time()
def log(m):
    print("[%.1fmin] %s" % ((time.time() - t0) / 60, m), flush=True)


def union_scores(q, nn, na):
    """Full union with separate channel scores (no truncation)."""
    z = q.z
    P, W, C = [], [], []   # postings, weight, channel id 0..4
    def add(a, w, ch):
        P.append(a)
        W.append(np.full(len(a), w, dtype=np.float32))
        C.append(np.full(len(a), ch, dtype=np.uint8))
    qself = np.zeros(5)
    if nn in q.exact:
        add(q.exact[nn], 1.0, 0)
        qself[0] = 1.0
    for t in set(nn.split()):
        r = q.range_of(z, "tok", t)
        if r:
            s, e, j = r
            add(z["tok_data"][s:e], float(q.tok_idf[j]), 1)
            qself[1] += float(q.tok_idf[j])
    tgs = set((" " + nn + " ")[i:i + 3] for i in range(len(nn))) if nn else set()
    for tg in tgs:
        r = q.range_of(z, "tri", tg)
        if r:
            s, e, j = r
            if q.tri_df[j] > TRI_DF_CAP:
                continue
            add(z["tri_data"][s:e], float(q.tri_idf[j]), 2)
            qself[2] += float(q.tri_idf[j])
    for t in set(na.split()):
        r = q.range_of(q.a, "a", t)
        if r:
            s, e, j = r
            add(q.a["a_data"][s:e], float(q.a_idf[j]), 3)
            qself[3] += float(q.a_idf[j])
    for t in set(nn.split()):
        if t:
            r = q.range_of(z, "met", jellyfish.metaphone(t))
            if r:
                s, e, j = r
                if q.met_df[j] > 20000:
                    continue
                add(z["met_data"][s:e], 1.0, 4)
                qself[4] += 1.0
    if not P:
        return None
    Pall = np.concatenate(P)
    Wall = np.concatenate(W)
    Call = np.concatenate(C)
    u, inv = np.unique(Pall, return_inverse=True)
    S = np.zeros((5, len(u)), dtype=np.float32)
    for ch in range(5):
        m = Call == ch
        if m.any():
            S[ch] = np.bincount(inv[m], weights=Wall[m], minlength=len(u))
    return u, S, qself


def formulas(S, qself):
    ex, nt, tr, ad, ph = S
    name_cur = 100 * ex + nt + tr + 0.2 * ph
    out = {"F0_current": name_cur + ad,
           "F1_exact10": 10 * ex + nt + tr + 0.2 * ph + ad}
    # F2: per-channel self-normalized (query-side coverage in [0,1]) sum
    def nz(x, d):
        return x / d if d > 0 else np.zeros_like(x)
    ncov = 0.5 * nz(nt, qself[1]) + 0.5 * nz(tr, qself[2])
    acov = nz(ad, qself[3])
    out["F2_cov_sum"] = ncov + acov + 0.3 * ex
    # F3: reciprocal-rank fusion of a name ranking and an address ranking
    def rr(x):
        o = np.argsort(-x, kind="stable")
        r = np.empty(len(x), dtype=np.float64)
        r[o] = np.arange(1, len(x) + 1)
        r[x <= 0] = 1e9
        return 1.0 / (60 + r)
    out["F3_rrf"] = rr(name_cur) + rr(ad)
    # F4: coverage max+min blend: rewards candidates good on BOTH axes, keeps
    # candidates excellent on one axis
    out["F4_cov_maxmin"] = np.maximum(ncov, acov) + 0.5 * np.minimum(ncov, acov) + 0.3 * ex
    return out


def trigram_set(s):
    s = " " + s + " "
    return set(s[i:i + 3] for i in range(len(s) - 2))


def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def main():
    rng = random.Random(SEED)
    split = json.load(open(os.path.join(ROOT, "artifacts/split.json")))
    qids = rng.sample(split["fit_ids"], 10000)[:NQ]
    want = set(qids)
    recs, chain = {}, Counter()
    with open(os.path.join(TR, "train_source1.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            nn = normalize_name(row["business_name"])
            chain[nn] += 1
            if row["entity_id"] in want:
                recs[row["entity_id"]] = (row["business_name"], row["business_address"],
                                          row["country"], nn)
    truth = {}
    with open(os.path.join(TR, "train_ground_truth.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["source1_entity_id"] in want:
                m = row["matched_entity_ids"]
                truth[row["source1_entity_id"]] = m.split(",") if m else []
    log("loaded queries; mapping truth")
    src = np.load(os.path.join(IDX, "pool.src.npy"))
    num = np.load(os.path.join(IDX, "pool.num.npy"))
    key = (src.astype(np.int64) << 32) | num
    order = np.argsort(key)
    skey = key[order]
    tro = {}
    for e, ids in truth.items():
        rows = []
        for x in ids:
            s = 0 if x.startswith("S2-") else 1
            k = (np.int64(s) << np.int64(32)) | np.int64(int(x.split("-", 1)[1]))
            j = int(np.searchsorted(skey, k))
            if j < len(skey) and skey[j] == k:
                rows.append(int(order[j]))
        tro[e] = rows
    del key, order, skey
    nblob = np.memmap(os.path.join(IDX, "pool.names.bin"), mode="r", dtype=np.uint8)
    noff = np.load(os.path.join(IDX, "pool.names.off.npy"), mmap_mode="r")
    ablob = np.memmap(os.path.join(IDX, "pool.addrs.bin"), mode="r", dtype=np.uint8)
    aoff = np.load(os.path.join(IDX, "pool.addrs.off.npy"), mmap_mode="r")
    def pname(x):
        return bytes(nblob[noff[x]:noff[x + 1]]).decode("utf-8", "replace")
    log("loading index")
    q = Q()
    log("index loaded")
    rows = []
    usizes = []
    qtime = 0.0
    for qi, e in enumerate(qids):
        name, addr, country, nn = recs[e]
        na = normalize_address(addr)
        t1 = time.time()
        res = union_scores(q, nn, na)
        fm = formulas(res[1], res[2]) if res is not None else None
        qtime += time.time() - t1
        usizes.append(0 if res is None else len(res[0]))
        ts = tro[e]
        if not ts:
            continue
        if res is not None:
            u = res[0]
            pos = np.searchsorted(u, ts)
            pos = np.minimum(pos, len(u) - 1)
            hit = u[pos] == np.asarray(ts)
        for k, x in enumerate(ts):
            pn = pname(x)
            pa = bytes(ablob[aoff[x]:aoff[x + 1]]).decode("utf-8", "replace")
            cn, sn = classify(pn)
            ca, _ = classify(pa)
            d = {"s1": e, "row": x, "country": country, "chain": chain[nn],
                 "n_truth": len(ts), "name_cls": cn, "name_script": sn, "addr_cls": ca,
                 "pname_norm_empty": int(bool(pn) and not normalize_name(pn)),
                 "paddr_norm_empty": int(not normalize_address(pa)),
                 "union": usizes[-1]}
            if res is not None and hit[k]:
                j = int(pos[k])
                chans = res[1][:, j]
                d["reach"] = 1
                for ci, cname in enumerate(["EX", "NT", "TR", "AD", "PH"]):
                    d["v_" + cname] = float(chans[ci])
                for fname, sc in fm.items():
                    d["rank_" + fname] = int((sc > sc[j]).sum() + (sc == sc[j]).sum())
            else:
                d["reach"] = 0
                # candidate new keys for unreachable pairs
                pnn = normalize_name(pn)
                rom = normalize_name(romanize_words_schwa(pn))
                s1t = set(nn.split())
                d["k_rom_tok"] = int(bool(s1t & set(rom.split())))
                d["k_rom_tri_jac"] = round(jac(trigram_set(nn), trigram_set(rom)), 3)
                d["k_rom_meta"] = int(bool({jellyfish.metaphone(t) for t in s1t if t} &
                                           {jellyfish.metaphone(t) for t in rom.split() if t}))
                cat1 = strip_suffix_tokens(nn).replace(" ", "")
                ptoks = pnn.split()
                d["k_concat"] = int(bool(cat1) and len(cat1) >= 6 and any(
                    (cat1 in t or (len(t) >= 6 and t in cat1)) for t in ptoks))
                d["k_postal"] = int(bool(set(postal_tokens(na)) &
                                         set(postal_tokens(normalize_address(pa)))))
                d["k_name_tri_jac"] = round(jac(trigram_set(nn), trigram_set(pnn)), 3)
                d["ex_s1"] = name[:60]
                d["ex_pool"] = pn[:60]
                d["ex_s1a"] = addr[:60]
                d["ex_poola"] = pa[:60]
            rows.append(d)
        if (qi + 1) % 1000 == 0:
            log("q %d/%d  avg union %.0f  %.1f ms/q" % (qi + 1, NQ, np.mean(usizes),
                                                      qtime / (qi + 1) * 1000))
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(ROOT, "artifacts/diag_pairs.parquet"))
    out = {"n_queries": NQ, "n_true_pairs": int(len(df)),
           "n_nonsingleton_entities": int(df["s1"].nunique()),
           "union_size": {"mean": float(np.mean(usizes)),
                          "p50": float(np.percentile(usizes, 50)),
                          "p90": float(np.percentile(usizes, 90)),
                          "p99": float(np.percentile(usizes, 99)),
                          "max": float(np.max(usizes))},
           "ms_per_query_full_union": qtime / NQ * 1000,
           "reachable_pair_frac": float(df["reach"].mean())}
    fn = [c for c in df.columns if c.startswith("rank_")]
    def rec_table(sub):
        r = {}
        for c in fn:
            rk = sub[c].fillna(10 ** 12)
            pr = {str(K): round(float((rk <= K).mean()), 4) for K in KS}
            ent = sub.assign(h=rk).groupby("s1")["h"].min()
            er = {str(K): round(float((ent <= K).mean()), 4) for K in KS}
            r[c[5:]] = {"pair": pr, "entity": er}
        return r
    out["recall_all"] = rec_table(df)
    segs = {}
    df["chain_b"] = pd.cut(df["chain"], [0, 1, 9, 10 ** 9], labels=["1", "2-9", "10+"]).astype(str)
    for col in ["chain_b", "country", "name_cls", "addr_cls"]:
        for v, sub in df.groupby(col):
            if len(sub) >= 50:
                segs["%s=%s" % (col, v)] = {"n": int(len(sub)),
                                            "reach": round(float(sub["reach"].mean()), 4),
                                            "F0_current": rec_table(sub)["F0_current"]["pair"]}
    out["segments"] = segs
    um = df[df["reach"] == 0]
    out["unreachable"] = {
        "n": int(len(um)),
        "by_name_cls": um["name_cls"].value_counts().to_dict(),
        "by_name_script": um["name_script"].value_counts().head(10).to_dict(),
        "by_addr_cls": um["addr_cls"].value_counts().to_dict(),
        "pname_norm_empty": int(um["pname_norm_empty"].sum()),
        "paddr_norm_empty": int(um["paddr_norm_empty"].sum()),
        "new_key_hits": {k: int(um[k].sum()) for k in ["k_rom_tok", "k_rom_meta", "k_concat", "k_postal"]},
        "rom_tri_jac_ge_0.3": int((um["k_rom_tri_jac"] >= 0.3).sum()),
        "name_tri_jac_ge_0.3": int((um["k_name_tri_jac"] >= 0.3).sum()),
        "any_new_key": int(((um["k_rom_tok"] + um["k_rom_meta"] + um["k_concat"] + um["k_postal"]) > 0).sum()),
    }
    reach = df[df["reach"] == 1]
    ch = ["v_EX", "v_NT", "v_TR", "v_AD", "v_PH"]
    voted = reach[ch] > 0
    out["reachable_channel_votes"] = {c[2:]: round(float(voted[c].mean()), 4) for c in ch}
    out["reachable_name_silent_addr_only"] = int((~voted[["v_EX", "v_NT", "v_TR", "v_PH"]].any(axis=1)).sum())
    json.dump(out, open(os.path.join(MET, "diag_blocking.json"), "w"), indent=1, ensure_ascii=False)
    log("WROTE diag_blocking.json + artifacts/diag_pairs.parquet")


if __name__ == "__main__":
    main()
