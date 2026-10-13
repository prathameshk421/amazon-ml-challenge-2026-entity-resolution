"""M3 learned transliteration dictionary and its usable-token gate (plan §5.3; P2 order in §7).

Input texts are the no-dict normalizations (`normalize --split train --no-dict`, i.e.
normalize_field(..., tdict={})). Pairs are train true pairs; learning uses DICT_SRC, the gate DEV_A.
"""

import json
import math
import time
import unicodedata
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.distance import JaroWinkler

from ber import io_utils
from ber.splits import load_query_sets
from ber.translit import INDIC_HI, INDIC_LO

MIN_COUNT = 5
MIN_RATIO = 0.80
JW_MIN = 0.60
PASS_BATCH = 1_000_000


# ---------------------------------------------------------------- learner (pure)

def learn(pairs):
    """pairs: iterable of (pool_tokens, s1_tokens) -> {r: l} (plan §5.3).

    Each distinct pool token r (len >= 2) absent verbatim from the S1 tokens counts once per pair in
    c(r); its best S1 token l* by Jaro-Winkler (ties: first S1 token) counts in c(r, l*) if >= JW_MIN.
    Keep r -> l* iff c(r, l*) >= MIN_COUNT and c(r, l*) / c(r) >= MIN_RATIO.
    """
    c_r, c_rl = Counter(), Counter()
    for ptoks, stoks in pairs:
        sset = set(stoks)
        rs = [r for r in dict.fromkeys(ptoks) if len(r) >= 2 and r not in sset]
        if not rs:
            continue
        c_r.update(rs)
        if not stoks:
            continue
        for r in rs:
            best, bs = None, -1.0
            for l in stoks:
                s = JaroWinkler.normalized_similarity(r, l)
                if s > bs:  # strict: ties keep the first S1 token
                    best, bs = l, s
            if bs >= JW_MIN:
                c_rl[(r, best)] += 1
    return {r: l for (r, l), v in sorted(c_rl.items()) if v >= MIN_COUNT and v / c_r[r] >= MIN_RATIO}


# ---------------------------------------------------------------- gate (pure)

def name_tokens(text, tdict=None):
    """NT key set of a name: distinct tokens (after the dictionary, if given) with len >= 2."""
    toks = text.split()
    if tdict:
        toks = [tdict.get(t, t) for t in toks]
    return {t for t in toks if len(t) >= 2}


def usable_flags(pairs, df, cap):
    """pairs: (s1 token set, pool token set) -> bool array: share >= 1 token with df <= cap."""
    return np.array([any(df.get(t, 0) <= cap for t in s & p) for s, p in pairs], dtype=bool)


def shares(flags, scripts):
    """Usable-token share overall and per script."""
    flags, scripts = np.asarray(flags, dtype=bool), np.asarray(scripts, dtype=object)
    per = {}
    for sc in sorted(set(scripts.tolist())):
        m = scripts == sc
        per[sc] = {"n": int(m.sum()), "share": float(flags[m].mean())}
    overall = float(flags.mean()) if flags.size else float("nan")
    return {"n": int(flags.size), "overall": overall, "per_script": per}


def gate_decision(before, after):
    """Pass iff the overall share increases and no script's share decreases."""
    return bool(after["overall"] > before["overall"] and
                all(after["per_script"][sc]["share"] >= v["share"]
                    for sc, v in before["per_script"].items()))


def pool_df(names, indic, tdict, needed):
    """df over pool name token sets restricted to `needed`, before and after the dictionary
    (applied only to records with name_indic, plan §5.2 step 13). -> (df_before, df_after)."""
    before, after = Counter(), Counter()
    for name, ind in zip(names, indic):
        b = name_tokens(name) & needed
        if b:
            before.update(b)
        a = (name_tokens(name, tdict) & needed) if ind else b
        if a:
            after.update(a)
    return before, after


def first_indic_script(raw):
    """First word of unicodedata.name of the first Indic character (U+0900-U+0D7F), e.g. BENGALI."""
    for c in raw:
        if INDIC_LO <= ord(c) <= INDIC_HI:
            return unicodedata.name(c, "UNKNOWN").split()[0]
    return "UNKNOWN"


# ---------------------------------------------------------------- command

def _take(path, cols, idx):
    """Rows `idx` (row position == idx column per the data contract) of a parquet -> {col: list}."""
    tbl = pq.read_table(path, columns=cols).take(pa.array(idx))
    return {c: tbl[c].to_pylist() for c in cols}


def _bool_col(path, col):
    return pq.read_table(path, columns=[col])[col].to_numpy()


def run_learn_dict(args):
    """learn-dict: fit the dictionary on DICT_SRC, gate on DEV_A, write translit/dict_v1.json."""
    t0 = time.time()
    cfg = io_utils.load_config()
    log = io_utils.get_logger("learn-dict")
    meta = io_utils.read_json(io_utils.work_path("norm", "train_meta.json"))
    if meta["dict"] is not False:
        raise RuntimeError("norm/train_* were normalized with a dictionary; run "
                           "`normalize --split train --no-dict` first (plan §7 P2)")
    norm_s1 = io_utils.work_path("norm", "train_s1.parquet")
    norm_pool = io_utils.work_path("norm", "train_pool.parquet")
    sets = load_query_sets()
    gt = pq.read_table(io_utils.work_path("records", "train_gt.parquet"))
    g_s1 = gt["s1_idx"].to_numpy()
    g_pool = gt["pool_idx"].to_numpy()
    del gt
    p_name_indic = _bool_col(norm_pool, "name_indic")
    p_addr_indic = _bool_col(norm_pool, "addr_indic")
    n_pool = p_name_indic.size
    in_src = np.isin(g_s1, sets["DICT_SRC"])
    sel = {"name": in_src & p_name_indic[g_pool], "addr": in_src & p_addr_indic[g_pool],
           "gate": np.isin(g_s1, sets["DEV_A"]) & p_name_indic[g_pool]}
    del p_addr_indic
    any_sel = sel["name"] | sel["addr"] | sel["gate"]
    s1_ids, pool_ids = np.unique(g_s1[any_sel]), np.unique(g_pool[any_sel])
    s1 = _take(norm_s1, ["name_norm", "addr_norm", "name_indic"], s1_ids)
    pool = _take(norm_pool, ["name_norm", "addr_norm"], pool_ids)

    def pairs(kind):
        qs = np.searchsorted(s1_ids, g_s1[sel[kind]])
        ps = np.searchsorted(pool_ids, g_pool[sel[kind]])
        return qs, ps

    tdict, n_pairs = {}, {}
    for kind in ("name", "addr"):
        qs, ps = pairs(kind)
        col = f"{kind}_norm"
        n_pairs[kind] = int(qs.size)
        tdict[kind] = learn((pool[col][p].split(), s1[col][q].split()) for q, p in zip(qs, ps))
        log.info("%s: %d pairs, %d entries", kind, qs.size, len(tdict[kind]))

    # gate on DEV_A true pairs with an Indic pool name
    qs, ps = pairs("gate")
    raw = _take(io_utils.work_path("records", "train_pool.parquet"), ["name_raw"], pool_ids)["name_raw"]
    scripts = [first_indic_script(raw[p]) for p in ps]
    del raw
    d = tdict["name"]
    pairs_before = [(name_tokens(s1["name_norm"][q]), name_tokens(pool["name_norm"][p]))
                    for q, p in zip(qs, ps)]
    pairs_after = [(name_tokens(s1["name_norm"][q], d if s1["name_indic"][q] else None),
                    name_tokens(pool["name_norm"][p], d)) for q, p in zip(qs, ps)]
    needed = set().union(*(s & p for s, p in pairs_before + pairs_after))
    df_before, df_after = Counter(), Counter()
    for batch in pq.ParquetFile(norm_pool).iter_batches(batch_size=PASS_BATCH,
                                                        columns=["name_norm", "name_indic"]):
        b, a = pool_df(batch.column(0).to_pylist(), batch.column(1).to_pylist(), d, needed)
        df_before.update(b)
        df_after.update(a)
    cap = math.floor(cfg["cap_frac"]["NT"] * n_pool)
    before = shares(usable_flags(pairs_before, df_before, cap), scripts)
    after = shares(usable_flags(pairs_after, df_after, cap), scripts)
    passed = gate_decision(before, after)

    dict_meta = {"gate_pass": passed, "gate_overall_before": before["overall"],
                 "gate_overall_after": after["overall"], "n_pairs": n_pairs,
                 "n_entries": {k: len(v) for k, v in tdict.items()}, "source_set": "DICT_SRC",
                 "rule": {"min_count": MIN_COUNT, "min_ratio": MIN_RATIO, "jw_min": JW_MIN,
                          "pool_token_min_len": 2},
                 "config_hash": io_utils.config_hash(cfg)}
    out = io_utils.work_path("translit", "dict_v1.json")
    doc = {"name": tdict["name"], "addr": tdict["addr"], "meta": dict_meta}
    io_utils.atomic_write_text(out, json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    payload = {"gate_set": "DEV_A", "gate_pass": passed, "cap": cap, "cap_frac_NT": cfg["cap_frac"]["NT"],
               "n_pool": n_pool, "before": before, "after": after, "n_pairs_learn": n_pairs,
               "n_entries": dict_meta["n_entries"],
               "rule": "overall usable-token share increases and no script decreases",
               "wall_time_s": round(time.time() - t0, 1), "peak_rss_mb": round(io_utils.peak_rss_mb(), 1)}
    io_utils.write_metrics("translit_dict", payload, len(sets["DEV_A"]), int(qs.size), cfg)
    log.info("gate: usable-token share %.4f -> %.4f, pass=%s; wrote %s", before["overall"],
             after["overall"], passed, out)
    if not passed:
        log.error("translit dictionary gate FAILED (plan §5.3): %s -> %s", before, after)
        raise RuntimeError("BLOCKER: translit dictionary gate failed (metrics translit_dict.json)")
