"""Reference (normative) key generators for plan §5.5. Port verbatim into src/ber/keys.py.
Input strings are outputs of translit_ref.normalize_field. Stdlib + jellyfish only.
"""

import hashlib

import jellyfish

from translit_ref import skeleton

# v2 default retrieval families. SK (single skeleton token) is kept as a function (stage-2 feature,
# E2 candidate) but is not a default family: only 5.54% of Indic true pairs share a skeleton whose
# df is under the cap (plan/checks/verify_critique.json).
FAMILIES = ("EN", "NT", "NG", "NP", "PH", "SP", "NC", "AT", "AB", "XK")


def key64(family, key):
    """Stable 64-bit key hash; the family code is part of the hashed string."""
    b = (family + "\x1f" + key).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


def _tok2(tokens):
    return [t for t in tokens if len(t) >= 2]


def _first8(tokens):
    out = []
    for t in _tok2(tokens):
        if t not in out:
            out.append(t)
            if len(out) == 8:
                break
    return out


def _pairs(items):
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            out.append(a + "|" + b if a <= b else b + "|" + a)
    return out


def concat_variants(tokens, G):
    """NC variants (plan §5.5): the full concatenation plus fixed edge-strip variants.
    Every variant is emitted only if its length >= 6. Order: all, -last1, -last2, -first, -first-last1."""
    t = list(tokens)
    cand = [t]
    if len(t) >= 2 and t[-1] in G:
        cand.append(t[:-1])
        if len(t) >= 3 and t[-2] in G:
            cand.append(t[:-2])
    if len(t) >= 2 and t[0] in G:
        cand.append(t[1:])
        if len(t) >= 3 and t[-1] in G:
            cand.append(t[1:-1])
    out = []
    for c in cand:
        k = "".join(c)
        if len(k) >= 6 and k not in out:
            out.append(k)
    return out


def name_keys(name_norm, G, side):
    """side: 'query' or 'pool'. Returns {family: [key strings]} (deduplicated, order stable)."""
    toks = name_norm.split()
    out = {f: [] for f in ("EN", "NT", "NG", "NP", "PH", "SK", "SP", "NC")}
    if not name_norm:
        return out
    out["EN"] = [name_norm]
    out["NT"] = list(dict.fromkeys(_tok2(toks)))
    s = " " + name_norm + " "
    out["NG"] = list(dict.fromkeys(s[i:i + 3] for i in range(len(s) - 2)))
    f8 = _first8(toks)
    out["NP"] = list(dict.fromkeys(_pairs(f8)))
    out["PH"] = list(dict.fromkeys(c for c in (jellyfish.metaphone(t) for t in _tok2(toks) if len(t) >= 3) if c))
    out["SK"] = list(dict.fromkeys(k for k in (skeleton(t) for t in _tok2(toks)) if len(k) >= 2))
    sk8 = list(dict.fromkeys(k for k in (skeleton(t) for t in f8) if len(k) >= 2))
    out["SP"] = list(dict.fromkeys(_pairs(sk8)))
    nc = concat_variants(toks, G)
    if side == "pool":
        nc += [t for t in toks if len(t) >= 8]
    out["NC"] = list(dict.fromkeys(nc))
    return out


def addr_keys(addr_norm):
    toks = addr_norm.split()
    return {"AT": list(dict.fromkeys(toks)),
            "AB": list(dict.fromkeys(toks[i] + " " + toks[i + 1] for i in range(len(toks) - 1)))}


def edge_token_set(pool_name_norms, n_pool):
    """G per plan §5.5 (computed on the split's pool names)."""
    from collections import Counter
    c_all, c_edge = Counter(), Counter()
    for nm in pool_name_norms:
        toks = nm.split()
        if not toks:
            continue
        uniq = set(toks)
        c_all.update(uniq)
        edge = {toks[0]} | set(toks[-2:])
        c_edge.update(edge)
    thr = 0.0005 * n_pool
    return sorted(t for t, c in c_all.items() if c >= thr and c_edge[t] / c >= 0.80)


def xk_keys(name_norm, addr_norm, at_df, at_cap):
    """XK (plan v2 §5.5): skeleton of each distinct first-8 name token (skeleton length >= 2) x each
    distinct address token whose AT df (same split pool) is <= at_cap. Key string = skeleton + "#" + token.
    Same function for pool and query sides."""
    f8 = _first8(name_norm.split())
    sks = list(dict.fromkeys(k for k in (skeleton(t) for t in f8) if len(k) >= 2))
    ats = [a for a in dict.fromkeys(addr_norm.split()) if at_df.get(a, 0) <= at_cap]
    return [k + "#" + a for k in sks for a in ats]
