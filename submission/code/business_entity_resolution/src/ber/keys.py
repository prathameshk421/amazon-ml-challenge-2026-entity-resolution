"""Key families (plan §5.5): verbatim port of plan/reference/keys_ref.py.

Inputs are outputs of textnorm.normalize_field. Hashing lives in ber.hashing.
"""

from collections import Counter

import jellyfish

from ber.textnorm import skeleton

# v2 default retrieval families. SK (single skeleton token) is kept as a function (stage-2 feature,
# E2 candidate) but is not a default family.
FAMILIES = ("EN", "NT", "NG", "NP", "PH", "SP", "NC", "AT", "AB", "XK")
NAME_FAMILIES = ("EN", "NT", "NG", "NP", "PH", "SK", "SP", "NC")
ADDR_FAMILIES = ("AT", "AB")


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


def concat_variants(tokens, G, nc_min_len=6):
    """NC variants: the full concatenation plus fixed edge-strip variants.
    Every variant is emitted only if its length >= nc_min_len. Order: all, -last1, -last2, -first,
    -first-last1."""
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
        if len(k) >= nc_min_len and k not in out:
            out.append(k)
    return out


def name_keys(name_norm, G, side, nc_min_len=6, nc_pool_token_min_len=8):
    """side: 'query' or 'pool'. Returns {family: [key strings]} (deduplicated, order stable)."""
    toks = name_norm.split()
    out = {f: [] for f in NAME_FAMILIES}
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
    nc = concat_variants(toks, G, nc_min_len)
    if side == "pool":
        nc += [t for t in toks if len(t) >= nc_pool_token_min_len]
    out["NC"] = list(dict.fromkeys(nc))
    return out


def addr_keys(addr_norm):
    toks = addr_norm.split()
    return {"AT": list(dict.fromkeys(toks)),
            "AB": list(dict.fromkeys(toks[i] + " " + toks[i + 1] for i in range(len(toks) - 1)))}


def edge_token_set(pool_name_norms, n_pool, min_frac, min_edge_share):
    """G (computed on the split's pool names); min_frac / min_edge_share from cfg["edge_tokens"]."""
    c_all, c_edge = Counter(), Counter()
    for nm in pool_name_norms:
        toks = nm.split()
        if not toks:
            continue
        uniq = set(toks)
        c_all.update(uniq)
        edge = {toks[0]} | set(toks[-2:])
        c_edge.update(edge)
    thr = min_frac * n_pool
    return sorted(t for t, c in c_all.items() if c >= thr and c_edge[t] / c >= min_edge_share)


def xk_keys(name_norm, addr_norm, at_df, at_cap):
    """XK: skeleton of each distinct first-8 name token (skeleton length >= 2) x each distinct address
    token whose AT df (same split pool) is <= at_cap. Key string = skeleton + "#" + token.
    Same function for pool and query sides."""
    f8 = _first8(name_norm.split())
    sks = list(dict.fromkeys(k for k in (skeleton(t) for t in f8) if len(k) >= 2))
    ats = [a for a in dict.fromkeys(addr_norm.split()) if at_df.get(a, 0) <= at_cap]
    return [k + "#" + a for k in sks for a in ats]


def record_keys(name_norm, addr_norm, families, G, at_df, at_cap, side):
    """{family: [key strings]} for the requested families only."""
    out = {}
    if any(f in NAME_FAMILIES for f in families):
        nk = name_keys(name_norm, G, side)
        out.update((f, nk[f]) for f in families if f in NAME_FAMILIES)
    if any(f in ADDR_FAMILIES for f in families):
        ak = addr_keys(addr_norm)
        out.update((f, ak[f]) for f in families if f in ADDR_FAMILIES)
    if "XK" in families:
        out["XK"] = xk_keys(name_norm, addr_norm, at_df, at_cap)
    return out
