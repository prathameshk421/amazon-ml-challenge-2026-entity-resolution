"""Phase 3 feature extraction. featurize(df_pairs, ...) -> float32 DataFrame.
Features (no closed-set country encoding; country = agree/unknown flags):
 n_lev_raw, n_lev_norm, n_jacc, n_tfidf, n_meta, n_snd, n_tset,
 a_lev_raw, a_lev_norm, a_jacc, a_tfidf, a_tset, a_postal3, a_digit,
 co_agree, co_unknown, len_name_diff, len_addr_diff, rare_shared
TF-IDF vectorizers are fit separately (fit_tfidf.py) and passed in.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name, normalize_address, postal_tokens
import jellyfish

FEATURES = ["n_lev_raw", "n_lev_norm", "n_jacc", "n_tfidf", "n_meta", "n_snd", "n_tset",
            "a_lev_raw", "a_lev_norm", "a_jacc", "a_tfidf", "a_tset", "a_postal3", "a_digit",
            "co_agree", "co_unknown", "len_name_diff", "len_addr_diff", "rare_shared"]


_NN_CACHE = {}
_NA_CACHE = {}
_META_CACHE = {}
_SND_CACHE = {}


def nn_cache(s):
    v = _NN_CACHE.get(s)
    if v is None:
        v = normalize_name(s)
        _NN_CACHE[s] = v
    return v


def na_cache(s):
    v = _NA_CACHE.get(s)
    if v is None:
        v = normalize_address(s)
        _NA_CACHE[s] = v
    return v


def meta_cache(t):
    v = _META_CACHE.get(t)
    if v is None:
        v = jellyfish.metaphone(t)
        _META_CACHE[t] = v
    return v


def snd_cache(t):
    v = _SND_CACHE.get(t)
    if v is None:
        v = jellyfish.soundex(t)
        _SND_CACHE[t] = v
    return v


def jacc(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build(df, vec_n=None, vec_a=None, tok_idf=None):
    """df columns: s1_name, s1_addr, s1_co, c_name, c_addr, c_co. Returns DataFrame[FEATURES]."""
    n = len(df)
    X = np.zeros((n, len(FEATURES)), dtype=np.float32)
    sn = df["s1_name"].values
    sa = df["s1_addr"].values
    sc = df["s1_co"].values
    cn = df["c_name"].values
    ca = df["c_addr"].values
    cc = df["c_co"].values
    for i in range(n):
        qn, qaddr = sn[i] or "", sa[i] or ""
        rn, raddr = cn[i] or "", ca[i] or ""
        nn = nn_cache(qn)
        nr = nn_cache(rn)
        na = na_cache(qaddr)
        ar = na_cache(raddr)
        nt1, nt2 = nn.split(), nr.split()
        at1, at2 = na.split(), ar.split()
        X[i, 0] = fuzz.ratio(qn, rn) / 100.0
        X[i, 1] = fuzz.ratio(nn, nr) / 100.0
        X[i, 2] = jacc(nt1, nt2)
        X[i, 4] = jacc(set(meta_cache(t) for t in nt1 if t),
                        set(meta_cache(t) for t in nt2 if t))
        X[i, 5] = jacc(set(snd_cache(t) for t in nt1 if t and t[0].isalpha()),
                        set(snd_cache(t) for t in nt2 if t and t[0].isalpha()))
        X[i, 6] = fuzz.token_set_ratio(nn, nr) / 100.0
        X[i, 7] = fuzz.ratio(qaddr, raddr) / 100.0
        X[i, 8] = fuzz.ratio(na, ar) / 100.0
        X[i, 9] = jacc(at1, at2)
        X[i, 11] = fuzz.token_set_ratio(na, ar) / 100.0
        pq, pr = set(postal_tokens(na)), set(postal_tokens(ar))
        if pq or pr:
            X[i, 12] = 1.0 if (pq & pr) else 0.0
        else:
            X[i, 12] = 0.5
        X[i, 13] = jacc(set(t for t in at1 if t[:1].isdigit()),
                         set(t for t in at2 if t[:1].isdigit()))
        X[i, 14] = 1.0 if (sc[i] == cc[i] and sc[i] != "") else 0.0
        X[i, 15] = 1.0 if (sc[i] == "" or cc[i] == "") else 0.0
        X[i, 16] = abs(len(nn) - len(nr))
        X[i, 17] = abs(len(na) - len(ar))
        if tok_idf is not None:
            s = 0.0
            for t in set(nt1) & set(nt2):
                s += tok_idf.get(t, 0.0)
            X[i, 18] = s
    if vec_n is not None:
        Q = vec_n.transform([nn_cache(x) or " " for x in sn])
        R = vec_n.transform([nn_cache(x) or " " for x in cn])
        X[:, 3] = np.asarray(Q.multiply(R).sum(axis=1)).ravel()
    if vec_a is not None:
        Q = vec_a.transform([na_cache(x) or " " for x in sa])
        R = vec_a.transform([na_cache(x) or " " for x in ca])
        X[:, 10] = np.asarray(Q.multiply(R).sum(axis=1)).ravel()
    return pd.DataFrame(X, columns=FEATURES)
