"""Mine noise-catalog examples + chain illustrations from eda_pairs_sample.parquet.
Writes artifacts/metrics/eda_catalog.json. Run: .venv/bin/python legacy/src/eda_catalog.py
"""

import json
import os
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import strip_suffix_tokens, normalize_name

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MET = os.path.join(ROOT, "artifacts/metrics")


def row(r):
    return {"s1": r.s1, "cand": r.cand, "s1_name": r.s1_name, "c_name": r.c_name,
            "s1_addr": r.s1_addr, "c_addr": r.c_addr,
            "n_lev_norm": round(float(r.n_lev_norm), 3), "n_jacc": round(float(r.n_jacc), 3),
            "a_lev_norm": round(float(r.a_lev_norm), 3), "a_jacc": round(float(r.a_jacc), 3)}


def main():
    df = pd.read_parquet(os.path.join(ROOT, "artifacts/eda_pairs_sample.parquet"))
    pos = df[df.label == 1]
    hard = df[df.label == 2]
    cat = {}

    def take(frame, mask, n=2):
        sub = frame[mask].head(n)
        return [row(r) for r in sub.itertuples()]

    # 1. legal-suffix variants: positives equal after suffix strip but not before
    nn1 = pos["s1_name"].map(normalize_name)
    nn2 = pos["c_name"].map(normalize_name)
    m = (nn1 != nn2) & (nn1.map(strip_suffix_tokens) == nn2.map(strip_suffix_tokens))
    cat["legal_suffix_gap"] = {"n": int(m.sum()),
                               "frac_of_pos": float(m.mean()),
                               "examples": take(pos, m)}
    # 2. & vs and
    m = pos["s1_name"].str.contains("&", na=False) ^ pos["c_name"].str.contains("&", na=False)
    cat["ampersand_vs_and"] = {"n": int(m.sum()), "examples": take(pos, m)}
    # 3. transposition: same token multiset, different order
    m = (pos["n_jacc"] == 1.0) & (pos["n_lev_norm"] < 0.9)
    cat["word_order_transposition"] = {"n": int(m.sum()), "examples": take(pos, m)}
    # 4. typo-level: high but imperfect lev
    m = (pos["n_lev_norm"] >= 0.8) & (pos["n_lev_norm"] < 1.0)
    cat["typo_level"] = {"n": int(m.sum()), "examples": take(pos, m)}
    # 5. transliteration/phonetic: low lev, high metaphone
    m = (pos["n_lev_norm"] < 0.6) & (pos["n_meta"] >= 0.5)
    cat["phonetic_like"] = {"n": int(m.sum()), "examples": take(pos, m)}
    # 6. address landmark: positives with near/opp in either address
    lam = pos["s1_addr"].str.contains("near|opp|behind", case=False, na=False) | \
        pos["c_addr"].str.contains("near|opp|behind", case=False, na=False)
    cat["landmark_address"] = {"n": int(lam.sum()), "examples": take(pos, lam)}
    # 7. empty address on pool side
    m = pos["c_addr"] == ""
    cat["empty_pool_address"] = {"n": int(m.sum()), "examples": take(pos, m)}
    # 8. domain-name businesses
    m = pos["s1_name"].str.contains(r"\.com|\.in|\.org|\.net", na=False) | \
        pos["c_name"].str.contains(r"\.com|\.in|\.org|\.net", na=False)
    cat["domain_names"] = {"n": int(m.sum()), "examples": take(pos, m)}
    # 9. country disagreement among positives
    m = pos["s1_country"] != pos["c_country"]
    cat["country_disagree"] = {"n": int(m.sum()), "examples": take(pos, m)}
    # 10. hard negatives: same normalized name, different entity (chain confusion)
    nn1h = hard["s1_name"].map(normalize_name)
    nn2h = hard["c_name"].map(normalize_name)
    m = nn1h == nn2h
    cat["same_name_hard_negative"] = {"n": int(m.sum()),
                                      "frac_of_hard": float(m.mean()),
                                      "examples": take(hard, m, n=3)}
    # 11. acronym-like hard negatives: initials match
    def initials(s):
        return "".join(t[0] for t in normalize_name(s).split() if t)
    mh = hard.head(20000)
    mi = mh["s1_name"].map(initials) == mh["c_name"].map(initials)
    cat["acronym_like_hardneg_rate_sample20k"] = float(mi.mean())
    with open(os.path.join(MET, "eda_catalog.json"), "w") as f:
        json.dump(cat, f, indent=1, ensure_ascii=False)
    for k, v in cat.items():
        print(k, ({kk: vv for kk, vv in v.items() if kk != "examples"} if isinstance(v, dict) else v))


if __name__ == "__main__":
    main()
