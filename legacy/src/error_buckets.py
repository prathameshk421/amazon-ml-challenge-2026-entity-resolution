"""Phase 4: bucket every val FP/FN pair (priority order) + blocking misses.
Buckets: chain / acronym / phonetic / suffix-gap / addr-mismatch / ambiguous / other.
Writes artifacts/metrics/error_buckets.json. Uses val_probs + threshold.json + GT.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name, strip_suffix_tokens
import jellyfish

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "artifacts")
MET = os.path.join(ROOT, "artifacts/metrics")
TR = os.path.join(ROOT, "student_resource/dataset/train")


def initials(s):
    return "".join(t[0] for t in normalize_name(s).split() if t)


def main():
    th = json.load(open(os.path.join(ART, "threshold.json")))
    t = th["t"]
    Pv = pd.read_parquet(os.path.join(ART, "val_probs.parquet"))
    # R1 predictions at t
    keep = Pv[Pv.p >= t]
    pred = keep.groupby("s1_id")["cand_id"].apply(set).to_dict()
    import csv
    truth, s1t = {}, {}
    want = set(Pv.s1_id.unique().tolist())
    with open(os.path.join(TR, "train_ground_truth.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["source1_entity_id"] in want:
                m = row["matched_entity_ids"]
                truth[row["source1_entity_id"]] = set(m.split(",")) if m else set()
    with open(os.path.join(TR, "train_source1.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in want:
                s1t[row["entity_id"]] = (row["business_name"] or "", row["business_address"] or "")
    # pool texts for involved candidates
    from featurize import preload
    import featurize as FZ
    from featurize import pool_text
    preload([])
    involved = set()
    for e, ps in pred.items():
        involved |= set(ps)
    for e, ts in truth.items():
        involved |= set(ts)
    pt = {c: pool_text(c) for c in involved}
    # chain names: normalized S1 name in >=2 S1 entities (val entities)
    namecount = {}
    for e, (nm, ad) in s1t.items():
        nn = normalize_name(nm)
        namecount[nn] = namecount.get(nn, 0) + 1

    def bucket(e, c, is_fp):
        qn, qa = s1t[e]
        rn, ra = pt[c][0], pt[c][1]
        n1, n2 = normalize_name(qn), normalize_name(rn)
        if namecount.get(n1, 0) >= 2:
            return "chain"
        i1, i2 = initials(qn), initials(rn)
        if i1 and i1 == i2 and n1 != n2:
            return "acronym"
        m1 = set(jellyfish.metaphone(x) for x in n1.split() if x)
        m2 = set(jellyfish.metaphone(x) for x in n2.split() if x)
        lev = 1 - __import__("rapidfuzz").fuzz.ratio(n1, n2) / 100.0
        mj = len(m1 & m2) / len(m1 | m2) if (m1 | m2) else 0
        if mj >= 0.5 and lev > 0.4:
            return "phonetic"
        if strip_suffix_tokens(n1) == strip_suffix_tokens(n2) and n1 != n2:
            return "suffix_gap"
        from rapidfuzz import fuzz
        ns = fuzz.ratio(n1, n2) / 100.0
        a1 = qa or ""
        a2 = ra or ""
        asl = fuzz.ratio(a1.lower(), a2.lower()) / 100.0 if (a1 or a2) else 1.0
        if ns >= 0.7 and asl < 0.5:
            return "addr_mismatch"
        if 0.4 <= ns < 0.7 and 0.3 <= asl < 0.7:
            return "ambiguous"
        return "other"

    buckets = {"chain": 0, "acronym": 0, "phonetic": 0, "suffix_gap": 0,
               "addr_mismatch": 0, "ambiguous": 0, "other": 0}
    examples = {k: [] for k in buckets}
    n_fp = n_fn = 0
    candsets = Pv.groupby("s1_id")["cand_id"].apply(set).to_dict()
    for e in want:
        ps, ts = pred.get(e, set()), truth.get(e, set())
        cs = candsets.get(e, set())
        for c in ps - ts:
            n_fp += 1
            b = bucket(e, c, True)
            buckets[b] += 1
            if len(examples[b]) < 3:
                examples[b].append({"type": "FP", "s1": e, "s1_name": s1t[e][0],
                                    "s1_addr": s1t[e][1], "c": c, "c_name": pt[c][0],
                                    "c_addr": pt[c][1]})
        for c in ts - ps:
            n_fn += 1
            kind = "FN_scored_low" if c in cs else "FN_blocking_miss"
            b = bucket(e, c, False)
            buckets[b] += 1
            if len(examples[b]) < 3:
                examples[b].append({"type": kind, "s1": e, "s1_name": s1t[e][0],
                                    "s1_addr": s1t[e][1], "c": c, "c_name": pt[c][0],
                                    "c_addr": pt[c][1]})
    # blocking misses counted separately
    bmiss = sum(len(truth.get(e, set()) - candsets.get(e, set())) for e in want)
    out = {"t": t, "n_fp": n_fp, "n_fn": n_fn,
           "buckets": buckets, "bucket_sum": sum(buckets.values()),
           "fp_plus_fn": n_fp + n_fn,
           "blocking_miss_pairs": bmiss,
           "examples": examples}
    assert out["bucket_sum"] == out["fp_plus_fn"], "bucket counts must sum to error count"
    json.dump(out, open(os.path.join(MET, "error_buckets.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    main()
