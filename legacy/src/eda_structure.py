"""Phase 1c: ground-truth structure (fit part for conclusions; full-train for orphans/chains).
Writes artifacts/metrics/eda_structure.json. Run: .venv/bin/python legacy/src/eda_structure.py
"""

import csv
import json
import os
from collections import Counter

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
MET = os.path.join(ROOT, "artifacts/metrics")


def main():
    split = json.load(open(os.path.join(ROOT, "artifacts/split.json")))
    fit = set(split["fit_ids"])
    # fit-part GT structure
    n_fit = n_sing = 0
    mc = Counter()
    s2only = s3only = both = 0
    # full-train link prefix counts (unique == total: each record <=1 owner, gt_checks.json)
    link2 = link3 = 0
    ent_match_count = {}
    for row in _rows("train_ground_truth.tsv"):
        m = row["matched_entity_ids"]
        if m:
            ids = m.split(",")
            for x in ids:
                if x.startswith("S2-"):
                    link2 += 1
                else:
                    link3 += 1
        if row["source1_entity_id"] in fit:
            n_fit += 1
            if not m:
                n_sing += 1
                mc[0] += 1
            else:
                ids = m.split(",")
                mc[len(ids)] += 1
                ent_match_count[row["source1_entity_id"]] = len(ids)
                h2 = any(x.startswith("S2-") for x in ids)
                h3 = any(x.startswith("S3-") for x in ids)
                if h2 and h3:
                    both += 1
                elif h2:
                    s2only += 1
                else:
                    s3only += 1
    fit_nonsing = n_fit - n_sing
    struct = {
        "fit_n": n_fit, "fit_singleton": n_sing, "fit_singleton_rate": n_sing / n_fit,
        "fit_matchcount": {str(k): v for k, v in sorted(mc.items())},
        "fit_s2only_frac": s2only / fit_nonsing, "fit_s3only_frac": s3only / fit_nonsing,
        "fit_both_frac": both / fit_nonsing,
        "s2_links": link2, "s3_links": link3,
    }
    # orphans per source (full train)
    pool = json.load(open(os.path.join(MET, "data_summary.json")))
    s2tot = pool["train_source2.tsv"]["rows"]
    s3tot = pool["train_source3.tsv"]["rows"]
    struct["orphan_s2"] = s2tot - link2
    struct["orphan_s3"] = s3tot - link3
    struct["orphan_frac_s2"] = (s2tot - link2) / s2tot
    struct["orphan_frac_s3"] = (s3tot - link3) / s3tot

    # chains: normalized S1 name in >=2 S1 entities (full train S1)
    name2ents = {}
    for row in _rows("train_source1.tsv"):
        nn = normalize_name(row["business_name"] or "")
        if nn:
            name2ents.setdefault(nn, []).append(row["entity_id"])
    chains = {k: v for k, v in name2ents.items() if len(v) > 1}
    struct["n_s1_names"] = len(name2ents)
    struct["n_chains"] = len(chains)
    struct["chain_entities"] = sum(len(v) for v in chains.values())
    struct["chain_size_dist"] = dict(Counter(min(len(v), 10) for v in chains.values()))
    top = sorted(chains.items(), key=lambda kv: -len(kv[1]))[:15]
    struct["top_chains"] = [{"name": k, "n_entities": len(v), "entities": v[:8]} for k, v in top]
    with open(os.path.join(MET, "eda_structure.json"), "w") as f:
        json.dump(struct, f, indent=1)
    print(json.dumps({k: v for k, v in struct.items() if k != "top_chains"}, indent=1))
    print("top chains:", [(c["name"][:40], c["n_entities"]) for c in struct["top_chains"][:8]])


def _rows(fn):
    with open(os.path.join(TR, fn), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            yield row


if __name__ == "__main__":
    main()
