"""Evidence for the train -> (DEV, HOLDOUT_TEST) split design. Read-only on data.
Writes plan/checks/split_design.json.

Measures:
 1. GT structure per country: singleton rate, match-count distribution, links in S2 vs S3.
 2. Country agreement of ALL true links (S1 country vs pool country), empty/<NULL>-only pool
    addresses among true matches.
 3. Name overlap ("does this S1's normalized name also occur among the other side's S1 names"):
    a) official test S1 vs all train S1   (test inputs only, no labels; logged as test usage)
    b) random 20% of train S1 vs the other 80%  (what a random holdout would see)
    c) same for a name-grouped 20% (by construction 0)
    so we can pick the split whose overlap matches the official test.
 4. Official test vs train size ratios.
Normalization = plan/reference/translit_ref.normalize_field (S1 is Latin-only).
"""

import json
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "plan/reference"))
from translit_ref import normalize_field  # noqa: E402

D = os.path.join(ROOT, "student_resource/dataset")


def rows(rel):
    with open(os.path.join(D, rel), encoding="utf-8", newline="") as f:
        f.readline()
        for line in f:
            yield line.rstrip("\n").split("\t")


def main():
    out = {}
    s1 = {}
    for eid, name, addr, co in rows("train/train_source1.tsv"):
        s1[eid] = (normalize_field(name, "name")[0], co)
    pool_co, pool_addr_empty, pool_name_empty = {}, set(), set()
    for rel in ("train/train_source2.tsv", "train/train_source3.tsv"):
        for eid, name, addr, co in rows(rel):
            pool_co[eid] = co
            if not addr.replace("<NULL>", "").replace(",", "").strip():
                pool_addr_empty.add(eid)
            if not name.strip():
                pool_name_empty.add(eid)
    # 1 + 2
    per_co = {}
    agree = Counter()
    t_empty_addr = 0
    t_empty_name = 0
    links = 0
    for a, b in rows("train/train_ground_truth.tsv"):
        co = s1[a][1]
        d = per_co.setdefault(co, {"n": 0, "singleton": 0, "match_count": Counter(), "S2": 0, "S3": 0})
        d["n"] += 1
        ids = b.split(",") if b else []
        if not ids:
            d["singleton"] += 1
        d["match_count"][min(len(ids), 6)] += 1
        for x in ids:
            links += 1
            d["S2" if x.startswith("S2-") else "S3"] += 1
            agree["same" if pool_co[x] == co else "diff"] += 1
            t_empty_addr += x in pool_addr_empty
            t_empty_name += x in pool_name_empty
    for co, d in per_co.items():
        d["singleton_rate"] = round(d["singleton"] / d["n"], 4)
        d["match_count"] = {("6+" if k == 6 else str(k)): v for k, v in sorted(d["match_count"].items())}
    out["gt_by_country"] = per_co
    out["true_links"] = links
    out["true_link_country"] = dict(agree)
    out["true_links_pool_addr_empty_or_only_NULL"] = t_empty_addr
    out["true_links_pool_name_empty"] = t_empty_name
    out["pool_addr_empty_or_only_NULL_total"] = len(pool_addr_empty)
    # 3
    train_names = Counter(n for n, _ in s1.values())
    test_names = []
    for eid, name, addr, co in rows("test/test_source1.tsv"):
        test_names.append((normalize_field(name, "name")[0], co))
    ov = Counter()
    for n, co in test_names:
        ov[(co, n in train_names)] += 1
    out["test_S1_name_in_train_S1"] = {
        co: round(ov[(co, True)] / (ov[(co, True)] + ov[(co, False)]), 4)
        for co in sorted({c for c, _ in ov})}
    out["test_S1_name_in_train_S1_all"] = round(sum(v for (c, t), v in ov.items() if t) / len(test_names), 4)
    ids = sorted(s1)
    rng = random.Random(42)
    hold = set(rng.sample(ids, len(ids) // 5))
    dev_names = Counter(s1[i][0] for i in ids if i not in hold)
    ovr = Counter()
    for i in hold:
        ovr[(s1[i][1], s1[i][0] in dev_names)] += 1
    out["random20_holdout_name_in_dev80"] = {
        co: round(ovr[(co, True)] / (ovr[(co, True)] + ovr[(co, False)]), 4)
        for co in sorted({c for c, _ in ovr})}
    out["random20_holdout_name_in_dev80_all"] = round(sum(v for (c, t), v in ovr.items() if t) / len(hold), 4)
    # the test/train size ratio differs from 20/80: also compare against a holdout whose reference
    # side has the same size as train S1 (use all other train S1 as reference: leave-one-out rate)
    loo = sum(1 for i in hold if train_names[s1[i][0]] >= 2) / len(hold)
    out["random20_holdout_name_in_other_train_S1_all"] = round(loo, 4)
    # 4
    out["sizes"] = {"train_S1": len(s1), "train_pool": len(pool_co), "test_S1": len(test_names),
                    "test_pool": 4887273 + 5082316,
                    "train_pool_per_S1": round(len(pool_co) / len(s1), 3),
                    "test_pool_per_S1": round((4887273 + 5082316) / len(test_names), 3),
                    "train_links_per_S1": round(links / len(s1), 3),
                    "train_orphan_pool_records": len(pool_co) - links}
    json.dump(out, open(os.path.join(ROOT, "plan/checks/split_design.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
