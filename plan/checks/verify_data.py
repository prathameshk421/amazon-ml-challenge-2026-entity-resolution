"""Read-only fact check of the raw dataset for the rebuild plan.
Writes plan/checks/data_facts.json. Touches test files only for structural facts
(ID format, parsing, row counts, character scripts), no matching.
"""

import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "student_resource/dataset")
csv.field_size_limit(10 ** 9)
ID_RE = re.compile(r"^S([123])-(\d+)$")

FILES = ["train/train_source1.tsv", "train/train_source2.tsv", "train/train_source3.tsv",
         "test/test_source1.tsv", "test/test_source2.tsv", "test/test_source3.tsv"]


def script_counts(s, acc):
    for c in s:
        if ord(c) < 128 or not unicodedata.category(c).startswith("L"):
            continue
        acc[unicodedata.name(c, "UNKNOWN").split(" ")[0]] += 1


def check_source(rel):
    p = os.path.join(D, rel)
    out = {}
    raw = open(p, "rb").read()
    out["bytes"] = len(raw)
    out["n_lines"] = raw.count(b"\n")
    out["has_CR"] = raw.count(b"\r")
    out["has_BOM"] = raw.startswith(b"\xef\xbb\xbf")
    out["n_doublequote_chars"] = raw.count(b'"')
    del raw
    fields_hist = Counter()
    ids = []
    empty = Counter()
    countries = Counter()
    lead0 = 0
    maxnum = 0
    widths = Counter()
    bad_id = 0
    pref = Counter()
    scripts = Counter()
    ws_edge = Counter()
    nameaddr = Counter()
    with open(p, encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
        out["header"] = header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            fields_hist[len(parts)] += 1
            if len(parts) != 4:
                continue
            eid, name, addr, co = parts
            m = ID_RE.match(eid)
            if not m:
                bad_id += 1
            else:
                pref[m.group(1)] += 1
                num = m.group(2)
                widths[len(num)] += 1
                if len(num) > 1 and num[0] == "0":
                    lead0 += 1
                maxnum = max(maxnum, int(num))
            ids.append(eid)
            if not name.strip():
                empty["name"] += 1
            if not addr.strip():
                empty["addr"] += 1
            if not co.strip():
                empty["country"] += 1
            for k, v in (("name", name), ("addr", addr), ("country", co)):
                if v != v.strip():
                    ws_edge[k] += 1
            countries[co] += 1
            if not rel.endswith("source1.tsv"):
                nameaddr[(name, addr, co)] += 1
            if not (name.isascii() and addr.isascii()):
                script_counts(name + addr, scripts)
    out["field_count_hist"] = dict(fields_hist)
    out["n_rows"] = len(ids)
    out["n_unique_ids"] = len(set(ids))
    out["bad_id_format"] = bad_id
    out["id_prefix"] = dict(pref)
    out["id_leading_zero"] = lead0
    out["id_max_num"] = maxnum
    out["id_num_width_hist"] = dict(sorted(widths.items()))
    out["empty"] = dict(empty)
    out["leading_trailing_ws"] = dict(ws_edge)
    out["countries"] = dict(countries)
    out["scripts_nonascii_letters"] = dict(scripts.most_common(30))
    if nameaddr:
        dups = [v for v in nameaddr.values() if v > 1]
        out["exact_dup_name_addr_country_groups"] = len(dups)
        out["exact_dup_rows_in_groups"] = int(sum(dups))
    # pandas default parser (quoting ON) vs line split
    try:
        n_pd = len(pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, usecols=[0]))
    except Exception as e:  # noqa
        n_pd = "ERROR %s" % e
    out["pandas_default_quoting_rows"] = n_pd
    n_pd_none = len(pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False,
                                quoting=csv.QUOTE_NONE, usecols=[0]))
    out["pandas_quote_none_rows"] = n_pd_none
    return out, set(ids)


def main():
    facts = {}
    idsets = {}
    for rel in FILES:
        facts[rel], idsets[rel] = check_source(rel)
        print(rel, json.dumps({k: facts[rel][k] for k in ("n_rows", "n_unique_ids", "field_count_hist",
              "pandas_default_quoting_rows", "pandas_quote_none_rows", "id_leading_zero")}), flush=True)
    # numeric overlap between S2 and S3 ids (does the number alone identify a record?)
    for split in ("train", "test"):
        s2 = {i.split("-", 1)[1] for i in idsets["%s/%s_source2.tsv" % (split, split)]}
        s3 = {i.split("-", 1)[1] for i in idsets["%s/%s_source3.tsv" % (split, split)]}
        facts["%s_S2_S3_numeric_overlap" % split] = len(s2 & s3)
    # train vs test id overlap
    for s in ("1", "2", "3"):
        a = idsets["train/train_source%s.tsv" % s]
        b = idsets["test/test_source%s.tsv" % s]
        facts["train_test_id_overlap_S%s" % s] = len(a & b)
    # ground truth format
    gt = os.path.join(D, "train/train_ground_truth.tsv")
    pool = idsets["train/train_source2.tsv"] | idsets["train/train_source3.tsv"]
    s1 = idsets["train/train_source1.tsv"]
    g = {"rows": 0, "empty": 0, "ids_with_space": 0, "dup_within_list": 0, "missing_in_pool": 0,
         "s1_not_in_source1": 0, "dup_s1_rows": 0, "n_links": 0, "max_list": 0}
    seen = set()
    owner = Counter()
    with open(gt, encoding="utf-8", newline="") as f:
        f.readline()
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            g["rows"] += 1
            if a in seen:
                g["dup_s1_rows"] += 1
            seen.add(a)
            if a not in s1:
                g["s1_not_in_source1"] += 1
            if not b:
                g["empty"] += 1
                continue
            lst = b.split(",")
            g["n_links"] += len(lst)
            g["max_list"] = max(g["max_list"], len(lst))
            if len(lst) != len(set(lst)):
                g["dup_within_list"] += 1
            for x in lst:
                if x != x.strip():
                    g["ids_with_space"] += 1
                if x not in pool:
                    g["missing_in_pool"] += 1
                owner[x] += 1
    g["s1_without_gt_row"] = len(s1 - seen)
    g["pool_records_with_2plus_owners"] = sum(1 for v in owner.values() if v > 1)
    facts["ground_truth"] = g
    json.dump(facts, open(os.path.join(ROOT, "plan/checks/data_facts.json"), "w"), indent=1,
              ensure_ascii=False)
    print("WROTE plan/checks/data_facts.json")


if __name__ == "__main__":
    main()
