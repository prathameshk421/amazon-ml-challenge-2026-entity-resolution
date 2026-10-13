"""Phase 0 structural summary (streaming, memory-light).

Reads train+test source files + train GT with the csv module (one pass each),
writes artifacts/metrics/data_summary.json and reports/data_summary.md,
and artifacts/metrics/test_inspection.json + reports/test_inspection_log.md.

Test inspection is STRUCTURAL ONLY (counts/schema/ID/country/tokens), logged.
Run: .venv/bin/python legacy/src/phase0_summary.py
"""

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
TE = os.path.join(ROOT, "student_resource/dataset/test")
MET = os.path.join(ROOT, "artifacts/metrics")
REP = os.path.join(ROOT, "reports")

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokens(text):
    return TOKEN_RE.findall(text.lower())


def has_non_ascii(s):
    return any(ord(c) > 127 for c in s)


def has_non_latin(s):
    for c in s:
        o = ord(c)
        if o > 127 and not (0x300 <= o <= 0x36F or 0x2000 <= o <= 0x206F or 0x20A0 <= o <= 0x20CF):
            # crude: non-ascii that is a letter outside Latin blocks
            if 0x2500 <= o <= 0x2BFF or o in (0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D,
                                              0x2026, 0x2010, 0x2011, 0xFF08, 0xFF09):
                continue
            return True
    return False


def scan_source(path, detailed_tokens=False):
    """One streaming pass: counts, countries, null/empty, prefixes, non-ascii, top tokens."""
    n = 0
    cols = None
    countries = Counter()
    empty = Counter()
    prefixes = Counter()
    non_ascii_name = Counter()
    non_ascii_addr = Counter()
    non_ascii_rows = Counter()
    nonlatin_rows = Counter()
    tok_by_country = defaultdict(Counter)
    seen_ids = set() if "test_source1" in path or "train_source1" in path else None
    dup_ids = 0
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        cols = list(r.fieldnames)
        for row in r:
            n += 1
            eid = row.get("entity_id", "")
            prefixes[eid.split("-")[0] if "-" in eid else "?"] += 1
            if seen_ids is not None:
                if eid in seen_ids:
                    dup_ids += 1
                else:
                    seen_ids.add(eid)
            for c in cols:
                if row.get(c, "") == "":
                    empty[c] += 1
            co = row.get("country", "")
            countries[co] += 1
            nm = row.get("business_name", "")
            ad = row.get("business_address", "")
            if has_non_ascii(nm):
                non_ascii_name[co] += 1
            if has_non_ascii(ad):
                non_ascii_addr[co] += 1
            if has_non_ascii(nm) or has_non_ascii(ad):
                non_ascii_rows[co] += 1
            if has_non_latin(nm) or has_non_latin(ad):
                nonlatin_rows[co] += 1
            if detailed_tokens:
                tc = tok_by_country[co]
                for t in tokens(nm):
                    tc["N:" + t] += 1
                for t in tokens(ad):
                    tc["A:" + t] += 1
    out = {
        "rows": n,
        "columns": cols,
        "countries": dict(countries),
        "empty_counts": dict(empty),
        "id_prefixes": dict(prefixes),
    }
    out["non_ascii_name_rows"] = dict(non_ascii_name)
    out["non_ascii_addr_rows"] = dict(non_ascii_addr)
    out["non_ascii_any_rows"] = dict(non_ascii_rows)
    out["non_latin_rows"] = dict(nonlatin_rows)
    if seen_ids is not None:
        out["unique_ids"] = len(seen_ids)
        out["dup_ids"] = dup_ids
    if detailed_tokens:
        out["top_tokens"] = {c: tc.most_common(30) for c, tc in tok_by_country.items()}
    return out


def scan_gt(path, max_example_buckets=6):
    n = 0
    n_singleton = 0
    match_counts = Counter()
    s2_in_gt = s3_in_gt = 0
    max_k = 0
    bad_format = 0
    len_hist = Counter()
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        cols = list(r.fieldnames)
        for row in r:
            n += 1
            s1 = row.get("source1_entity_id", "")
            m = row.get("matched_entity_ids", "")
            if not s1.startswith("S1-"):
                bad_format += 1
            if m == "":
                n_singleton += 1
                match_counts[0] += 1
                continue
            ids = [x for x in m.split(",") if x]
            k = len(ids)
            match_counts[k] += 1
            max_k = max(max_k, k)
            len_hist[min(k // 5 * 5, 50)] += 1
            for x in ids:
                if x.startswith("S2-"):
                    s2_in_gt += 1
                elif x.startswith("S3-"):
                    s3_in_gt += 1
                else:
                    bad_format += 1
    return {
        "rows": n,
        "columns": cols,
        "n_singleton": n_singleton,
        "singleton_rate": n_singleton / n if n else 0.0,
        "match_count_dist": {str(k): v for k, v in sorted(match_counts.items())},
        "max_matches": max_k,
        "s2_links": s2_in_gt,
        "s3_links": s3_in_gt,
        "bad_format_rows": bad_format,
    }


def main():
    os.makedirs(MET, exist_ok=True)
    summary = {}
    print("scanning train sources...", flush=True)
    for name in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
        summary[name] = scan_source(os.path.join(TR, name))
        print("done", name, summary[name]["rows"], flush=True)
    print("scanning train GT...", flush=True)
    summary["train_ground_truth.tsv"] = scan_gt(os.path.join(TR, "train_ground_truth.tsv"))
    print("done GT", flush=True)
    print("scanning test sources (structural only)...", flush=True)
    test = {}
    for name in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
        test[name] = scan_source(os.path.join(TE, name), detailed_tokens=(name == "test_source1.tsv" or True))
        print("done", name, test[name]["rows"], flush=True)
    summary["test"] = test

    # machine limits (script-collected, so reportable)
    try:
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        batt = "pmset failed: %r" % e
    summary["machine"] = {
        "cpu_count": os.cpu_count(),
        "ncpu_sysctl": None,
        "mem_bytes": None,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "disk_total_bytes": shutil.disk_usage(ROOT).total,
        "battery": batt,
    }
    try:
        out = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True, timeout=30).stdout.strip()
        summary["machine"]["ncpu_sysctl"] = int(out)
    except Exception:
        pass
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=30).stdout.strip()
        summary["machine"]["mem_bytes"] = int(out)
    except Exception:
        pass

    with open(os.path.join(MET, "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    # test inspection: structural-only subset + log
    insp = {
        "what": "row counts, schema, ID prefix/uniqueness, distinct country labels+counts, "
                "non-ASCII rates per country, top-30 tokens per country (test_source files only, no labels, no matching)",
        "why": "prompt Phase 0.1 + C7: verify France present structurally; check normalization/tokenization robustness",
        "files": {k: {kk: v[kk] for kk in ("rows", "columns", "countries", "id_prefixes",
                                           "non_ascii_any_rows", "non_latin_rows", "top_tokens")
                       if kk in v} for k, v in test.items()},
    }
    if "test_source1.tsv" in test:
        s1 = test["test_source1.tsv"]
        insp["files"]["test_source1.tsv"]["unique_ids"] = s1.get("unique_ids")
        insp["files"]["test_source1.tsv"]["dup_ids"] = s1.get("dup_ids")
    with open(os.path.join(MET, "test_inspection.json"), "w") as f:
        json.dump(insp, f, indent=1, ensure_ascii=False)

    md = []
    md.append("# Data summary (from artifacts/metrics/data_summary.json)")
    md.append("")
    for name in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
        s = summary[name]
        md.append("## %s: %d rows; cols=%s" % (name, s["rows"], ",".join(s["columns"])))
        md.append("countries: %s" % json.dumps(s["countries"], ensure_ascii=False))
        md.append("empty: %s; prefixes: %s" % (json.dumps(s["empty_counts"]), json.dumps(s["id_prefixes"])))
        md.append("non-ascii rows by country: %s" % json.dumps(s["non_ascii_any_rows"], ensure_ascii=False))
        md.append("non-latin rows by country: %s" % json.dumps(s["non_latin_rows"], ensure_ascii=False))
        md.append("")
    g = summary["train_ground_truth.tsv"]
    md.append("## train_ground_truth.tsv: %d rows; cols=%s" % (g["rows"], ",".join(g["columns"])))
    md.append("singleton_rate=%.4f (%d/%d); max_matches=%d; s2_links=%d s3_links=%d bad_format=%d"
              % (g["singleton_rate"], g["n_singleton"], g["rows"], g["max_matches"],
                 g["s2_links"], g["s3_links"], g["bad_format_rows"]))
    md.append("match-count dist: %s" % json.dumps(g["match_count_dist"]))
    md.append("")
    for name in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
        s = test[name]
        md.append("## %s: %d rows; countries=%s; prefixes=%s"
                  % (name, s["rows"], json.dumps(s["countries"], ensure_ascii=False),
                     json.dumps(s["id_prefixes"])))
    md.append("")
    m = summary["machine"]
    md.append("## machine: ncpu=%s mem_GB=%.1f disk_free_GB=%.1f" %
              (m["ncpu_sysctl"], (m["mem_bytes"] or 0) / 1e9, m["disk_free_bytes"] / 1e9))
    md.append("battery: %s" % m["battery"])
    with open(os.path.join(REP, "data_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    log = ["# Test inspection log (C7: structural only, no matching/scoring/tuning)",
           "",
           "## 2026-09-25 Phase 0.1",
           "- WHAT: row counts, schema/columns, ID prefix + S1 uniqueness, distinct country labels+counts, "
           "non-ASCII/non-Latin rates per country, top-30 name/address tokens per country, on "
           "student_resource/dataset/test/test_source{1,2,3}.tsv via legacy/src/phase0_summary.py.",
           "- WHY: confirm France present structurally; check normalization will not break on French text.",
           "- FORBIDDEN (not done): no matching/scoring against test, no threshold/feature tuning on test, "
           "no rules from specific test rows, no hand-labeling.",
           "- ARTIFACT: artifacts/metrics/test_inspection.json",
           ""]
    with open(os.path.join(REP, "test_inspection_log.md"), "w") as f:
        f.write("\n".join(log))
    print("WROTE data_summary.json + data_summary.md + test_inspection files", flush=True)


if __name__ == "__main__":
    main()
