"""Phase 1a structural EDA (streaming; conclusions on fit-part S1 + full pool).
Writes artifacts/metrics/eda_struct.json. Run: .venv/bin/python legacy/src/eda_struct.py
"""

import csv
import json
import os
import re
import subprocess
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
MET = os.path.join(ROOT, "artifacts/metrics")
TMP = os.path.join(ROOT, "artifacts/tmp")
os.makedirs(TMP, exist_ok=True)

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
ENV = dict(os.environ, TMPDIR=TMP)
LEN_BINS = [0, 5, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200, 10 ** 9]


def bin_len(x):
    for b in LEN_BINS[1:]:
        if x < b:
            return "<%d" % b
    return "big"


def toks(text):
    return TOKEN_RE.findall(text.lower())


def load_fit():
    return set(json.load(open(os.path.join(ROOT, "artifacts/split.json")))["fit_ids"])


def pass1(fit_ids):
    """Stream all train sources; write name/country keys for dup counting; aggregate the rest."""
    stats = {}
    writers = {}
    raws = {}
    files = {"S1": ("train_source1.tsv", True), "S2": ("train_source2.tsv", False),
             "S3": ("train_source3.tsv", False)}
    for src, (fn, filt) in files.items():
        stats[src] = {"rows": 0, "rows_fit": 0, "empty": defaultdict(Counter),
                      "name_len": defaultdict(Counter), "addr_len": defaultdict(Counter),
                      "name_toklen": defaultdict(Counter), "addr_toklen": defaultdict(Counter),
                      "name_tok": defaultdict(Counter), "addr_tok": defaultdict(Counter)}
        raws[src] = os.path.join(TMP, "dup_%s.raw" % src)
        writers[src] = open(raws[src], "w", encoding="utf-8")
    for src, (fn, filt) in files.items():
        st = stats[src]
        with open(os.path.join(TR, fn), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if filt and row["entity_id"] not in fit_ids:
                    continue
                co = row["country"]
                nm = row["business_name"] or ""
                ad = row["business_address"] or ""
                st["rows"] += 1
                for col, v in (("business_name", nm), ("business_address", ad), ("country", co)):
                    if v == "":
                        st["empty"][col][co] += 1
                st["name_len"][bin_len(len(nm))][co] += 1
                st["addr_len"][bin_len(len(ad))][co] += 1
                tn, ta = toks(nm), toks(ad)
                st["name_toklen"][bin_len(len(tn))][co] += 1
                st["addr_toklen"][bin_len(len(ta))][co] += 1
                cn, ca = st["name_tok"][co], st["addr_tok"][co]
                for t in tn:
                    if len(t) <= 25:
                        cn[t] += 1
                for t in ta:
                    if t[:1].isdigit() and not (4 <= len(t) <= 6 and t.isdigit()):
                        continue
                    if len(t) <= 25:
                        ca[t] += 1
                writers[src].write("%s\t%s\t%s\n" % (co, nm.lower().strip(),
                                                       (nm.lower().strip() + "|" + ad.lower().strip())))
        writers[src].close()
        print("pass1 done", src, st["rows"], flush=True)
    return stats, raws


def dup_stats(raw):
    srt = raw + ".sorted"
    subprocess.run(["sort", "-o", srt, raw], check=True, env=ENV)
    os.remove(raw)
    per_country_groups = Counter()
    per_country_rows = Counter()
    top = Counter()
    prev = None
    cnt = 0
    with open(srt, encoding="utf-8") as f:
        for line in f:
            co, name, _full = line.rstrip("\n").split("\t", 2)
            key = (co, name)
            if key == prev:
                cnt += 1
            else:
                if prev is not None and cnt > 1:
                    per_country_groups[prev[0]] += 1
                    per_country_rows[prev[0]] += cnt
                    top[prev[1]] += cnt
                prev = key
                cnt = 1
    if prev is not None and cnt > 1:
        per_country_groups[prev[0]] += 1
        per_country_rows[prev[0]] += cnt
        top[prev[1]] += cnt
    os.remove(srt)
    return {"dup_name_groups": dict(per_country_groups), "rows_in_dup_names": dict(per_country_rows),
            "top_dup_names": [[k, v] for k, v in top.most_common(20)]}


def main():
    fit_ids = load_fit()
    print("fit ids:", len(fit_ids), flush=True)
    stats, raws = pass1(fit_ids)
    out = {}
    for src in ["S1", "S2", "S3"]:
        st = stats[src]
        d = {"rows": st["rows"],
             "empty": {c: dict(v) for c, v in st["empty"].items()},
             "name_len": {b: dict(v) for b, v in st["name_len"].items()},
             "addr_len": {b: dict(v) for b, v in st["addr_len"].items()},
             "name_toklen": {b: dict(v) for b, v in st["name_toklen"].items()},
             "addr_toklen": {b: dict(v) for b, v in st["addr_toklen"].items()}}
        d.update(dup_stats(raws[src]))
        d["top_name_tokens"] = {co: [[t, n] for t, n in c.most_common(120)]
                                for co, c in st["name_tok"].items()}
        d["top_addr_tokens"] = {co: [[t, n] for t, n in c.most_common(120)]
                                for co, c in st["addr_tok"].items()}
        out[src] = d
    with open(os.path.join(MET, "eda_struct.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("WROTE eda_struct.json", flush=True)


if __name__ == "__main__":
    main()
