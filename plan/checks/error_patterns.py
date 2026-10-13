"""Count named noise patterns in the MATCH_TRAIN OOF error exports (train S1 only).

Reads reports/match_errors/train_{false_negatives,false_positives,blocking_misses}.csv (written by
`match-train` from out-of-fold predictions) and writes artifacts/metrics/error_patterns.json.
No validation-S1 or test rows are read. Run from repo root: .venv/bin/python plan/checks/error_patterns.py
"""
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

ERR = Path("reports/match_errors")
OUT = Path("artifacts/metrics/error_patterns.json")

DOMAIN = re.compile(r"\.(?:com|in|net|org|co|fr)\b", re.I)
NONLATIN = re.compile(r"[^\x00-ɏ]")
LEADING_ZERO = re.compile(r"(?<!\d)0\d+")
ORDINAL = re.compile(r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
                     r"eleventh|twelfth)\b", re.I)
# compound house identifiers: 8-1-293/B/4/4, 86/1, H2-87, 815/A
HOUSE_ID = re.compile(r"(?<![\w])(?:[a-z]{0,2}\d+[a-z]?)(?:\s*[-/]\s*[0-9a-z]+)+", re.I)
UNIT = re.compile(r"\b(?:apt|apartment|suite|ste|unit|fl|floor|bldg|building|rm|room)\b\.?\s*#?\s*"
                  r"([0-9a-z]+)|#\s*([0-9a-z]+)", re.I)
PO_BOX = re.compile(r"\bp\.?\s*o\.?\s*box\s*(\d+)", re.I)


def _canon(s):
    return re.sub(r"\s", "", s.lower()).lstrip("0")


def _share(mask):
    return {"n": int(mask.sum()), "share": float(mask.mean())}


def patterns(df):
    pa = df["Address_Other"].astype(str).str.strip()
    empty = (pa == "") | (pa.str.contains("<NULL>", regex=False) & (pa.str.len() < 10))
    n_tset = np.array([fuzz.token_set_ratio(a.lower(), b.lower())
                       for a, b in zip(df["Name_S1"], df["Name_Other"])])
    a_tset = np.array([fuzz.token_set_ratio(a.lower(), b.lower()) if b.strip() else np.nan
                       for a, b in zip(df["Address_S1"], df["Address_Other"])])
    exact = n_tset >= 95
    alias = (n_tset < 50) & (a_tset >= 80)
    hid_conf = hid_shared = unit_conf = unit_shared = pobox_pool_only = 0
    for a, b in zip(df["Address_S1"], df["Address_Other"]):
        ha, hb = {_canon(m) for m in HOUSE_ID.findall(a)}, {_canon(m) for m in HOUSE_ID.findall(b)}
        if ha and hb:
            hid_shared += bool(ha & hb)
            hid_conf += not (ha & hb)
        ua = {(x or y).lower() for x, y in UNIT.findall(a)}
        ub = {(x or y).lower() for x, y in UNIT.findall(b)}
        if ua and ub:
            unit_shared += bool(ua & ub)
            unit_conf += not (ua & ub)
        pobox_pool_only += bool(PO_BOX.search(b)) and not PO_BOX.search(a)
    n = len(df)
    return {
        "n_rows": n,
        "country_S1": {k: int(v) for k, v in df["country_S1"].value_counts().items()},
        "pool_addr_empty": _share(empty),
        "name_tset_ge95": _share(exact),
        "name_tset_ge95_and_pool_addr_empty": _share(exact & empty),
        "alias_name_tset_lt50_addr_tset_ge80": _share(alias),
        "domain_like_pool_name": _share(df["Name_Other"].str.contains(DOMAIN)),
        "nonlatin_pool_name": _share(df["Name_Other"].str.contains(NONLATIN)),
        "nonlatin_pool_addr": _share(df["Address_Other"].str.contains(NONLATIN)),
        "leading_zero_number_pool_addr": _share(df["Address_Other"].str.contains(LEADING_ZERO)),
        "ordinal_word_pool_addr": _share(df["Address_Other"].str.contains(ORDINAL)),
        "house_id_both_present": {"conflict": hid_conf, "shared": hid_shared,
                                  "conflict_share_of_rows": hid_conf / n},
        "unit_both_present": {"conflict": unit_conf, "shared": unit_shared,
                              "conflict_share_of_rows": unit_conf / n},
        "po_box_pool_only": {"n": pobox_pool_only, "share": pobox_pool_only / n},
    }


def main():
    t0 = time.time()
    out = {"script": "plan/checks/error_patterns.py",
           "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
           "source": "reports/match_errors/*.csv (MATCH_TRAIN OOF, threshold 0.70 + one-owner)",
           "definitions": {
               "pool_addr_empty": "pool address blank or only <NULL>",
               "name_tset_ge95": "rapidfuzz token_set_ratio on lowercased raw names >= 95",
               "alias": "name token_set_ratio < 50 and address token_set_ratio >= 80",
               "house_id": "regex for compound ids like 8-1-293/B/4, 86/1, H2-87; compared after "
                           "removing spaces and leading zeros"}}
    for name in ("train_false_negatives", "train_false_positives", "train_blocking_misses"):
        df = pd.read_csv(ERR / f"{name}.csv", keep_default_na=False, dtype=str)
        out[name] = patterns(df)
    out["wall_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k.startswith("train_")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
