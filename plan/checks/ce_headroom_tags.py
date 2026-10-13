"""Pattern tags over work/reports/errors/headroom_full_ce.parquet (plan/checks/ce_headroom.py) -> artifacts/metrics/ce_headroom_tags.json.
Run: PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/ce_headroom_tags.py"""
import json
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from ber import io_utils

d = pd.read_parquet(io_utils.work_path("reports", "errors", "headroom_full_ce.parquet"))
low = lambda s: s.fillna("").astype(str).str.lower()
sn, pn, sa, pa = low(d.s1_name), low(d.pool_name), low(d.s1_addr), low(d.pool_addr)
nt = np.array([fuzz.token_set_ratio(a, b) for a, b in zip(sn, pn)])
at = np.array([fuzz.token_set_ratio(a, b) for a, b in zip(sa, pa)])
empty = pa.str.strip().isin(["", "<null>", "null", "nan"])
tags = {
    "pool_addr_empty": empty,
    "pool_name_nonlatin": d.pool_name.fillna("").str.contains(r"[^\x00-\x7FÀ-ɏ]", regex=True),
    "pool_name_domain": pn.str.contains(r"www\.|\.com|\.in\b|\.net|\.org|\.co\b", regex=True),
    "pool_name_acronym(<=4 chars, no space)": pn.str.replace(r"[^a-z0-9]", "", regex=True).str.len().le(4) & ~pn.str.contains(" "),
    "alias(name<50, addr>=80, addr present)": (nt < 50) & (at >= 80) & ~empty,
    "name_tset>=95": nt >= 95,
    "name_tset_50_95(typo-ish)": (nt >= 50) & (nt < 95),
    "name_tset<50": nt < 50,
}
out = {}
for k in ["FP", "FN_in_list", "FN_cut", "FN_stage1"]:
    m = (d.kind == k).to_numpy()
    out[k] = {"n": int(m.sum()), **{t: round(float(v[m].mean()), 3) for t, v in tags.items()}}
    if k == "FN_in_list":
        out[k]["p_ce>=0.5_among_pool_addr_empty"] = round(float((d.p_ce[m & empty.to_numpy()] >= 0.5).mean()), 3)
io_utils.write_metrics("ce_headroom_tags", out, 0, 0, io_utils.load_config())
print(json.dumps(out, indent=1))
