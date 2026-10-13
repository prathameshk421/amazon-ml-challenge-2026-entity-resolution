"""Pattern tags + loss attribution over work/reports/errors/errors_full_comp.parquet (plan/checks/error_report.py).
Writes artifacts/metrics/error_report_tags.json. Run: PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/error_report_tags.py"""
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from ber import io_utils

df = pd.read_parquet(io_utils.work_path("reports", "errors", "errors_full_comp.parquet"))
low = lambda s: s.fillna("").str.lower()
sn, pn, sa, pa = low(df.s1_name), low(df.pool_name), low(df.s1_addr), low(df.pool_addr)
nt = np.array([fuzz.token_set_ratio(a, b) for a, b in zip(sn, pn)])
at = np.array([fuzz.token_set_ratio(a, b) for a, b in zip(sa, pa)])
empty = lambda s: s.str.strip().isin(["", "<null>"])
nonlatin = lambda s: s.str.contains(r"[^\x00-\x7F]", regex=True)
num = lambda s: s.str.findall(r"\d+").map(lambda x: frozenset(t.lstrip("0") for t in x))
sn_num, pn_num = num(sa), num(pa)
tags = {
    "pool_addr_empty": empty(pa), "s1_addr_empty": empty(sa),
    "name_tset_ge95": nt >= 95, "name_tset_lt50": nt < 50,
    "alias_name_lt50_addr_ge80": (nt < 50) & (at >= 80),
    "same_name_diff_addr(name>=95,addr<60)": (nt >= 95) & (at < 60) & ~empty(pa),
    "nonlatin_either": nonlatin(df.s1_name.fillna("") + df.pool_name.fillna("") + df.pool_addr.fillna("")),
    "number_conflict(both have nums, disjoint)": np.array([bool(a) and bool(b) and not (a & b) for a, b in zip(sn_num, pn_num)]),
    "country_differs": df.s1_country.fillna("").str.lower() != df.pool_country.fillna("").str.lower(),
}
out = {}
for kind in ["FP", "FN", "FN_MISSED"]:
    m = (df.kind == kind).to_numpy()
    out[kind] = {"n": int(m.sum()), "by_country": df[m].s1_country.value_counts().to_dict(),
                 "tags": {k: round(float(np.asarray(v)[m].mean()), 4) for k, v in tags.items()}}
    if kind != "FN_MISSED":
        out[kind]["p_cal_quantiles"] = df[m].p_cal.quantile([.1, .25, .5, .75, .9]).round(3).to_dict()
# loss attribution per S1: which error kinds it has
per = df.groupby("s1_idx").agg(f=("f05_entity", "first"), kinds=("kind", lambda k: "+".join(sorted(set(k)))))
per["loss"] = 1 - per.f
out["loss_by_error_combo"] = per.groupby("kinds").loss.agg(["count", "sum"]).round(1).sort_values("sum", ascending=False).to_dict("index")
out["loss_total"] = round(float(per.loss.sum()), 1)
out["n_true_of_missed_s1"] = df[df.kind == "FN_MISSED"].n_true.describe().round(2).to_dict()
io_utils.write_metrics("error_report_tags", out, int(df.s1_idx.nunique()), 0)
import json; print(json.dumps(out, indent=1, default=str))
