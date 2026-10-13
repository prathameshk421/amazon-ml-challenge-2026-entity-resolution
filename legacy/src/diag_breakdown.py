"""Breakdown of blocking losses from artifacts/diag_pairs.parquet (diag_blocking.py output).
Writes artifacts/metrics/diag_breakdown.json.
"""

import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
df = pd.read_parquet(os.path.join(ROOT, "artifacts/diag_pairs.parquet"))
r = df[df.reach == 1]
silent = (r.v_EX == 0) & (r.v_NT == 0) & (r.v_TR == 0)
lost = r[r.rank_F1_exact10 > 150]
lsil = (lost.v_EX == 0) & (lost.v_NT == 0) & (lost.v_TR == 0)
deep = df[(df.reach == 0) | (df.rank_F1_exact10 > 2000)]
dr = deep[deep.reach == 1]
dsil = (dr.v_EX == 0) & (dr.v_NT == 0) & (dr.v_TR == 0)
dc = dr[dr.chain >= 10]
out = {
    "n_pairs": int(len(df)),
    "share_nonlatin_or_mixed_name": round(float(df.name_cls.isin(["nonlatin", "mixed"]).mean()), 4),
    "share_empty_addr": round(float((df.addr_cls == "empty").mean()), 4),
    "share_chain10plus": round(float((df.chain >= 10).mean()), 4),
    "reachable_name_silent": int(silent.sum()),
    "lost_at150_F1": {"n": int(len(lost)), "name_silent": int(lsil.sum()),
                      "nonlatin": int((lost.name_cls == "nonlatin").sum()),
                      "addr_empty": int((lost.addr_cls == "empty").sum()),
                      "chain10plus": int((lost.chain >= 10).sum()),
                      "addr_channel_silent": int((lost.v_AD == 0).sum())},
    "deep_gt2000_or_unreachable": {"n": int(len(deep)),
                                   "nonlatin_or_mixed": int(deep.name_cls.isin(["nonlatin", "mixed"]).sum()),
                                   "addr_empty": int((deep.addr_cls == "empty").sum()),
                                   "chain10plus": int((deep.chain >= 10).sum()),
                                   "reachable": int(len(dr)), "reachable_name_silent": int(dsil.sum()),
                                   "chain10plus_reachable": int(len(dc)),
                                   "chain10plus_exact_voted": int((dc.v_EX > 0).sum()),
                                   "chain10plus_nametoken_voted": int((dc.v_NT > 0).sum())},
    "chain_channel_vote_rate": {
        str(b): {c: round(float((g[c] > 0).mean()), 3) for c in ["v_EX", "v_NT", "v_AD"]}
        for b, g in r.assign(b=pd.cut(r.chain, [0, 1, 9, 10 ** 9], labels=["1", "2-9", "10+"]))
        .groupby("b", observed=True)},
}
json.dump(out, open(os.path.join(ROOT, "artifacts/metrics/diag_breakdown.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
