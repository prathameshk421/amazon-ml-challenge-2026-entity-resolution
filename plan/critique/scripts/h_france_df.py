"""Critique job H: document frequency (test France pool names/addresses, structural only) of tokens in the
largest France test S1 name chains vs the plan's df cap (0.002 x test pool size 9,969,589)."""
import json, os
from collections import Counter
import pyarrow.parquet as pq
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "plan", "critique", "out")
p = pq.read_table(os.path.join(OUT, "norm_test_pool_france.parquet"), columns=["name_norm", "addr_norm"]).to_pydict()
s = pq.read_table(os.path.join(OUT, "norm_test_s1.parquet"), columns=["name_norm", "country"]).to_pydict()
fr = [n for n, c in zip(s["name_norm"], s["country"]) if c == "France"]
ch = Counter(fr)
dn, da = Counter(), Counter()
for n, a in zip(p["name_norm"], p["addr_norm"]):
    dn.update(set(n.split())); da.update(set(a.split()))
cap = int(0.002 * 9969589)
toks = sorted({t for n, _ in ch.most_common(40) for t in n.split()})
R = {"cap": cap, "france_s1_chain_ge10_share": sum(v for v in ch.values() if v >= 10) / len(fr),
     "top_chains": ch.most_common(15),
     "chain_token_df_france_pool_names": {t: dn[t] for t in toks},
     "chain_token_df_france_pool_addrs": {t: da[t] for t in toks},
     "share_top200_chain_names_all_tokens_capped_in_france_pool_alone": sum(
         1 for n, _ in ch.most_common(200) if all(dn[t] > cap for t in n.split())) / 200}
json.dump(R, open(os.path.join(OUT, "h_france_df.json"), "w"), indent=1)
print(json.dumps(R))
