"""Compute the edge-token set G on the TRAIN pool with the reference normalizer (sanity check of
the §5.5 rule) and print key goldens for 3 records. Writes plan/reference/keys_check.json."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "plan/reference"))
from translit_ref import normalize_field
from keys_ref import name_keys, addr_keys, edge_token_set, key64
D = os.path.join(ROOT, "student_resource/dataset/train")
names = []
CACHE = os.path.join(ROOT, "plan/reference/keys_check.json")
for s in (() if os.path.exists(CACHE) else ("2", "3")):
    with open(os.path.join(D, "train_source%s.tsv" % s), encoding="utf-8") as f:
        f.readline()
        for line in f:
            names.append(normalize_field(line.split("\t")[1])[0])
if names:
    G = edge_token_set(names, len(names)); NPOOL = len(names)
else:  # reuse the G computed on the full train pool in the first run
    _c = json.load(open(CACHE)); G = _c["G"]; NPOOL = _c["n_pool"]
ex = {}
for raw in ["Phaedra Zeng Pegasus Corp", "phaedrazengpegasus.com", "HU Desert Alto LLC", "HU DESERT ALTO", "Happy Nails LLC", "happynails.com", "राम मार्केटिंग प्राइवेट लिमिटेड"]:
    nn = normalize_field(raw)[0]
    ex[raw] = {"norm": nn, "query": name_keys(nn, set(G), "query"), "pool_NC": name_keys(nn, set(G), "pool")["NC"]}
out = {"n_pool": NPOOL, "G_size": len(G), "G": G, "examples": ex,
       "key64_golden": {"EN|ram marketing private limited": key64("EN", "ram marketing private limited")}}
json.dump(out, open(os.path.join(ROOT, "plan/reference/keys_check.json"), "w"), indent=1, ensure_ascii=False)
print("G size", len(G)); print(G)
for k, v in ex.items(): print(k, '->', v["norm"], '| NC q', v["query"]["NC"], '| NC pool', v["pool_NC"], '| NP', v["query"]["NP"][:3], '| SP', v["query"]["SP"][:3])
