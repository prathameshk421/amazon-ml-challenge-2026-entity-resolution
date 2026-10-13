"""Freeze golden outputs of the reference normalizer for the executor's unit tests.
Writes plan/reference/goldens.json (inputs -> expected normalize_field output / skeleton)."""
import json, os, sys, unicodedata
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "plan/reference"))
from translit_ref import normalize_field, skeleton
D = os.path.join(ROOT, "student_resource/dataset/train")
tok_by_script = defaultdict(Counter)
with open(os.path.join(D, "train_source2.tsv"), encoding="utf-8") as f:
    f.readline()
    for i, line in enumerate(f):
        if i > 3000000: break
        name = line.split("\t")[1]
        for t in name.split():
            c = next((c for c in t if 0x0900 <= ord(c) <= 0x0D7F), None)
            if c:
                tok_by_script[unicodedata.name(c).split()[0]][t] += 1
cases = []
for sc, cnt in sorted(tok_by_script.items()):
    for t, _ in cnt.most_common(6):
        cases.append({"kind": "name", "in": t, "script": sc})
extra_names = ["ਅਨੰਦ", "ਇੰਡਸਟ੍ਰੀਜ਼", "ਪੱਕਾ", "ॐ साईं", "BORDEAUX CLUB S.A.S.", "J S W Steel", "राम मार्केटिंग प्राइवेट लिमिटेड", "चैंपियन", "ज़ू", "ಸನ್‌ರೈಸ್ ಕನ್ಸಲ್ಟಿಂಗ್", "സിൽവർ",
               "Café Crème SARL", "Œuvre d'Art", "O'Connor & Sons, Inc.", "<CITY_NAME> Homes Pvt Ltd",
               "HU DESERT ALTO LLC", "phaedrazengpegasus.com", "Jrm Sciences-<CITY_NAME>", "@n", ""]
extra_addrs = ["57 R. JEAN MOULIN STE MARIE", "12 Main St, St Louis", "Suite 5, 100 Saint Street", "2612 SOUTHVIEW COURT, <NULL>, BRAINERD, MN", "#009, A BLOCK, RAHEJA ENCLAVE236, RACE COURSE",
               "Nº 12 Rue de l'Église, Lille", "1945 214TH LN, BUCKEYE, AZ", "PLOT NO B-78/1, THANE, महाराष्ट्र", ""]
cases += [{"kind": "name", "in": s} for s in extra_names] + [{"kind": "addr", "in": s} for s in extra_addrs]
for c in cases:
    c["out"], c["had_indic"] = normalize_field(c["in"], c["kind"])
    if c["kind"] == "name":
        c["skeletons"] = [skeleton(t) for t in c["out"].split()]
json.dump(cases, open(os.path.join(ROOT, "plan/reference/goldens.json"), "w"), indent=1, ensure_ascii=False)
for c in cases: print(c["kind"], repr(c["in"])[:45].ljust(47), "->", repr(c["out"]), c.get("skeletons", ""))
