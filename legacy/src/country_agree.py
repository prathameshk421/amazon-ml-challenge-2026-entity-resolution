"""Country agreement on 200K sampled GT rows (seed 42). Writes metrics/country_agree.json."""
import csv, json, os, random
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
rng = random.Random(42)
rows = []
with open(os.path.join(TR, "train_ground_truth.tsv"), newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if rng.random() < 200000 / 2206821:
            rows.append((row["source1_entity_id"], row["matched_entity_ids"]))
print("sampled gt rows:", len(rows))
s1c = {}
need_s1 = set(r[0] for r in rows)
need_pool = set()
for e, m in rows:
    if m:
        need_pool.update(m.split(","))
with open(os.path.join(TR, "train_source1.tsv"), newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["entity_id"] in need_s1:
            s1c[row["entity_id"]] = row["country"]
poolc = {}
for fn in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(TR, fn), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in need_pool:
                poolc[row["entity_id"]] = row["country"]
tot = agree = differ = missing = 0
for e, m in rows:
    if not m:
        continue
    for x in m.split(","):
        tot += 1
        pc = poolc.get(x, "")
        if pc == "" or s1c.get(e, "") == "":
            missing += 1
        elif pc == s1c[e]:
            agree += 1
        else:
            differ += 1
out = {"gt_rows": len(rows), "pairs": tot, "agree": agree, "differ": differ,
       "missing_country": missing, "agree_frac": agree / tot if tot else 0}
json.dump(out, open(os.path.join(ROOT, "artifacts/metrics/country_agree.json"), "w"), indent=1)
print(out)
