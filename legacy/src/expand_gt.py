"""Expand GT matched_entity_ids to one ID per line (streaming)."""
import csv
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
n_rows = n_ids = 0
with open(os.path.join(ROOT, "student_resource/dataset/train/train_ground_truth.tsv"),
          newline="", encoding="utf-8") as f, \
     open(os.path.join(ROOT, "artifacts/tmp/gt_matched.raw"), "w", encoding="utf-8") as out:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        n_rows += 1
        m = row["matched_entity_ids"]
        if m:
            for x in m.split(","):
                if x:
                    out.write(x + "\n")
                    n_ids += 1
print("rows=%d ids=%d" % (n_rows, n_ids))
