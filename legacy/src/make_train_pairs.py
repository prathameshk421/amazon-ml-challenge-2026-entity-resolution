"""Phase 3a: build labeled training frame from fit-sample candidates.
Loads candidates/fit_sample_K150.parquet + pool mmap + GT truth; writes
artifacts/train_pairs.parquet (edge list + labels + s1/cand ids) for feature jobs.
Run: .venv/bin/python legacy/src/make_train_pairs.py
"""

import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(ROOT, "student_resource/dataset/train")
IDX = os.path.join(ROOT, "artifacts/index")
CAND = os.path.join(ROOT, "artifacts/candidates")
ART = os.path.join(ROOT, "artifacts")


def main():
    cand = pd.read_parquet(os.path.join(CAND, "fit_sample_K150.parquet"))
    src = np.load(os.path.join(IDX, "pool.src.npy"))
    num = np.load(os.path.join(IDX, "pool.num.npy"))
    ext_pool = np.array([("S2-" if s == 0 else "S3-") + str(int(n)) for s, n in zip(src, num)])
    truth = {}
    import csv
    want = set(cand["s1_id"].tolist())
    with open(os.path.join(TR, "train_ground_truth.tsv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["source1_entity_id"] in want:
                m = row["matched_entity_ids"]
                truth[row["source1_entity_id"]] = set(m.split(",")) if m else set()
    s1, cr, lab = [], [], []
    for r in cand.itertuples():
        t = truth.get(r.s1_id, set())
        for rowid in r.cand:
            s1.append(r.s1_id)
            cr.append(ext_pool[int(rowid)])
            lab.append(1 if ext_pool[int(rowid)] in t else 0)
    df = pd.DataFrame({"s1_id": s1, "cand_id": cr, "label": np.array(lab, dtype=np.int8)})
    df.to_parquet(os.path.join(ART, "train_pairs.parquet"))
    print("pairs=%d pos=%d posrate=%.4f entities=%d" % (len(df), int(df.label.sum()),
          df.label.mean(), df.s1_id.nunique()))


if __name__ == "__main__":
    main()
