"""Phase 0 GT integrity checks via sort/comm (low RAM), writes JSON.
Writes artifacts/metrics/gt_checks.json. Run: .venv/bin/python legacy/src/phase0_gtcheck.py
"""

import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TMP = os.path.join(ROOT, "artifacts/tmp")
MET = os.path.join(ROOT, "artifacts/metrics")


def run(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        raise RuntimeError("FAILED %r: %s" % (cmd, p.stderr[-2000:]))
    return p.stdout.strip()


def main():
    s1_n = int(run("wc -l < artifacts/tmp/s1.ids"))
    gt_n = int(run("wc -l < artifacts/tmp/gt_s1.ids"))
    s1_gt_diff = int(run("comm -3 artifacts/tmp/s1.ids artifacts/tmp/gt_s1.ids | wc -l"))
    raw = int(run("wc -l < artifacts/tmp/gt_matched.raw"))
    uniq = int(run("wc -l < artifacts/tmp/gt_matched.ids"))
    missing = int(run("comm -23 artifacts/tmp/gt_matched.ids artifacts/tmp/pool.ids | wc -l"))
    out = {
        "s1_unique_ids": s1_n,
        "gt_s1_unique_ids": gt_n,
        "s1_minus_gt_diff_lines": s1_gt_diff,
        "s1_equals_gt_sets": (s1_gt_diff == 0 and s1_n == gt_n),
        "gt_matched_raw": raw,
        "gt_matched_unique": uniq,
        "each_record_at_most_one_owner": (raw == uniq),
        "gt_ids_missing_from_pool": missing,
    }
    with open(os.path.join(MET, "gt_checks.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
