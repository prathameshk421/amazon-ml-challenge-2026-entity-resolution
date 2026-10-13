#!/usr/bin/env bash
# Submission 2 chain: wait for the MATCH_ALL 5-fold measurement run (dev_rest_full.py) to exit, then
# train ONE full-data model and save it (train_final_single.py), rebuild TEST features with the current
# 68-column schema, score TEST, validate, and write submission/output_v2/. Run from the repository root.
set -euo pipefail
export PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks
PY=.venv/bin/python
ts() { date +%H:%M:%S; }

while pgrep -f "plan/checks/dev_rest_full.py" >/dev/null; do sleep 30; done
echo ">>> $(ts) dev_rest_full.py exited; last lines:"
tail -4 work/logs/dev_rest_full.log

echo ">>> $(ts) train_final_single.py"
$PY plan/checks/train_final_single.py

echo ">>> $(ts) match-features --set TEST"
$PY -m ber.cli match-features --set TEST

echo ">>> $(ts) match-predict"
$PY -m ber.cli match-predict

mkdir -p submission/output_v2
cp work/outputs/matching_results.tsv submission/output_v2/matching_results.tsv
cp submission/output/candidate_pairs.tsv submission/output_v2/candidate_pairs.tsv  # same TEST candidates (K_min 5)
echo ">>> $(ts) official validator"
$PY student_resource/utils/validate_submission.py \
  --matching submission/output_v2/matching_results.tsv \
  --candidate submission/output_v2/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
echo ">>> $(ts) DONE: submission/output_v2/{matching_results,candidate_pairs}.tsv"
