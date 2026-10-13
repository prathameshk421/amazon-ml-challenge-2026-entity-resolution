#!/usr/bin/env bash
# Submission 8 candidate: M027 (68 live features + ce_rec_c0 p_ce, MATCH_TRAIN 5-fold models) on TEST.
# Waits for HF job c0-score-test, pulls TEST scores, exp-predict, official validator.
#   bash plan/checks/run_submission_c0.sh 2>&1 | tee logs/run_submission_c0.log
set -euo pipefail
PY=.venv/bin/python
export PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks
JOB=prathameshk421/6ab8b4b56b030d633f6981af
ts() { date "+%H:%M:%S"; }
echo ">>> $(ts) waiting for $JOB"
hf jobs wait $JOB || true
hf jobs logs $JOB 2>&1 | grep -E "all sets done|AUC|Error|Traceback" | tail -5
echo ">>> $(ts) collect TEST (scores_c0)"
$PY ce/hf_job/collect.py --ce ce_rec_c0 --prefix scores_c0 --sets TEST
echo ">>> $(ts) exp-predict M027"
$PY -m ber.cli exp-predict --run M027
mkdir -p submission/output_v8
cp work/outputs/M027/matching_results.tsv submission/output_v8/matching_results.tsv
cp submission/output/candidate_pairs.tsv submission/output_v8/candidate_pairs.tsv
echo ">>> $(ts) official validator"
$PY student_resource/utils/validate_submission.py --matching submission/output_v8/matching_results.tsv \
  --candidate submission/output_v8/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
shasum submission/output_v8/matching_results.tsv
wc -l submission/output_v8/matching_results.tsv
echo ">>> $(ts) DONE: submission/output_v8/matching_results.tsv"
