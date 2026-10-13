#!/usr/bin/env bash
# Submission 7 candidate: full_ce (live features + ce:ce_rec) + nfreq (M019), MATCH_ALL minus DEV_REST_180K,
# then TEST prediction + official validator.
#   bash plan/checks/run_submission_ce_nfreq.sh 2>&1 | tee logs/run_submission_ce_nfreq.log
set -euo pipefail
PY=.venv/bin/python
export PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks
ts() { date "+%H:%M:%S"; }
echo ">>> $(ts) train full_ce_nf"
EXCLUDE_SETS=DEV_REST_180K $PY plan/checks/train_final_comp.py full_ce_nf ce:ce_rec nfreq
echo ">>> $(ts) exp-predict full_ce_nf"
$PY -m ber.cli exp-predict --run full_ce_nf
mkdir -p submission/output_v7
cp work/outputs/full_ce_nf/matching_results.tsv submission/output_v7/matching_results.tsv
cp submission/output/candidate_pairs.tsv submission/output_v7/candidate_pairs.tsv
echo ">>> $(ts) official validator"
$PY student_resource/utils/validate_submission.py --matching submission/output_v7/matching_results.tsv \
  --candidate submission/output_v7/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
shasum submission/output_v7/matching_results.tsv
echo ">>> $(ts) DONE: submission/output_v7/matching_results.tsv"
