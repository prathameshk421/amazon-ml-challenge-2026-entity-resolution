#!/usr/bin/env bash
# Submission B: LightGBM (live full_single features + ce:ce_rec p_ce group) on MATCH_ALL minus DEV_REST_180K
# (the cross-encoder's training S1, where p_ce is in-sample), then TEST prediction + official validator.
# Waits for the HF scoring job of MATCH_TRAIN + DEV_REST_REST first.
#   bash plan/checks/run_submission_ce.sh <hf_job_id> 2>&1 | tee logs/run_submission_ce.log
set -euo pipefail
JOB=${1:?hf job id}
PY=.venv/bin/python
export PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks
ts() { date "+%H:%M:%S"; }

echo ">>> $(ts) waiting for HF job $JOB"
until .venv/bin/hf jobs ps -a | grep "${JOB##*/}" | grep -q "ERROR\|CANCELED\|COMPLETED"; do sleep 60; done
.venv/bin/hf jobs ps -a | grep "${JOB##*/}"
.venv/bin/hf jobs ps -a | grep "${JOB##*/}" | grep -q COMPLETED || { echo "HF job did not complete"; exit 1; }

echo ">>> $(ts) collect MATCH_TRAIN, DEV_REST_REST"
$PY ce/hf_job/collect.py --sets MATCH_TRAIN,DEV_REST_REST 2>&1 | grep -v "it/s\|B/s"
test -f work/ce/ce_rec/MATCH_TRAIN.parquet && test -f work/ce/ce_rec/DEV_REST_REST.parquet

echo ">>> $(ts) train full_ce"
EXCLUDE_SETS=DEV_REST_180K $PY plan/checks/train_final_comp.py full_ce ce:ce_rec

echo ">>> $(ts) exp-predict full_ce"
$PY -m ber.cli exp-predict --run full_ce

mkdir -p submission/output_v6
cp work/outputs/full_ce/matching_results.tsv submission/output_v6/matching_results.tsv
cp submission/output/candidate_pairs.tsv submission/output_v6/candidate_pairs.tsv
echo ">>> $(ts) official validator"
$PY student_resource/utils/validate_submission.py --matching submission/output_v6/matching_results.tsv \
  --candidate submission/output_v6/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
shasum submission/output_v6/matching_results.tsv
echo ">>> $(ts) DONE: submission/output_v6/matching_results.tsv"
