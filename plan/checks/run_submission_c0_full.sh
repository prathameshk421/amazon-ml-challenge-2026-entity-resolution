#!/usr/bin/env bash
# Submission 9 candidate: full_c0 = 68 live features + ce_rec_c0 p_ce (cex_p), one LightGBM on
# MATCH_TRAIN + MATCH_VALID + DEV_REST_REST fold 1 (DEV_REST_180K and DEV_REST_REST fold 0 excluded: the
# cross-encoder trained there). Same seeded fit / early-stop / calibration split as full_single / full_ce.
#   bash plan/checks/run_submission_c0_full.sh 2>&1 | tee logs/run_submission_c0_full.log
set -euo pipefail
PY=.venv/bin/python
export PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks
JOB=prathameshk421/6ab8b8d752d0dbd7f1d9a670
ts() { date "+%H:%M:%S"; }
echo ">>> $(ts) waiting for $JOB (DEV_REST_REST fold 1 on L4)"
hf jobs wait $JOB || true
hf jobs logs $JOB 2>&1 | grep -E "all sets done|AUC|Error|Traceback" | tail -5
echo ">>> $(ts) collect DEV_REST_REST_F1"
$PY ce/hf_job/collect.py --ce ce_rec_c0 --prefix scores_c0 --sets DEV_REST_REST_F1
cp work/ce/ce_rec_c0/DEV_REST_REST_F1.parquet work/ce/ce_rec_c0/DEV_REST_REST.parquet
echo ">>> $(ts) waiting for submission 8 chain (one heavy job at a time)"
until grep -qE "DONE|Error|Traceback" logs/run_submission_c0.log; do sleep 20; done
echo ">>> $(ts) train full_c0"
REST_FOLD=1 EXCLUDE_SETS=DEV_REST_180K $PY plan/checks/train_final_comp.py full_c0 cex_p:ce_rec_c0
echo ">>> $(ts) exp-predict full_c0"
$PY -m ber.cli exp-predict --run full_c0
mkdir -p submission/output_v9
cp work/outputs/full_c0/matching_results.tsv submission/output_v9/matching_results.tsv
cp submission/output/candidate_pairs.tsv submission/output_v9/candidate_pairs.tsv
echo ">>> $(ts) official validator"
$PY student_resource/utils/validate_submission.py --matching submission/output_v9/matching_results.tsv \
  --candidate submission/output_v9/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
shasum submission/output_v9/matching_results.tsv
wc -l submission/output_v9/matching_results.tsv
echo ">>> $(ts) DONE: submission/output_v9/matching_results.tsv"
