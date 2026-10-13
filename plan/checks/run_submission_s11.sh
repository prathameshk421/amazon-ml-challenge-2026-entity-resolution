#!/usr/bin/env bash
# S11: full_c0 + reverse-name ownership (rev) + nfreq, same training rows and split as S9 (full_c0), then DRACE on top.
#   bash plan/checks/run_submission_s11.sh 2>&1 | tee logs/run_submission_s11.log
set -euo pipefail
PY=.venv/bin/python
export PYTHONPATH=submission/code/business_entity_resolution/src:ce:plan/checks
ts() { date "+%H:%M:%S"; }
echo ">>> $(ts) train full_c0_rev"
REST_FOLD=1 EXCLUDE_SETS=DEV_REST_180K $PY plan/checks/train_final_comp.py full_c0_rev cex_p:ce_rec_c0 rev nfreq
echo ">>> $(ts) exp-predict full_c0_rev"
$PY -m ber.cli exp-predict --run full_c0_rev
echo ">>> $(ts) DRACE merge on full_c0_rev (t 0.995)"
$PY plan/checks/drace_test.py merge --t 0.995 --base work/outputs/full_c0_rev/matching_results.tsv --out submission/output_v11 --tag drace_test_s11
echo ">>> $(ts) official validator"
$PY student_resource/utils/validate_submission.py --matching submission/output_v11/matching_results.tsv \
  --candidate submission/output_v11/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
shasum submission/output_v11/matching_results.tsv
echo ">>> $(ts) DONE: submission/output_v11/"
