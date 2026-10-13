#!/usr/bin/env bash
# S16 (final, public LB 0.983958) = full_c0_rgrp at LightGBM threshold 0.73 + DRACE (untrained e5 S1@5 / seed@5, distilled
# e5 S1@2 and S1@5), c0 >= 0.995, one-owner. Assumes the S13 inputs exist (drace_test_*, dracd*_test_*, their c0 scores).
# S14 = the same chain with threshold 0.70 (run full_c0_rgrp, out output_v14).
set -euo pipefail
PY=.venv/bin/python
export PYTHONPATH=submission/code/business_entity_resolution/src:ce:plan/checks
ts() { date "+%H:%M:%S"; }
# 1. seed-expansion ranks 3-5 pairs from the existing drace_test_topk, c0 scoring on HF
$PY plan/checks/seed5_test_pairs.py
hf upload prathameshk421/ber-ce-scoring work/ce/pairs_SEED5_TEST.parquet pairs_SEED5_TEST.parquet --repo-type dataset --commit-message "SEED5 test pairs"
CMD='pip install -q "transformers==5.17.0" "peft==0.21.0" pyarrow "huggingface_hub>=1.0" sentencepiece && python -c "from huggingface_hub import hf_hub_download as d; print(d(\"prathameshk421/ber-ce-scoring\",\"score_job.py\",repo_type=\"dataset\",local_dir=\"/tmp/j\"))" && python -u /tmp/j/score_job.py'
J=$(hf jobs run --detach --flavor a100-large --timeout 30m --secrets HF_TOKEN -l name=seed5-test-c0 \
  -e CE_ADAPTER=adapter_c0 -e CE_OUT=scores_c0 -e CE_SETS=SEED5_TEST \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime bash -c "$CMD" 2>&1 | grep -o "id=[0-9a-f]*" | cut -d= -f2)
echo ">>> $(ts) scoring job $J"; hf jobs wait prathameshk421/$J || true
$PY ce/hf_job/collect.py --ce ce_rec_c0 --prefix scores_c0 --sets SEED5_TEST
# 2. matcher at threshold 0.73 (copy of full_c0_rgrp, decision.threshold changed only)
rm -rf work/match/runs/full_c0_rgrp_t073 && cp -R work/match/runs/full_c0_rgrp work/match/runs/full_c0_rgrp_t073
$PY -c "import json; p='work/match/runs/full_c0_rgrp_t073/models/matcher.json'; d=json.load(open(p)); d['decision']['threshold']=0.73; json.dump(d, open(p, 'w'), indent=1)"
$PY -m ber.cli exp-predict --run full_c0_rgrp_t073
# 3. DRACE merge + validation
$PY plan/checks/drace_test.py merge --t 0.995 --base work/outputs/full_c0_rgrp_t073/matching_results.tsv --out submission/output_v16 \
  --tag drace_test_s16 --extra DRACD_TEST,DRACD5_TEST,SEED5_TEST
$PY student_resource/utils/validate_submission.py --matching submission/output_v16/matching_results.tsv \
  --candidate submission/output_v16/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
shasum submission/output_v16/matching_results.tsv   # expect 77149494...
echo ">>> $(ts) DONE: submission/output_v16/"
