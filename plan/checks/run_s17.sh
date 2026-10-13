#!/usr/bin/env bash
# S17 = S16 + distilled-e5 S1 query ranks 6-7 (D10 restricted to ranks 6-7; train +0.00004 on top of seed ranks 3-5,
# seed5_eval.json). H200 jobs. Base: full_c0_rgrp_t073 (threshold 0.73).
set -euo pipefail
PY=.venv/bin/python
export PYTHONPATH=submission/code/business_entity_resolution/src:ce:plan/checks
ts() { date "+%H:%M:%S"; }
RJOB=prathameshk421/6ab94dd052d0dbd7f1d9d136
echo ">>> $(ts) waiting for $RJOB"; hf jobs wait $RJOB || true
hf jobs logs $RJOB 2>&1 | grep -E "country|wrote|Traceback" | tail -4
$PY -c "from huggingface_hub import hf_hub_download as d; print(d('prathameshk421/ber-ce-scoring','retrieval/dracd7_test_topk.parquet',repo_type='dataset',local_dir='work'))"
$PY plan/checks/drace_test.py pairs_extra --topk-tag dracd7_test --set DRACD7_TEST --k1 7 --skip DRACD_TEST,DRACD5_TEST,SEED5_TEST
hf upload prathameshk421/ber-ce-scoring work/ce/pairs_DRACD7_TEST.parquet pairs_DRACD7_TEST.parquet --repo-type dataset --commit-message "DRACD7 test pairs"
CMD='pip install -q "transformers==5.17.0" "peft==0.21.0" pyarrow "huggingface_hub>=1.0" sentencepiece && python -c "from huggingface_hub import hf_hub_download as d; print(d(\"prathameshk421/ber-ce-scoring\",\"score_job.py\",repo_type=\"dataset\",local_dir=\"/tmp/j\"))" && python -u /tmp/j/score_job.py'
J=$(hf jobs run --detach --flavor h200 --timeout 20m --secrets HF_TOKEN -l name=dracd7-test-c0 \
  -e CE_ADAPTER=adapter_c0 -e CE_OUT=scores_c0 -e CE_SETS=DRACD7_TEST \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime bash -c "$CMD" 2>&1 | grep -o "id=[0-9a-f]*" | cut -d= -f2)
echo ">>> $(ts) scoring job $J"; hf jobs wait prathameshk421/$J || true
hf jobs logs prathameshk421/$J 2>&1 | grep -E "chunk|all sets|Traceback" | tail -3
$PY ce/hf_job/collect.py --ce ce_rec_c0 --prefix scores_c0 --sets DRACD7_TEST
echo ">>> $(ts) merge"
$PY plan/checks/drace_test.py merge --t 0.995 --base work/outputs/full_c0_rgrp_t073/matching_results.tsv --out submission/output_v17 \
  --tag drace_test_s17 --extra DRACD_TEST,DRACD5_TEST,SEED5_TEST,DRACD7_TEST
$PY student_resource/utils/validate_submission.py --matching submission/output_v17/matching_results.tsv \
  --candidate submission/output_v17/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
shasum submission/output_v17/matching_results.tsv
echo ">>> $(ts) DONE: submission/output_v17/"
