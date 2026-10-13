#!/usr/bin/env bash
# S13 = S12 with the distilled-e5 S1 query widened to top-5 (R1 variant C).
set -euo pipefail
PY=.venv/bin/python
export PYTHONPATH=submission/code/business_entity_resolution/src:ce:plan/checks
ts() { date "+%H:%M:%S"; }
RJOB=prathameshk421/6ab9087752d0dbd7f1d9be6d
echo ">>> $(ts) waiting for $RJOB"; hf jobs wait $RJOB || true
hf jobs logs $RJOB 2>&1 | grep -E "country|wrote|Traceback" | tail -4
$PY -c "from huggingface_hub import hf_hub_download as d; print(d('prathameshk421/ber-ce-scoring','retrieval/dracd5_test_topk.parquet',repo_type='dataset',local_dir='work'))"
$PY plan/checks/drace_test.py pairs_extra --topk-tag dracd5_test --set DRACD5_TEST --k1 5 --skip DRACD_TEST
hf upload prathameshk421/ber-ce-scoring work/ce/pairs_DRACD5_TEST.parquet pairs_DRACD5_TEST.parquet --repo-type dataset --commit-message "DRACD5 test pairs"
CMD='pip install -q "transformers==5.17.0" "peft==0.21.0" pyarrow "huggingface_hub>=1.0" sentencepiece && python -c "from huggingface_hub import hf_hub_download as d; print(d(\"prathameshk421/ber-ce-scoring\",\"score_job.py\",repo_type=\"dataset\",local_dir=\"/tmp/j\"))" && python -u /tmp/j/score_job.py'
SJOB=$(hf jobs run --detach --flavor a100-large --timeout 30m --secrets HF_TOKEN -l name=dracd5-test-c0 \
  -e CE_ADAPTER=adapter_c0 -e CE_OUT=scores_c0 -e CE_SETS=DRACD5_TEST \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime bash -c "$CMD" 2>&1 | grep -o "id=[0-9a-f]*" | cut -d= -f2)
echo ">>> $(ts) scoring job $SJOB"; hf jobs wait prathameshk421/$SJOB || true
hf jobs logs prathameshk421/$SJOB 2>&1 | grep -E "chunk|all sets|Traceback" | tail -3
$PY ce/hf_job/collect.py --ce ce_rec_c0 --prefix scores_c0 --sets DRACD5_TEST
echo ">>> $(ts) merge"
$PY plan/checks/drace_test.py merge --t 0.995 --base work/outputs/full_c0_rgrp/matching_results.tsv --out submission/output_v13 --tag drace_test_s13 --extra DRACD_TEST,DRACD5_TEST
$PY student_resource/utils/validate_submission.py --matching submission/output_v13/matching_results.tsv \
  --candidate submission/output_v13/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
shasum submission/output_v13/matching_results.tsv
echo ">>> $(ts) DONE: submission/output_v13/"
