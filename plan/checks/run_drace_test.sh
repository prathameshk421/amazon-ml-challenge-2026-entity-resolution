#!/usr/bin/env bash
# DRACE on TEST after the retrieval job: pairs -> c0 scoring (HF a100) -> merge on S9 -> official validator.
#   bash plan/checks/run_drace_test.sh 2>&1 | tee logs/run_drace_test.log
set -euo pipefail
PY=.venv/bin/python
export PYTHONPATH=submission/code/business_entity_resolution/src:ce:plan/checks
RJOB=prathameshk421/6ab8de096b030d633f698827
ts() { date "+%H:%M:%S"; }
if [ -z "${RESUME:-}" ]; then echo ">>> $(ts) waiting for retrieval $RJOB"
hf jobs wait $RJOB || true
hf jobs logs $RJOB 2>&1 | grep -E "embedded|country|wrote|uploaded|Error|Traceback" | tail -8
$PY -c "from huggingface_hub import hf_hub_download as d; print(d('prathameshk421/ber-ce-scoring','retrieval/drace_test_topk.parquet',repo_type='dataset',local_dir='work'))"
fi
echo ">>> $(ts) build new pairs (S1@5 + seed@2)"
$PY plan/checks/drace_test.py pairs --k1 5 --k2 2
hf upload prathameshk421/ber-ce-scoring work/ce/pairs_DRACE_TEST.parquet pairs_DRACE_TEST.parquet --repo-type dataset --commit-message "DRACE test pairs"
CMD='pip install -q "transformers==5.17.0" "peft==0.21.0" pyarrow "huggingface_hub>=1.0" sentencepiece && python -c "from huggingface_hub import hf_hub_download as d; print(d(\"prathameshk421/ber-ce-scoring\",\"score_job.py\",repo_type=\"dataset\",local_dir=\"/tmp/j\"))" && python -u /tmp/j/score_job.py'
SJOB=$(hf jobs run --detach --flavor a100-large --timeout 60m --secrets HF_TOKEN -l name=drace-test-c0 \
  -e CE_ADAPTER=adapter_c0 -e CE_OUT=scores_c0 -e CE_SETS=MATCH_VALID,DRACE_TEST \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime bash -c "$CMD" 2>&1 | grep -o "id=[0-9a-f]*" | cut -d= -f2)
echo ">>> $(ts) scoring job $SJOB"
hf jobs wait prathameshk421/$SJOB || true
hf jobs logs prathameshk421/$SJOB 2>&1 | grep -E "AUC|all sets|Error|Traceback" | tail -4
$PY ce/hf_job/collect.py --ce ce_rec_c0 --prefix scores_c0 --sets DRACE_TEST
echo ">>> $(ts) merge on S9 (t 0.995)"
$PY plan/checks/drace_test.py merge --t 0.995
echo ">>> $(ts) official validator"
$PY student_resource/utils/validate_submission.py --matching submission/output_v10/matching_results.tsv \
  --candidate submission/output_v10/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
shasum submission/output_v10/matching_results.tsv
echo ">>> $(ts) DONE: submission/output_v10/"
