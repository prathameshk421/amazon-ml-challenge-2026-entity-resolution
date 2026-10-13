#!/usr/bin/env bash
# S12 HF side: distilled-e5 test retrieval -> new S1@2 pairs -> c0 scoring -> collect.
set -euo pipefail
PY=.venv/bin/python
export PYTHONPATH=submission/code/business_entity_resolution/src:ce:plan/checks
ts() { date "+%H:%M:%S"; }
RJOB=prathameshk421/6ab901f452d0dbd7f1d9bbd0
echo ">>> $(ts) waiting for $RJOB"; hf jobs wait $RJOB || true
hf jobs logs $RJOB 2>&1 | grep -E "embedded|country|wrote|Traceback" | tail -6
$PY -c "from huggingface_hub import hf_hub_download as d; print(d('prathameshk421/ber-ce-scoring','retrieval/dracd_test_topk.parquet',repo_type='dataset',local_dir='work'))"
$PY plan/checks/drace_test.py pairs_extra --topk-tag dracd_test --set DRACD_TEST --k1 2
hf upload prathameshk421/ber-ce-scoring work/ce/pairs_DRACD_TEST.parquet pairs_DRACD_TEST.parquet --repo-type dataset --commit-message "DRACD test pairs"
CMD='pip install -q "transformers==5.17.0" "peft==0.21.0" pyarrow "huggingface_hub>=1.0" sentencepiece && python -c "from huggingface_hub import hf_hub_download as d; print(d(\"prathameshk421/ber-ce-scoring\",\"score_job.py\",repo_type=\"dataset\",local_dir=\"/tmp/j\"))" && python -u /tmp/j/score_job.py'
SJOB=$(hf jobs run --detach --flavor a100-large --timeout 30m --secrets HF_TOKEN -l name=dracd-test-c0 \
  -e CE_ADAPTER=adapter_c0 -e CE_OUT=scores_c0 -e CE_SETS=DRACD_TEST \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime bash -c "$CMD" 2>&1 | grep -o "id=[0-9a-f]*" | cut -d= -f2)
echo ">>> $(ts) scoring job $SJOB"; hf jobs wait prathameshk421/$SJOB || true
hf jobs logs prathameshk421/$SJOB 2>&1 | grep -E "chunk|all sets|Traceback" | tail -3
$PY ce/hf_job/collect.py --ce ce_rec_c0 --prefix scores_c0 --sets DRACD_TEST
echo ">>> $(ts) HF side DONE"
