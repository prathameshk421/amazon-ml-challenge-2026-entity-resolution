#!/usr/bin/env bash
# D10 test chain (distilled-e5 S1 query top-10). NOT RUN to completion: stopped for HF credit (RESEARCH_LOG D10, S14-file).
set -euo pipefail
PY=.venv/bin/python
export PYTHONPATH=submission/code/business_entity_resolution/src:ce:plan/checks
ts() { date "+%H:%M:%S"; }
RJOB=prathameshk421/6ab9375f6b030d633f6999ce
echo ">>> $(ts) waiting for $RJOB"; hf jobs wait $RJOB || true
hf jobs logs $RJOB 2>&1 | grep -E "country|wrote|Traceback" | tail -4
$PY -c "from huggingface_hub import hf_hub_download as d; print(d('prathameshk421/ber-ce-scoring','retrieval/dracd10_test_topk.parquet',repo_type='dataset',local_dir='work'))"
$PY plan/checks/drace_test.py pairs_extra --topk-tag dracd10_test --set DRACD10_TEST --k1 10 --skip DRACD_TEST,DRACD5_TEST
# split into two halves by row for parallel scoring
$PY - <<'PY'
import pyarrow.parquet as pq, pandas as pd
t = pq.read_table("work/ce/pairs_DRACD10_TEST.parquet"); m = pd.read_parquet("work/retrieval/dracd10_test_meta.parquet")
h = t.num_rows // 2
for n, sl in (("A", slice(0, h)), ("B", slice(h, t.num_rows))):
    pq.write_table(t.slice(sl.start, sl.stop - sl.start), f"work/ce/pairs_DRACD10{n}_TEST.parquet", compression="zstd")
    m.iloc[sl].reset_index(drop=True).to_parquet(f"work/retrieval/dracd10{n.lower()}_test_meta.parquet")
print("split", t.num_rows, h)
PY
CMD='pip install -q "transformers==5.17.0" "peft==0.21.0" pyarrow "huggingface_hub>=1.0" sentencepiece && python -c "from huggingface_hub import hf_hub_download as d; print(d(\"prathameshk421/ber-ce-scoring\",\"score_job.py\",repo_type=\"dataset\",local_dir=\"/tmp/j\"))" && python -u /tmp/j/score_job.py'
JOBS=()
for n in A B; do
  hf upload prathameshk421/ber-ce-scoring work/ce/pairs_DRACD10${n}_TEST.parquet pairs_DRACD10${n}_TEST.parquet --repo-type dataset --commit-message "DRACD10${n} test pairs"
  J=$(hf jobs run --detach --flavor a100-large --timeout 45m --secrets HF_TOKEN -l name=dracd10${n}-test-c0 \
    -e CE_ADAPTER=adapter_c0 -e CE_OUT=scores_c0 -e CE_SETS=DRACD10${n}_TEST \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime bash -c "$CMD" 2>&1 | grep -o "id=[0-9a-f]*" | cut -d= -f2)
  echo ">>> $(ts) scoring job $n $J"; JOBS+=($J)
done
for J in "${JOBS[@]}"; do hf jobs wait prathameshk421/$J || true; hf jobs logs prathameshk421/$J 2>&1 | grep -E "chunk|all sets|Traceback" | tail -2; done
$PY ce/hf_job/collect.py --ce ce_rec_c0 --prefix scores_c0 --sets DRACD10A_TEST,DRACD10B_TEST
echo ">>> $(ts) merge"
$PY plan/checks/drace_test.py merge --t 0.995 --base work/outputs/full_c0_rgrp/matching_results.tsv --out submission/output_v14 --tag drace_test_s14 --extra DRACD_TEST,DRACD5_TEST,DRACD10A_TEST,DRACD10B_TEST
$PY student_resource/utils/validate_submission.py --matching submission/output_v14/matching_results.tsv \
  --candidate submission/output_v14/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
shasum submission/output_v14/matching_results.tsv
echo ">>> $(ts) DONE: submission/output_v14/"
