#!/usr/bin/env bash
# CE-derived feature ablation A..G (M020..M026) on MATCH_TRAIN 180K, 5-fold GroupKFold by S1, parent x1_strsim.
# p_ce = ce_rec, trained on DEV_REST_180K only, so out-of-sample on every MATCH_TRAIN S1.
#   bash plan/checks/run_cefe_ablation.sh 2>&1 | tee logs/run_cefe_ablation.log
set -uo pipefail
PY=.venv/bin/python
export PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks
for id in M020 M021 M022 M023 M024 M025 M026; do
  echo ">>> $(date +%H:%M:%S) exp-train $id"
  $PY -m ber.cli exp-train --exp exps/$id.json || echo "!!! $id FAILED"
done
echo ">>> $(date +%H:%M:%S) summary"
$PY plan/checks/cefe_ablation.py M018 M020 M021 M022 M023 M024 M025 M026
echo ">>> $(date +%H:%M:%S) DONE"
