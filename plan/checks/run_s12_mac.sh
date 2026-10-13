#!/usr/bin/env bash
# S12 Mac side: wait for full grp features -> train full_c0_rgrp (c0 + rev + nfreq + grp_full) -> TEST prediction.
set -euo pipefail
PY=.venv/bin/python
export PYTHONPATH=submission/code/business_entity_resolution/src:ce:plan/checks
ts() { date "+%H:%M:%S"; }
until grep -q "grp TEST" logs/grp_full.log 2>/dev/null; do sleep 20; done
echo ">>> $(ts) train full_c0_rgrp"
REST_FOLD=1 EXCLUDE_SETS=DEV_REST_180K $PY plan/checks/train_final_comp.py full_c0_rgrp cex_p:ce_rec_c0 rev nfreq grp_full
echo ">>> $(ts) exp-predict full_c0_rgrp"
$PY -m ber.cli exp-predict --run full_c0_rgrp
echo ">>> $(ts) Mac side DONE"
