#!/usr/bin/env bash
# Matcher experiment chain (2026-09-26): candidate width K_min x new address/uniqueness features.
# Needs work/candidates/{MATCH_TRAIN,MATCH_VALID} generated with the wide cut (K_min 20, K_max 60,
# p_min 0.05); narrower cuts are applied by match.json cand_filter. Every run uses the same 5 folds
# (GroupKFold over the same S1), so per-entity OOF scores are paired across runs.
# Run from the repository root:  bash plan/checks/k_feature_sweep.sh
set -euo pipefail
export PYTHONPATH=submission/code/business_entity_resolution/src
PY=.venv/bin/python
ber() { echo ">>> $(date +%H:%M:%S) ber $*"; "$PY" -m ber.cli "$@"; }
NEW='["n_shared_maxidf","a_shared_maxidf","a_partial","a_char3_jacc","a_num_minlev","a_numstreet",
"q_name_cnt","p_name_cnt","q_addr_cnt","p_addr_cnt","l_n_addr_close"]'

setcfg() {  # setcfg K_min drop_features_json
  "$PY" - "$1" "$2" <<'EOF'
import json, sys
p = "submission/code/business_entity_resolution/configs/match.json"
d = json.load(open(p))
d["cand_filter"] = {"K_min": int(sys.argv[1]), "p_min": 0.05}
d["drop_features"] = json.loads(sys.argv[2])
open(p, "w").write(json.dumps(d, indent=2) + "\n")
print("match.json:", d["cand_filter"], "drop", len(d["drop_features"]))
EOF
}

features() { ber match-features --set MATCH_TRAIN; ber match-features --set MATCH_VALID; }
train_eval() { ber match-train --run "$1"; ber match-eval --set MATCH_VALID; }
compare() { ber match-compare --a "$1" --b "$2" --set OOF; ber match-compare --a "$1" --b "$2" --set MATCH_VALID; }

setcfg 5 "$NEW";  features; train_eval k5_base
setcfg 5 '[]';    train_eval k5_feat;  compare k5_base k5_feat
setcfg 20 '[]';   features; train_eval k20_feat; compare k5_feat k20_feat
setcfg 10 '[]';   features; train_eval k10_feat; compare k5_feat k10_feat; compare k10_feat k20_feat
echo ">>> $(date +%H:%M:%S) sweep done"
