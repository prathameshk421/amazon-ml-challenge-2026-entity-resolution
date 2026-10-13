#!/usr/bin/env bash
# Stage-3 matcher, end to end (plan/MATCHER_PLAN.md). Run from the repository root: the folder that
# holds student_resource/ and work/. Needs the bucketing outputs listed in README.md ("Matcher only").
#
#   bash submission/code/business_entity_resolution/run_matcher.sh
#   WORKERS=4 PYTHON=.venv/bin/python bash submission/code/business_entity_resolution/run_matcher.sh
#
# Scoring val-S1 (HOLDOUT_TEST) is NOT part of this script: run it once, by hand, when the matcher is
# final (README step M6). Nothing here reads validation or test labels.
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
W=()
if [[ -n "${WORKERS:-}" ]]; then W=(--workers "$WORKERS"); fi
ber() { echo ">>> ber $*"; "$PY" -m ber.cli "$@"; }

ber match-features --set MATCH_TRAIN "${W[@]}"
ber match-features --set MATCH_VALID "${W[@]}"
ber match-train "${W[@]}"
ber match-eval --set MATCH_VALID
ber match-features --set TEST "${W[@]}"
ber match-predict
echo "done: work/outputs/matching_results.tsv (metrics in artifacts/metrics/match_*.json)"
