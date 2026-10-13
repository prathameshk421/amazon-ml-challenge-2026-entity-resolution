#!/bin/bash
# Local machine <-> GPU box over rsync/SSH (no git push needed; work-in-progress code stays local).
#   scripts/sync_results.sh push                 # code -> /workspace/project, CE pair parquet -> /workspace/data/ce_pairs
#   scripts/sync_results.sh pull                 # /workspace/{experiments,logs} -> work/gpu/ (checkpoint weights skipped)
#   scripts/sync_results.sh pull --with-weights  # also *.safetensors / *.bin
# pairs_TEST.parquet is uploaded too (inputs only, no labels) because CE scoring of test runs on the box.
set -euo pipefail
source "$(dirname "$0")/../cloud/lib.sh"
need_host
R="$SSH_USER@$GPU_IP"
RS=(rsync -az -e "ssh $SSH_OPTS")

case ${1:?push|pull} in
push)
  cd "$ROOT"
  { echo "commit $(git rev-parse HEAD)"; echo "branch $(git rev-parse --abbrev-ref HEAD)";
    echo "dirty_files $(git status --porcelain | grep -c .)";
    echo "worktree_sha $( (git diff HEAD; git ls-files --others --exclude-standard ce cloud scripts | xargs cat 2>/dev/null) | shasum | cut -c1-12)";
    echo "synced $(date -u +%FT%TZ)"; } > CODE_VERSION
  "${RS[@]}" --delete --exclude __pycache__ --exclude .DS_Store --exclude kaggle \
    ce cloud scripts requirements.txt AGENTS.md CODE_VERSION "$R:/workspace/project/"
  rm -f CODE_VERSION
  ssh $SSH_OPTS "$R" "mkdir -p /workspace/data/ce_pairs"
  "${RS[@]}" work/ce/pairs_*.parquet "$R:/workspace/data/ce_pairs/"
  ;;
pull)
  mkdir -p "$ROOT/work/gpu"
  EX=(); [ "${2:-}" = --with-weights ] || EX=(--exclude "*.safetensors" --exclude "*.bin" --exclude "*.pt")
  "${RS[@]}" "${EX[@]}" "$R:/workspace/experiments" "$R:/workspace/logs" "$ROOT/work/gpu/"
  echo "pulled to work/gpu/"
  ;;
*) echo "push|pull"; exit 2 ;;
esac
