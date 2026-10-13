#!/bin/bash
# Runs ON the GPU box (start_gpu.sh calls it after mounting /workspace). Idempotent: the venv, HF cache and all
# outputs live on the persistent /workspace volume, so later sessions only re-link ~/.bashrc.
set -euo pipefail
W=/workspace
mountpoint -q $W || { echo "$W is not mounted"; exit 1; }
mkdir -p $W/{project,data,models,checkpoints,experiments,hf_cache,logs}
LOG=$W/logs/setup_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

# DLAMI keeps torch (CUDA build) in /opt/pytorch; the venv inherits it via --system-site-packages, so pip never
# replaces the CUDA wheel. The path is fixed by AMI_ID in cloud/gpu.env.
BASE=/opt/pytorch/bin/python
[ -x $BASE ] || BASE=$(command -v python3)
if [ ! -x $W/venv/bin/python ]; then
  echo "creating $W/venv from $BASE ($($BASE -c 'import torch; print("torch", torch.__version__)'))"
  $BASE -m venv --system-site-packages $W/venv
fi
$W/venv/bin/pip install -q --upgrade pip
$W/venv/bin/pip install -q -r $W/project/cloud/requirements-gpu.txt
$W/venv/bin/pip freeze > $W/logs/pip_freeze.txt

cat > $W/env.sh <<'EOF'
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CE_PAIRS=/workspace/data/ce_pairs
source /workspace/venv/bin/activate
cd /workspace/project
EOF
grep -q "workspace/env.sh" ~/.bashrc || echo '[ -f /workspace/env.sh ] && source /workspace/env.sh' >> ~/.bashrc
command -v tmux >/dev/null || sudo apt-get install -y -q tmux
echo "setup done: $(du -sh $W/venv | cut -f1) venv, $(df -h $W | awk 'NR==2{print $4}') free on $W"
