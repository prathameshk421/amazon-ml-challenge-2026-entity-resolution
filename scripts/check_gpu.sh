#!/bin/bash
# Environment verification. On the box: runs the one-line torch check, then cloud/verify_env.py (CUDA, VRAM,
# allocation, bf16/fp16 matmul, NCCL, all imports); exit 1 if anything fails.
#   scripts/check_gpu.sh            # on the box
#   scripts/check_gpu.sh --remote   # from the local machine, over SSH
set -euo pipefail
if [ "${1:-}" = --remote ]; then
  source "$(dirname "$0")/../cloud/lib.sh"
  gssh "bash /workspace/project/scripts/check_gpu.sh"
  exit $?
fi
source /workspace/env.sh
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory)"
nvidia-smi --query-gpu=name,driver_version,memory.total,utilization.gpu --format=csv
python cloud/verify_env.py --out /workspace/logs/verify_env.json
