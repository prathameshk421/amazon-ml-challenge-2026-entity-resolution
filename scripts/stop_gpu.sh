#!/bin/bash
# Local machine. Pull results, unmount /workspace, terminate the GPU instance, confirm nothing is left billing
# except the /workspace volume (~$0.09/GB-month).
#   scripts/stop_gpu.sh              # pull experiments/logs (no weights) first
#   scripts/stop_gpu.sh --no-sync    # terminate immediately
set -euo pipefail
source "$(dirname "$0")/../cloud/lib.sh"
log_to gpu_stop
LIVE=$(live_instance)
if [ -z "$LIVE" ]; then echo "no live $NAME instance"; exit 0; fi
IID=$(echo "$LIVE" | awk '{print $1}')
echo "live: $LIVE"
if [ "${1:-}" != --no-sync ]; then
  bash "$ROOT/scripts/sync_results.sh" pull || echo "WARNING: pull failed; terminating anyway (data stays on the volume)"
fi
gssh "sync; sudo umount /workspace" 2>/dev/null || echo "umount skipped (instance unreachable)"
$A ec2 terminate-instances --instance-ids "$IID" --query 'TerminatingInstances[0].CurrentState.Name' --output text
$A ec2 wait instance-terminated --instance-ids "$IID"
if [ -f "$HOST_FILE" ]; then
  source "$HOST_FILE"
  S=$(( $(date -u +%s) - $(date -j -u -f %Y-%m-%dT%H:%M:%SZ "$GPU_STARTED" +%s 2>/dev/null || date -u -d "$GPU_STARTED" +%s) ))
  awk -v s="$S" -v p="$PRICE_PER_HOUR" 'BEGIN{printf "session: %.2f h, ~$%.2f at $%s/h\n", s/3600, s/3600*p, p}'
  rm -f "$HOST_FILE"
fi
echo "terminated $IID; live instances now: $(live_instance | wc -l | tr -d ' ')"
$A ec2 describe-volumes --volume-ids "$(volume_id)" --query 'Volumes[0].[VolumeId,State,Size]' --output text
