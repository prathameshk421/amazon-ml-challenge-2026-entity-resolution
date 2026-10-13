# Shared helpers for the local-side scripts. Sourced, not executed.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/cloud/gpu.env"
HOST_FILE=$ROOT/logs/gpu_host.env
A="aws --region $REGION"
SSH_OPTS="-i $KEY_FILE -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$ROOT/logs/gpu_known_hosts -o ServerAliveInterval=30"

log_to() {  # log_to <name>: tee all output of the calling script to logs/<name>_<time>.log
  mkdir -p "$ROOT/logs"
  LOG=$ROOT/logs/$1_$(date +%Y%m%d_%H%M%S).log
  exec > >(tee -a "$LOG") 2>&1
  echo "log: $LOG"
}

live_instance() {  # prints "<id> <state> <type> <public-ip>" for the live ber-gpu instance, or nothing
  $A ec2 describe-instances --filters "Name=tag:Name,Values=$NAME" \
    "Name=instance-state-name,Values=pending,running,stopping,shutting-down" \
    --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType,PublicIpAddress]' --output text
}

volume_id() {
  $A ec2 describe-volumes --filters "Name=tag:Name,Values=$NAME-workspace" --query 'Volumes[0].VolumeId' --output text
}

need_host() {
  [ -f "$HOST_FILE" ] || { echo "no $HOST_FILE — run scripts/start_gpu.sh first"; exit 1; }
  source "$HOST_FILE"
}

gssh() { need_host; ssh $SSH_OPTS "$SSH_USER@$GPU_IP" "$@"; }
