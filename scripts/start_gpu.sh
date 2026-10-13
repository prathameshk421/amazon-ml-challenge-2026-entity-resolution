#!/bin/bash
# Local machine. Start the GPU box and leave it verified and ready for SSH.
#
#   scripts/start_gpu.sh init     # once: SSH key pair, security group (SSH from this IP only), /workspace gp3 volume
#   scripts/start_gpu.sh          # launch one Spot instance in $AZ, attach + mount /workspace, push code/data,
#                                 # run setup_gpu.sh and check_gpu.sh on it, print the ssh command
#   ITYPES=g6e.xlarge scripts/start_gpu.sh   # force a type (here L40S 48 GB)
#
# Cost guards on the box (cloud/user_data.sh): hard power-off after MAX_HOURS, idle power-off after IDLE_MINUTES.
# Spot, one-time, terminate on interruption: nothing but the /workspace volume survives the instance.
set -euo pipefail
source "$(dirname "$0")/../cloud/lib.sh"
log_to gpu_start
STEP=${1:-launch}

my_ip() { curl -s https://checkip.amazonaws.com | tr -d '[:space:]'; }

sg_id() {
  $A ec2 describe-security-groups --filters "Name=group-name,Values=$NAME-ssh" --query 'SecurityGroups[0].GroupId' --output text
}

allow_my_ip() {
  local ip; ip=$(my_ip)
  $A ec2 authorize-security-group-ingress --group-id "$(sg_id)" --protocol tcp --port 22 --cidr "$ip/32" >/dev/null 2>&1 \
    && echo "SSH allowed from $ip/32" || echo "SSH already allowed from $ip/32"
}

if [ "$STEP" = init ]; then
  [ -f "$KEY_FILE" ] || ssh-keygen -t ed25519 -N "" -C "$NAME" -f "$KEY_FILE"
  $A ec2 describe-key-pairs --key-names "$NAME" >/dev/null 2>&1 || \
    $A ec2 import-key-pair --key-name "$NAME" --public-key-material "fileb://$KEY_FILE.pub" --query KeyName --output text
  if [ "$(sg_id)" = None ]; then
    VPC=$($A ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
    $A ec2 create-security-group --group-name "$NAME-ssh" --description "ber-gpu SSH from owner IP" --vpc-id "$VPC" \
      --query GroupId --output text
  fi
  allow_my_ip
  if [ "$(volume_id)" = None ]; then
    V=$($A ec2 create-volume --availability-zone "$AZ" --size "$VOL_SIZE_GB" --volume-type gp3 \
      --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=$NAME-workspace}]" --query VolumeId --output text)
    $A ec2 wait volume-available --volume-ids "$V"
  fi
  echo "init done: key=$NAME sg=$(sg_id) volume=$(volume_id) ($VOL_SIZE_GB GB gp3, $AZ)"
  exit 0
fi

LIVE=$(live_instance)
if [ -n "$LIVE" ]; then echo "already live: $LIVE — use it or scripts/stop_gpu.sh first"; exit 1; fi
VOL=$(volume_id); [ "$VOL" != None ] || { echo "no workspace volume — run: scripts/start_gpu.sh init"; exit 1; }
allow_my_ip

UD=$(mktemp)
sed -e "s/__MAX_HOURS__/$MAX_HOURS/g" -e "s/__IDLE__/$IDLE_MINUTES/g" "$ROOT/cloud/user_data.sh" > "$UD"
IID=""
for T in $ITYPES; do
  echo "trying Spot $T in $AZ (max \$$MAX_SPOT_PRICE/h)"
  IID=$($A ec2 run-instances --image-id "$AMI_ID" --instance-type "$T" --count 1 \
    --placement "AvailabilityZone=$AZ" --key-name "$NAME" --security-group-ids "$(sg_id)" \
    --instance-market-options "{\"MarketType\":\"spot\",\"SpotOptions\":{\"MaxPrice\":\"$MAX_SPOT_PRICE\",\"SpotInstanceType\":\"one-time\",\"InstanceInterruptionBehavior\":\"terminate\"}}" \
    --instance-initiated-shutdown-behavior terminate --metadata-options HttpTokens=required \
    --user-data "file://$UD" --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
    --query 'Instances[0].InstanceId' --output text) && break || { echo "  $T unavailable"; IID=""; }
done
rm -f "$UD"
[ -n "$IID" ] || { echo "no Spot capacity for: $ITYPES"; exit 1; }
echo "instance $IID ($T) starting — billing from now"
$A ec2 wait instance-running --instance-ids "$IID"
IP=$($A ec2 describe-instances --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
printf 'GPU_ID=%s\nGPU_IP=%s\nGPU_TYPE=%s\nGPU_STARTED=%s\n' "$IID" "$IP" "$T" "$(date -u +%FT%TZ)" > "$HOST_FILE"

$A ec2 attach-volume --volume-id "$VOL" --instance-id "$IID" --device /dev/sdf --query State --output text
$A ec2 wait volume-in-use --volume-ids "$VOL"

echo "waiting for SSH on $IP"
rm -f "$ROOT/logs/gpu_known_hosts"  # every launch is a new host
for i in $(seq 60); do ssh $SSH_OPTS -o ConnectTimeout=5 "$SSH_USER@$IP" true 2>/dev/null && break; sleep 5; done
gssh "sudo bash -s" <<EOF
set -e
D=/dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_${VOL/-/}
for i in \$(seq 30); do [ -e \$D ] && break; sleep 2; done
[ -e \$D ] || { echo "volume device \$D not found"; exit 1; }
blkid \$D >/dev/null || { echo "first use: formatting /workspace volume"; mkfs.ext4 -q -L workspace \$D; }
mkdir -p /workspace && mount \$D /workspace && chown $SSH_USER:$SSH_USER /workspace
df -h /workspace | tail -1
EOF

bash "$ROOT/scripts/sync_results.sh" push
gssh "bash /workspace/project/scripts/setup_gpu.sh && bash /workspace/project/scripts/check_gpu.sh"
echo
echo "READY: ssh $SSH_OPTS $SSH_USER@$IP"
echo "       (or: scripts/check_gpu.sh --remote) — stop with scripts/stop_gpu.sh"
