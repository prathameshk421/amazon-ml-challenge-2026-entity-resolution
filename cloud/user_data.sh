#!/bin/bash
# EC2 user-data (runs as root at first boot). Installs the two cost guards; the /workspace volume is attached and
# mounted afterwards by scripts/start_gpu.sh over SSH.
#   1. hard cap: power off __MAX_HOURS__ h after boot (instance-initiated shutdown = terminate).
#   2. idle watchdog: power off after __IDLE__ consecutive minutes with no GPU compute process, no logged-in
#      user and 1-min load average < 1.0.
# Cancel the hard cap for one boot with `sudo shutdown -c` (then the watchdog is the only guard).
shutdown -h +$((__MAX_HOURS__ * 60)) "ber-gpu hard cap __MAX_HOURS__ h"

cat > /usr/local/bin/gpu-idle-watchdog <<'EOF'
#!/bin/bash
IDLE=__IDLE__
idle=0
sleep 600  # grace period for boot + setup
while sleep 60; do
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  users=$(who | grep -c .)
  load=$(awk '{print ($1 >= 1.0)}' /proc/loadavg)
  if [ "$gpu" -gt 0 ] || [ "$users" -gt 0 ] || [ "$load" -eq 1 ]; then idle=0; else idle=$((idle + 1)); fi
  if [ "$idle" -ge "$IDLE" ]; then
    logger -t gpu-idle-watchdog "idle ${IDLE} min, powering off"
    sync; umount /workspace 2>/dev/null
    shutdown -h now
  fi
done
EOF
chmod +x /usr/local/bin/gpu-idle-watchdog
nohup /usr/local/bin/gpu-idle-watchdog >/var/log/gpu-idle-watchdog.log 2>&1 &
