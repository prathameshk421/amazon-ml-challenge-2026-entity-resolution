# GPU box on AWS EC2 (Spot) — runbook

One Spot GPU instance at a time, in `ap-south-1a`, from the pinned DLAMI (`cloud/gpu.env`). Everything that must
survive lives on one 100 GB gp3 EBS volume mounted at `/workspace`; the instance itself is disposable and
terminates on stop, Spot interruption, `MAX_HOURS` (8 h) after boot, or `IDLE_MINUTES` (30) with no GPU process,
no SSH login and load < 1 (`cloud/user_data.sh`). Code and data go over rsync (no git push needed).

```
/workspace/  project/ (rsync of ce/ cloud/ scripts/)  data/ce_pairs/  venv/  hf_cache/  models/
             checkpoints/  experiments/<EXP_ID>/  logs/  env.sh
```

| step | command (local unless noted) |
|---|---|
| one-time | `scripts/start_gpu.sh init` — SSH key `~/.ssh/ber_gpu_ed25519`, SG `ber-gpu-ssh` (your IP/32), volume `ber-workspace` |
| start + verify | `scripts/start_gpu.sh` — launch, attach/mount, `sync_results.sh push`, `setup_gpu.sh`, `check_gpu.sh` |
| SSH | printed by start (`logs/gpu_host.env` holds the IP) |
| verify again | `scripts/check_gpu.sh --remote` |
| smoke test (box) | `python cloud/smoke_test.py` → `/workspace/experiments/SMOKE/metrics.json` |
| CE smoke (box) | `scripts/train_ce.sh EXP000_ce_smoke --smoke-test` |
| CE run (box, in tmux) | `scripts/train_ce.sh EXP001_ce_baseline --model <hub id> --epochs 1 --batch-size 64 --yes` |
| push code again | `scripts/sync_results.sh push` |
| pull results | `scripts/sync_results.sh pull [--with-weights]` → `work/gpu/` |
| stop | `scripts/stop_gpu.sh` — pull, unmount, terminate, print session cost |

`train_ce.sh` prints GPU / train pairs / epochs / batch / max length / estimated hours and cost before every run and
refuses anything estimated over 0.5 h without `--yes`.
