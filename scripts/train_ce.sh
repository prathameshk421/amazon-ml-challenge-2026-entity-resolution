#!/bin/bash
# Runs ON the GPU box. One cross-encoder experiment = one folder /workspace/experiments/<EXP_ID>/ with
# config.json, environment.txt, git_commit.txt, train.log, metrics.json, checkpoint/ (model/ + train_meta.json).
# Trainer: ce/train.py (BCE on DEV_REST_180K pairs, 5% S1-hash held-out, early stop on held-out logloss).
#
#   scripts/train_ce.sh EXP001_ce_baseline --smoke-test
#   scripts/train_ce.sh EXP001_ce_baseline --model BAAI/bge-reranker-base --epochs 1 --batch-size 64 [--yes]
#
# Flags: --model --epochs --batch-size --max-steps --max-samples --seed --max-length --lr --precision
#        --output-dir (default /workspace/experiments/<EXP_ID>) --smoke-test (= --max-steps 50 --max-samples 20000)
#        --yes (required when the estimate exceeds 0.5 h). Run inside tmux so SSH drops do not kill it.
# Output is a CE checkpoint, not a submission: it becomes submittable after scoring the matcher sets and a
# matcher retrain with p_ce (plan/TRANSFORMER_PLAN.md §4-5).
set -euo pipefail
EXP=${1:?EXP_ID}; shift
MODEL=cross-encoder/ms-marco-MiniLM-L6-v2; EPOCHS=1; BATCH=64; MAX_STEPS=0; LIMIT=0; SEED=42; MAXLEN=128
LR=3e-5; PREC=auto; OUT=""; YES=0; SMOKE=0
while [ $# -gt 0 ]; do
  case $1 in
    --model) MODEL=$2; shift ;; --epochs) EPOCHS=$2; shift ;; --batch-size) BATCH=$2; shift ;;
    --max-steps) MAX_STEPS=$2; shift ;; --max-samples) LIMIT=$2; shift ;; --seed) SEED=$2; shift ;;
    --max-length) MAXLEN=$2; shift ;; --lr) LR=$2; shift ;; --precision) PREC=$2; shift ;;
    --output-dir) OUT=$2; shift ;; --smoke-test) SMOKE=1 ;; --yes) YES=1 ;;
    *) echo "unknown flag $1"; exit 2 ;;
  esac; shift
done
if [ $SMOKE = 1 ]; then [ "$MAX_STEPS" = 0 ] && MAX_STEPS=50; [ "$LIMIT" = 0 ] && LIMIT=20000; fi
source /workspace/env.sh
OUT=${OUT:-/workspace/experiments/$EXP}
[ -e "$OUT/metrics.json" ] && { echo "$OUT already has metrics.json — pick a new EXP_ID"; exit 1; }
mkdir -p "$OUT"

# Pre-run banner: size, throughput (last measured run of this model on this GPU, else a labelled guess), cost.
python - "$OUT" "$MODEL" "$EPOCHS" "$BATCH" "$MAX_STEPS" "$LIMIT" "$MAXLEN" "$YES" <<'EOF' || exit 1
import glob, json, sys, torch, pyarrow.parquet as pq
out, model, epochs, batch, max_steps, limit, maxlen, yes = sys.argv[1:]
epochs, batch, max_steps, limit, yes = float(epochs), int(batch), int(max_steps), int(limit), yes == "1"
rows = pq.ParquetFile("/workspace/data/ce_pairs/pairs_DEV_REST_180K.parquet").metadata.num_rows
n = min(rows, limit) if limit else rows
train = int(n * 0.95)
steps = int(-(-train // batch) * epochs)
steps = min(steps, max_steps) if max_steps else steps
gpu = torch.cuda.get_device_name(0)
meas = [json.load(open(f)) for f in sorted(glob.glob("/workspace/experiments/*/metrics.json"))]
meas = [m for m in meas if m.get("model") == model and m.get("gpu") == gpu and m.get("train_pairs_per_s")]
if meas:
    pps, src = meas[-1]["train_pairs_per_s"], "(a) measured in an earlier run"
else:
    pps, src = {"A10G": 500, "L4": 400, "L40S": 1100, "T4": 193}.get(
        next((k for k in ["A10G", "L4", "L40S", "T4"] if k in gpu), ""), 300), \
        "(d) guess for bge-reranker-base-size models; T4 193/s is (a) ce_rec_base_probe_k0.json"
price = float(open("/workspace/project/cloud/gpu.env").read().split("PRICE_PER_HOUR=")[1].split()[0])
hours = steps * batch / pps / 3600 + 0.1  # + tokenization/eval/save overhead
cfg = {"model": model, "train_set": "DEV_REST_180K", "pairs_in_file": rows, "train_pairs_used": train,
       "epochs": epochs, "batch": batch, "max_steps": max_steps, "max_samples": limit, "max_length": int(maxlen),
       "planned_steps": steps, "gpu": gpu, "est_pairs_per_s": pps, "est_source": src,
       "est_hours": round(hours, 2), "est_cost_usd": round(hours * price, 2), "price_per_hour": price}
print("=" * 72)
for k in ["gpu", "model", "train_set", "train_pairs_used", "epochs", "batch", "max_length", "planned_steps",
          "est_pairs_per_s", "est_source", "est_hours", "est_cost_usd"]:
    print(f"  {k:18s} {cfg[k]}")
print("=" * 72)
json.dump(cfg, open(f"{out}/estimate.json", "w"), indent=1)
if hours > 0.5 and not yes:
    print(f"estimate {hours:.2f} h > 0.5 h: re-run with --yes to launch"); sys.exit(1)
EOF

cat > "$OUT/config.json" <<EOF
{"exp_id": "$EXP", "trainer": "ce/train.py", "model": "$MODEL", "epochs": $EPOCHS, "batch_size": $BATCH,
 "max_steps": $MAX_STEPS, "max_samples": $LIMIT, "seed": $SEED, "max_length": $MAXLEN, "lr": $LR,
 "precision": "$PREC", "smoke_test": $SMOKE, "pairs": "/workspace/data/ce_pairs/pairs_DEV_REST_180K.parquet",
 "started_utc": "$(date -u +%FT%TZ)"}
EOF
cp /workspace/project/CODE_VERSION "$OUT/git_commit.txt" 2>/dev/null || echo "unknown (no CODE_VERSION)" > "$OUT/git_commit.txt"
{ nvidia-smi; python -V; pip freeze; } > "$OUT/environment.txt" 2>&1

cd /workspace/project/ce
python -u train.py --pairs /workspace/data/ce_pairs --model "$MODEL" --out "$OUT/checkpoint" \
  --epochs "$EPOCHS" --batch "$BATCH" --max-steps "$MAX_STEPS" --limit "$LIMIT" --seed "$SEED" \
  --max-length "$MAXLEN" --lr "$LR" --precision "$PREC" 2>&1 | tee "$OUT/train.log"
cp "$OUT/checkpoint/train_meta.json" "$OUT/metrics.json"
echo "done: $OUT  (pull with scripts/sync_results.sh pull)"
