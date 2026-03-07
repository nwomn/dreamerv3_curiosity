#!/bin/bash

# Auto-start Crafter Baseline after FEP training completes
# Usage: bash scripts/auto_start_baseline.sh

set -e

# Configuration
FEP_METRICS="/9950backfile/liguoqi/brainvlm/rzq/dreamerv3_curiosity/runs/fep_crafter/metrics.jsonl"
TARGET_STEPS=1100000
CHECK_INTERVAL=30  # Check every 30 seconds
GPU_ID=1

# Baseline training config
BASELINE_LOGDIR="runs/baseline_crafter"
BASELINE_LOG="${BASELINE_LOGDIR}.log"

echo "========================================"
echo "Auto-Start Baseline Training Script"
echo "========================================"
echo ""
echo "Monitoring GPU: $GPU_ID"
echo "Target steps: $TARGET_STEPS"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

# Function to get current step from metrics
get_current_step() {
    if [ ! -f "$FEP_METRICS" ]; then
        echo "0"
        return
    fi

    tail -1 "$FEP_METRICS" | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    print(int(data.get('step', 0)))
except:
    print(0)
" 2>/dev/null || echo "0"
}

# Function to check GPU utilization
get_gpu_util() {
    nvidia-smi -i $GPU_ID --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0"
}

# Function to check GPU memory usage
get_gpu_memory() {
    nvidia-smi -i $GPU_ID --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0"
}

# Function to check if FEP training process is running
is_fep_running() {
    pgrep -f "python.*fep_crafter" > /dev/null 2>&1
    return $?
}

# Main monitoring loop
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting monitoring..."
echo ""

IDLE_COUNT=0
IDLE_THRESHOLD=3  # Need 3 consecutive idle checks (90s total)

while true; do
    current_step=$(get_current_step)
    progress=$(python3 -c "print(f'{$current_step/$TARGET_STEPS*100:.1f}')" 2>/dev/null || echo "0.0")

    # Check if process is running
    if is_fep_running; then
        process_status="Running"
    else
        process_status="Stopped"
    fi

    # Output status
    echo "[$(date '+%H:%M:%S')] Step: ${current_step}/${TARGET_STEPS} (${progress}%) | Process: ${process_status}"

    # Check if FEP training process has stopped
    if ! is_fep_running; then
        IDLE_COUNT=$((IDLE_COUNT + 1))
        echo "  -> Training process stopped (${IDLE_COUNT}/${IDLE_THRESHOLD})"

        if [ $IDLE_COUNT -ge $IDLE_THRESHOLD ]; then
            echo ""
            echo "========================================"
            echo "✓ FEP Training Completed!"
            echo "========================================"
            echo "Final step: ${current_step}"
            echo ""
            break
        fi
    else
        # Reset idle count if process is still running
        IDLE_COUNT=0
    fi

    # Wait before next check
    sleep $CHECK_INTERVAL
done

# Start baseline training
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Preparing to start baseline training..."
echo ""

# Wait a bit to ensure GPU is fully released
sleep 10

# Check if baseline already running
if pgrep -f "python.*baseline_crafter" > /dev/null 2>&1; then
    echo "⚠ Baseline training already running. Exiting."
    exit 0
fi

# Create logdir if needed
mkdir -p "$BASELINE_LOGDIR"

# Start baseline training
echo "========================================"
echo "Starting Crafter Baseline Training"
echo "========================================"
echo ""
echo "GPU: $GPU_ID"
echo "Logdir: $BASELINE_LOGDIR"
echo "Log file: $BASELINE_LOG"
echo ""

cd /9950backfile/liguoqi/brainvlm/rzq/dreamerv3_curiosity

CUDA_VISIBLE_DEVICES=$GPU_ID python -u dreamerv3/main.py \
  --configs crafter \
  --agent.fep.enabled False \
  --jax.train_devices 0 \
  --jax.policy_devices 0 \
  --logdir "$BASELINE_LOGDIR" \
  > "$BASELINE_LOG" 2>&1 &

BASELINE_PID=$!

echo "✓ Baseline training started!"
echo "PID: $BASELINE_PID"
echo ""
echo "Monitor progress with:"
echo "  tail -f $BASELINE_LOG"
echo ""
echo "Check metrics with:"
echo "  tail -1 ${BASELINE_LOGDIR}/metrics.jsonl | python3 -m json.tool"
echo ""
echo "========================================"
echo "All Done!"
echo "========================================"
