#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <entity/project/sweep_id> [runs_per_agent]"
  exit 1
fi

SWEEP_ID="$1"
RUNS_PER_AGENT="${2:-0}"
GPUS=(0 1 2 3 4 5 6 7)

for gpu in "${GPUS[@]}"; do
  agent_cmd=(wandb agent "$SWEEP_ID")
  if [[ "$RUNS_PER_AGENT" -gt 0 ]]; then
    agent_cmd=(wandb agent --count "$RUNS_PER_AGENT" "$SWEEP_ID")
  fi

  mkdir -p "wandb_gpu${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  WANDB_DIR="./wandb_gpu${gpu}" \
  nohup "${agent_cmd[@]}" > "agent_gpu${gpu}.log" 2>&1 &
done

wait
