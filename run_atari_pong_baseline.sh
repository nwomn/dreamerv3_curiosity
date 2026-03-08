#!/bin/bash

# Baseline (无 FEP) 在 Atari Pong 上的对比
# GPU: 3
# 预计时间: 1-2 小时

export CUDA_VISIBLE_DEVICES=3

embodied-run \
  --script train \
  --configs atari \
  --task atari_pong \
  --agent.fep.enabled False \
  --logdir ./runs/baseline_atari/pong \
  --run.steps 400000 \
  2>&1 | tee ./runs/baseline_atari_pong.log

